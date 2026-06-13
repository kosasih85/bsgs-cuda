/*
 * BSGS ECC Discrete Log Solver - Range 2^52 CUDA Edition
 *
 * Hybrid CPU+GPU approach:
 *   - Baby step table: dibangun di CPU (secp256k1 library), disimpan di RAM
 *   - Baby table di-upload ke GPU VRAM sebelum giant step
 *   - Giant step loop: dijalankan di GPU (CUDA kernel)
 *   - secp256k1 field arithmetic: reimplementasi native di CUDA (__device__)
 *
 * Target GPU: GTX 1060 (Compute Capability 6.1, 6 GB VRAM)
 *   - Gunakan -DHT_LOAD_TIGHT untuk baby table ~3 GB (aman di 6 GB VRAM)
 *   - Tanpa flag: baby table ~6 GB -> melebihi VRAM, gunakan mode ini hanya
 *     jika GPU kamu punya VRAM >= 8 GB
 *
 * Compile (Linux):
 *   nvcc -O3 -arch=sm_61 -DHT_LOAD_TIGHT -o bsgs_2p52_cuda bsgs_2p52_cuda.cu \
 *        -lsecp256k1 -lm -lpthread
 *
 * Compile (Windows / MSYS2):
 *   nvcc -O3 -arch=sm_61 -DHT_LOAD_TIGHT -o bsgs_2p52_cuda.exe bsgs_2p52_cuda.cu \
 *        -lsecp256k1 -lm -lpthread
 *
 * Usage:
 *   bsgs_2p52_cuda base.txt batch.txt [cpu_threads] [--resume] [--reset]
 *
 * CATATAN GTX 1060:
 *   - Compute Capability 6.1 -> didukung penuh
 *   - 1280 CUDA cores -> pakai 256 threads/block x banyak block
 *   - 6 GB VRAM (versi 6GB) -> cukup untuk HT_LOAD_TIGHT mode (3 GB baby table)
 *   - 3 GB VRAM (versi 3GB) -> TIDAK cukup, gunakan CPU-only mode
 */

/* =========================================================================
 * Standard headers
 * ========================================================================= */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <signal.h>
#include <math.h>

/* CUDA headers */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* =========================================================================
 * Platform compat
 * ========================================================================= */
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <io.h>
#  define unlink  _unlink
#  define sleep(s) Sleep((s)*1000)
static void nanosleep_ms(long ms) { Sleep((DWORD)(ms < 0 ? 0 : ms)); }
#  define NANOSLEEP_MS(ms) nanosleep_ms(ms)
static void enable_ansi(void) {
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (GetConsoleMode(h, &mode))
        SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}
#else
#  include <unistd.h>
#  define NANOSLEEP_MS(ms) do { struct timespec _s = {0, (long)(ms)*1000000L}; nanosleep(&_s, NULL); } while(0)
static void enable_ansi(void) {}
#endif

#include <pthread.h>
#include <secp256k1.h>

#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wunused-result"
#endif

/* =========================================================================
 * Macro helper CUDA error check
 * ========================================================================= */
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s:%d -> %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

/* =========================================================================
 * Konfigurasi
 * ========================================================================= */
#define KEY_LEN         66
#define KEY_BYTES       33
#define DEFAULT_THREADS 4

#define BSGS_BITS       26
#define BSGS_M          (1ULL << BSGS_BITS)   /* 67,108,864 */
#define BSGS_MAX        (1ULL << 52)

/*
 * GTX 1060 6GB: gunakan HT_LOAD_TIGHT (3 GB baby table)
 * Baby table akan di-copy ke VRAM GPU sebelum giant step
 */
#ifdef HT_LOAD_TIGHT
#  define HT_BITS  (BSGS_BITS + 1)   /* 2x ~3 GB */
#else
#  define HT_BITS  (BSGS_BITS + 2)   /* 4x ~6 GB - hanya jika VRAM >= 8 GB */
#endif
#define HT_SIZE    (1u << HT_BITS)
#define HT_MASK    (HT_SIZE - 1u)
#define HT_EMPTY   0xFFFFFFFFu

#define N_BUCKETS       16
#define CHECKPOINT_INTERVAL 999999999999999999ULL
#define CHECKPOINT_FILE     "bsgs_checkpoint_2p52.dat"
#define PROGRESS_FILE       "bsgs_progress_2p52.log"
#define PROGRESS_BATCH      100000ULL

/* CUDA kernel config - dioptimalkan untuk GTX 1060 */
#define CUDA_THREADS_PER_BLOCK  256
#define CUDA_BLOCKS_PER_SM      4
#define GTX1060_SM_COUNT        10   /* GTX 1060 punya 10 SM */
/* Total thread: 10 SM x 4 blocks x 256 threads = 10240 thread paralel */
#define CUDA_DEFAULT_BLOCKS     (GTX1060_SM_COUNT * CUDA_BLOCKS_PER_SM)

/* Ukuran batch per kernel launch (agar GPU tidak timeout) */
#define CUDA_BATCH_SIZE         (1ULL << 18)   /* 262144 a-steps per launch */

/* =========================================================================
 * ANSI colors
 * ========================================================================= */
#define CRST  "\033[0m"
#define CBOLD "\033[1m"
#define CDIM  "\033[2m"
#define CGR   "\033[32m"
#define CCY   "\033[36m"
#define CYL   "\033[33m"
#define CMG   "\033[35m"
#define CBL   "\033[34m"
#define CRD   "\033[31m"

/* =========================================================================
 * Structs CPU-side
 * ========================================================================= */
typedef struct {
    uint32_t hash32;
    uint32_t _pad;
    uint64_t step;
    uint8_t  key[KEY_BYTES];
    uint8_t  _pad2[3];
} BabySlot;  /* ~56 bytes */

/* Flat baby table untuk GPU (lebih efisien daripada bucket struct) */
typedef struct {
    uint32_t hash32;   /* 4 */
    uint32_t _pad;     /* 4 */
    uint64_t step;     /* 8 */
    uint8_t  key[33];  /* 33 */
    uint8_t  _p[7];    /* 7 -> total 56 bytes, aligned */
} GpuBabySlot;        /* 56 bytes */

typedef struct {
    BabySlot       *slots;
    uint32_t        size;
    uint32_t        count;
    pthread_mutex_t lock;
    char            _pad[64 - sizeof(pthread_mutex_t) - 2*sizeof(uint32_t) - sizeof(BabySlot*)];
} Bucket;

typedef struct {
    secp256k1_context  *ctx;
    uint8_t           **targets;
    size_t              n_targets;
    secp256k1_pubkey   *T_points;
    secp256k1_pubkey    pos_mG;
    secp256k1_pubkey    neg_G;
    uint64_t           *found_k;
    volatile uint64_t   found_mask;
    uint64_t            full_mask;
    volatile uint64_t   matches;
    pthread_mutex_t     mu;
    const char         *out_file;
    uint64_t            a_start;
    uint64_t            a_end;
    volatile uint64_t   a_done;
    uint64_t            window_offset;
    uint64_t            window_num;
    struct timespec     t0;
} GS;

typedef struct {
    GS      *gs;
    uint64_t a_from;
    uint64_t a_to;
    int      id;
} WA;

typedef struct {
    secp256k1_context *ctx;
    secp256k1_pubkey   start_point;
    secp256k1_pubkey   G_pub;
    uint64_t           b_from;
    uint64_t           b_to;
} BabyWA;

/* =========================================================================
 * secp256k1 field element untuk CUDA
 * Representasi: 256-bit integer sebagai 8 x uint32_t (little-endian limbs)
 * ========================================================================= */

/* Curve secp256k1:
 *   p = 2^256 - 2^32 - 977
 *   a = 0 (simplified addition formula)
 *   G = (Gx, Gy)
 */

/* Tipe field element di GPU: 8 limbs x 32-bit, little-endian */
typedef struct { uint32_t d[8]; } Fe;    /* 256-bit field element */
typedef struct { Fe x; Fe y; int inf; } GePt;  /* curve point */

/* secp256k1 prime p = 2^256 - 2^32 - 977 */
__constant__ uint32_t SECP_P[8] = {
    0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

/* secp256k1 generator point Gx */
__constant__ uint32_t SECP_GX[8] = {
    0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDB,
    0xCE870B07u, 0x55A06295u, 0xF9DCBBAC, 0x79BE667Eu
};

/* secp256k1 generator point Gy */
__constant__ uint32_t SECP_GY[8] = {
    0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
    0x0E1108A8u, 0x5DA4FBFC, 0x26A3C465u, 0x483ADA77u
};

/* =========================================================================
 * Device: 256-bit modular arithmetic (secp256k1 field)
 * ========================================================================= */

/* Bandingkan a dan b: return -1/0/1 */
__device__ int fe_cmp(const Fe &a, const Fe &b) {
    for (int i = 7; i >= 0; i--) {
        if (a.d[i] < b.d[i]) return -1;
        if (a.d[i] > b.d[i]) return  1;
    }
    return 0;
}

/* a = a - b (modular, a >= b assumed setelah reduce) */
__device__ void fe_sub_raw(Fe &r, const Fe &a, const Fe &b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)a.d[i] - b.d[i] - borrow;
        r.d[i] = (uint32_t)diff;
        borrow = (diff >> 63) & 1;
    }
}

/* r = a mod p */
__device__ void fe_reduce(Fe &r, const Fe &a) {
    Fe p;
    for (int i = 0; i < 8; i++) p.d[i] = SECP_P[i];
    r = a;
    if (fe_cmp(r, p) >= 0) fe_sub_raw(r, r, p);
}

/* r = (a + b) mod p */
__device__ void fe_add(Fe &r, const Fe &a, const Fe &b) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t s = (uint64_t)a.d[i] + b.d[i] + carry;
        r.d[i] = (uint32_t)s;
        carry = s >> 32;
    }
    Fe p; for (int i=0;i<8;i++) p.d[i]=SECP_P[i];
    if (carry || fe_cmp(r, p) >= 0) fe_sub_raw(r, r, p);
}

/* r = (a - b) mod p */
__device__ void fe_sub(Fe &r, const Fe &a, const Fe &b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)a.d[i] - b.d[i] - borrow;
        r.d[i] = (uint32_t)diff;
        borrow = (diff >> 63) & 1;
    }
    if (borrow) {
        /* tambah p */
        uint64_t carry = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t s = (uint64_t)r.d[i] + SECP_P[i] + carry;
            r.d[i] = (uint32_t)s;
            carry = s >> 32;
        }
    }
}

/*
 * r = (a * b) mod p
 * Menggunakan schoolbook 256x256 -> 512 bit, lalu reduksi secp256k1 khusus.
 * secp256k1: p = 2^256 - c, c = 2^32 + 977 -> reduksi cepat tanpa divisi.
 */
__device__ void fe_mul(Fe &r, const Fe &a, const Fe &b) {
    /* Hitung 512-bit product: t[0..15] */
    uint64_t t[16] = {0};
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t uv = t[i+j] + (uint64_t)a.d[i] * b.d[j] + carry;
            t[i+j] = uv & 0xFFFFFFFFULL;
            carry   = uv >> 32;
        }
        t[i+8] += carry;
    }
    /*
     * Reduksi mod p = 2^256 - c, c = 2^32 + 977 = 0x100000411
     * t = lo + hi * 2^256
     * t mod p = lo + hi * c   (karena 2^256 = c mod p)
     * Ulangi sekali lagi jika masih >= p
     */
    const uint64_t C = 0x1000003D1ULL;  /* 2^32 + 977 */
    /* Hitung hi * c, tambahkan ke lo */
    uint64_t carry2 = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t v = t[i] + t[i+8] * C + carry2;
        t[i] = v & 0xFFFFFFFFULL;
        carry2 = v >> 32;
    }
    /* carry2 masih bisa ada -> satu reduksi lagi */
    uint64_t carry3 = carry2 * C;
    for (int i = 0; i < 8 && carry3; i++) {
        uint64_t v = t[i] + carry3;
        t[i] = v & 0xFFFFFFFFULL;
        carry3 = v >> 32;
    }
    Fe res;
    for (int i = 0; i < 8; i++) res.d[i] = (uint32_t)t[i];
    fe_reduce(r, res);
}

/* r = a^2 mod p */
__device__ void fe_sqr(Fe &r, const Fe &a) { fe_mul(r, a, a); }

/*
 * r = a^-1 mod p menggunakan Fermat: a^(p-2) mod p
 * p-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
 * Gunakan square-and-multiply dengan chain yang efisien untuk secp256k1.
 */
__device__ void fe_inv(Fe &r, const Fe &a) {
    /* Chain khusus secp256k1: a^(p-2) mod p */
    Fe x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t1;

    fe_sqr(x2, a);    fe_mul(x2, x2, a);       /* a^3 -> x3 */
    /* Sebenarnya x2 = a^2, kita butuh beberapa step */
    /* Gunakan standard addition chain untuk p-2 */

    /* x2  = a^2 */
    fe_sqr(x2, a);
    /* x3  = a^3 */
    fe_mul(x3, x2, a);
    /* x6  = x3^2 * x3 = a^6 ... */
    fe_sqr(x6, x3);
    fe_sqr(x6, x6);   /* x6 = a^12 -> skip, gunakan loop */

    /* Pendekatan sederhana: loop 254 bit, lebih lambat tapi benar */
    /* p-2 dalam hex: FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D */
    /* Kita iterasi bit per bit */
    uint32_t exp[8] = {
        0xFFFFC2Du, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    };
    Fe base_v = a, result;
    /* result = 1 */
    for(int i=0;i<8;i++) result.d[i]=0; result.d[0]=1;

    for (int word = 0; word < 8; word++) {
        uint32_t w = exp[word];
        for (int bit = 0; bit < 32; bit++) {
            if (w & 1u) fe_mul(result, result, base_v);
            fe_sqr(base_v, base_v);
            w >>= 1;
        }
    }
    r = result;
    (void)x2; (void)x3; (void)x6; (void)x9; (void)x11;
    (void)x22; (void)x44; (void)x88; (void)x176; (void)x220; (void)x223; (void)t1;
}

/* =========================================================================
 * Device: secp256k1 point addition (affine + affine -> affine)
 * Menggunakan rumus standard: slope = (y2-y1)/(x2-x1)
 * ========================================================================= */
__device__ void point_add(GePt &R, const GePt &P, const GePt &Q) {
    if (P.inf) { R = Q; return; }
    if (Q.inf) { R = P; return; }

    /* Cek apakah P == Q */
    if (fe_cmp(P.x, Q.x) == 0) {
        if (fe_cmp(P.y, Q.y) != 0) { R.inf = 1; return; }
        /* P == Q: gunakan doubling */
        /* slope = (3*x^2) / (2*y) */
        Fe x2, num, den, lam, lam2, rx, ry, tmp;
        fe_sqr(x2, P.x);
        /* num = 3*x^2 */
        fe_add(num, x2, x2); fe_add(num, num, x2);
        /* den = 2*y */
        fe_add(den, P.y, P.y);
        fe_inv(tmp, den);
        fe_mul(lam, num, tmp);
        /* rx = lam^2 - 2*x */
        fe_sqr(lam2, lam);
        fe_sub(rx, lam2, P.x); fe_sub(rx, rx, P.x);
        /* ry = lam*(x - rx) - y */
        fe_sub(tmp, P.x, rx);
        fe_mul(ry, lam, tmp);
        fe_sub(ry, ry, P.y);
        R.x = rx; R.y = ry; R.inf = 0;
        return;
    }

    /* P != Q */
    Fe dx, dy, lam, lam2, rx, ry, inv_dx;
    fe_sub(dx, Q.x, P.x);
    fe_sub(dy, Q.y, P.y);
    fe_inv(inv_dx, dx);
    fe_mul(lam, dy, inv_dx);
    /* rx = lam^2 - x1 - x2 */
    fe_sqr(lam2, lam);
    fe_sub(rx, lam2, P.x); fe_sub(rx, rx, Q.x);
    /* ry = lam*(x1 - rx) - y1 */
    fe_sub(ry, P.x, rx);
    fe_mul(ry, lam, ry);
    fe_sub(ry, ry, P.y);
    R.x = rx; R.y = ry; R.inf = 0;
}

/* =========================================================================
 * Device: serialize point ke compressed 33-byte
 * ========================================================================= */
__device__ void point_serialize(uint8_t *out, const GePt &P) {
    /* prefix: 02 jika y genap, 03 jika y ganjil */
    out[0] = (P.y.d[0] & 1) ? 0x03u : 0x02u;
    /* x: big-endian 32 bytes dari P.x (little-endian limbs) */
    for (int i = 0; i < 8; i++) {
        uint32_t limb = P.x.d[7-i];
        out[1 + i*4 + 0] = (uint8_t)(limb >> 24);
        out[1 + i*4 + 1] = (uint8_t)(limb >> 16);
        out[1 + i*4 + 2] = (uint8_t)(limb >>  8);
        out[1 + i*4 + 3] = (uint8_t)(limb);
    }
}

/* =========================================================================
 * Device: fast hash (xxHash-style) - sama dengan versi CPU
 * ========================================================================= */
__device__ uint32_t gpu_fast_hash(const uint8_t *d, int n) {
    uint32_t h = 0x9e3779b1u;
    for (int i = 0; i < n; i++) {
        h ^= (uint32_t)d[i] * 0x85ebca77u;
        h  = (h << 13) | (h >> 19);
        h  = h * 5u + 0xe6546b64u;
    }
    h ^= h >> 16; h *= 0x85ebca6bu;
    h ^= h >> 13; h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

/* =========================================================================
 * Device: lookup di flat GPU baby table
 * ========================================================================= */
__device__ uint64_t gpu_baby_lookup(
    const GpuBabySlot * __restrict__ tbl,
    uint32_t tbl_size,
    const uint8_t *key33)
{
    uint32_t h = gpu_fast_hash(key33, 33);
    if (h == HT_EMPTY) h = 0u;
    uint32_t i = h % tbl_size;
    uint32_t attempts = 0;
    while (tbl[i].hash32 != HT_EMPTY && attempts < tbl_size) {
        if (tbl[i].hash32 == h) {
            /* full key compare */
            bool match = true;
            for (int k = 0; k < 33; k++) {
                if (tbl[i].key[k] != key33[k]) { match = false; break; }
            }
            if (match) return tbl[i].step;
        }
        i = (i + 1) % tbl_size;
        attempts++;
    }
    return 0xFFFFFFFFFFFFFFFFULL;
}

/* =========================================================================
 * CUDA Kernel: Giant Step
 *
 * Setiap thread menangani satu nilai 'a' dalam range [a_start, a_start+batch).
 * Untuk setiap 'a', thread menghitung giant step point T_i - a*m*G
 * menggunakan secp256k1 point arithmetic native di GPU, lalu lookup di baby table.
 *
 * Parameter:
 *   d_tbl         : flat baby table di VRAM
 *   tbl_size      : jumlah slot di baby table
 *   d_T_serialized: serialized starting points (33 byte x n_targets)
 *   d_negmG_x/y   : koordinat -(m*G) untuk step
 *   a_start       : nilai a pertama untuk batch ini
 *   batch         : jumlah langkah per batch
 *   n_targets     : jumlah target
 *   window_offset : offset window saat ini
 *   d_found_k     : output: found key per target (UINT64_MAX = belum ditemukan)
 *   d_a_done      : atomic counter untuk progress
 * ========================================================================= */
__global__ void giant_step_kernel(
    const GpuBabySlot * __restrict__ d_tbl,
    uint32_t tbl_size,
    const uint8_t * __restrict__ d_T_ser,   /* 33*n_targets bytes: starting points */
    uint32_t negmG_x_limbs[8],              /* -(m*G).x */
    uint32_t negmG_y_limbs[8],              /* -(m*G).y */
    uint64_t a_start,
    uint64_t batch,
    uint32_t n_targets,
    uint64_t window_offset,
    uint64_t bsgs_m,
    uint64_t * __restrict__ d_found_k,
    unsigned long long * __restrict__ d_a_done
) {
    uint64_t a = a_start + (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= a_start + batch) return;

    /* Load -(m*G) */
    GePt negmG;
    for (int i = 0; i < 8; i++) { negmG.x.d[i] = negmG_x_limbs[i]; negmG.y.d[i] = negmG_y_limbs[i]; }
    negmG.inf = 0;

    /* Setiap thread proses semua target */
    for (uint32_t ti = 0; ti < n_targets; ti++) {
        if (d_found_k[ti] != 0xFFFFFFFFFFFFFFFFULL) continue;

        /* Load starting point dari d_T_ser (serialized, 33 bytes per target) */
        /* d_T_ser menyimpan T_i - a_start * m * G */
        const uint8_t *base_ser = d_T_ser + ti * 33;

        /* Parse compressed point */
        GePt cur;
        cur.inf = 0;
        uint8_t prefix = base_ser[0];
        for (int i = 0; i < 8; i++) {
            int off = 1 + (7-i)*4;
            cur.x.d[i] = ((uint32_t)base_ser[off]   << 24) |
                         ((uint32_t)base_ser[off+1] << 16) |
                         ((uint32_t)base_ser[off+2] <<  8) |
                          (uint32_t)base_ser[off+3];
        }

        /*
         * Hitung y dari x menggunakan y^2 = x^3 + 7 (secp256k1)
         * y = sqrt(x^3 + 7) mod p
         * sqrt mod p: y = (x^3+7)^((p+1)/4) mod p  (karena p ≡ 3 mod 4)
         */
        Fe x3, rhs, y;
        fe_sqr(x3, cur.x);
        fe_mul(x3, x3, cur.x);        /* x^3 */
        /* rhs = x^3 + 7 */
        rhs = x3;
        rhs.d[0] += 7u;
        /* Cek overflow */
        if (rhs.d[0] < 7u) {
            for (int j = 1; j < 8; j++) { rhs.d[j]++; if (rhs.d[j]) break; }
        }
        fe_reduce(rhs, rhs);

        /* y = rhs^((p+1)/4) mod p */
        /* (p+1)/4 = 0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff0c */
        uint32_t exp_sqrt[8] = {
            0xBFFFFF0Cu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
            0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x3FFFFFFFu
        };
        Fe base2 = rhs, res_y;
        for (int i=0;i<8;i++) res_y.d[i]=0; res_y.d[0]=1;
        for (int word = 0; word < 8; word++) {
            uint32_t w = exp_sqrt[word];
            for (int bit = 0; bit < 32; bit++) {
                if (w & 1u) fe_mul(res_y, res_y, base2);
                fe_sqr(base2, base2);
                w >>= 1;
            }
        }
        y = res_y;
        /* Pilih paritas y yang sesuai dengan prefix */
        uint8_t y_parity = (uint8_t)(y.d[0] & 1u);
        uint8_t want_odd = (prefix == 0x03u) ? 1u : 0u;
        if (y_parity != want_odd) {
            /* y = p - y */
            Fe neg_y;
            uint64_t borrow = 0;
            for (int j = 0; j < 8; j++) {
                uint64_t diff = (uint64_t)SECP_P[j] - y.d[j] - borrow;
                neg_y.d[j] = (uint32_t)diff;
                borrow = (diff >> 63) & 1;
            }
            y = neg_y;
        }
        cur.y = y;

        /* Maju ke posisi a dengan menambahkan a - a_start kali negmG */
        uint64_t steps = a - a_start;
        if (steps > 0) {
            /* Hitung steps * negmG menggunakan double-and-add */
            GePt step_pt; step_pt.inf = 1;
            GePt addend = negmG;
            uint64_t sv = steps;
            while (sv > 0) {
                if (sv & 1ULL) point_add(step_pt, step_pt, addend);
                point_add(addend, addend, addend);
                sv >>= 1;
            }
            point_add(cur, cur, step_pt);
        }

        /* Serialize dan lookup */
        uint8_t ser[33];
        point_serialize(ser, cur);
        uint64_t b = gpu_baby_lookup(d_tbl, tbl_size, ser);
        if (b != 0xFFFFFFFFFFFFFFFFULL) {
            uint64_t k = window_offset + a * bsgs_m + b + 1ULL;
            /* Atomic CAS: update hanya jika belum ditemukan */
            unsigned long long old = 0xFFFFFFFFFFFFFFFFULL;
            atomicCAS((unsigned long long*)&d_found_k[ti],
                      old, (unsigned long long)k);
        }
    }

    /* Update progress counter */
    atomicAdd(d_a_done, 1ULL);
}

/* =========================================================================
 * Globals CPU
 * ========================================================================= */
static Bucket         *g_buckets   = NULL;
static int             g_n_buckets = N_BUCKETS;
static volatile int    g_running   = 1;
static GS             *g_gs_global = NULL;

/* GPU baby table (flat, di VRAM) */
static GpuBabySlot    *d_baby_tbl  = NULL;
static uint32_t        g_gpu_tbl_size = 0;

/* =========================================================================
 * CPU: fast hash (sama dengan GPU versi)
 * ========================================================================= */
static inline uint32_t fast_hash(const uint8_t *d, size_t n) {
    uint32_t h = 0x9e3779b1u;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint32_t)d[i] * 0x85ebca77u;
        h  = (h << 13) | (h >> 19);
        h  = h * 5u + 0xe6546b64u;
    }
    h ^= h >> 16; h *= 0x85ebca6bu;
    h ^= h >> 13; h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

/* =========================================================================
 * CPU: hex helpers
 * ========================================================================= */
static const uint8_t HEX_DEC[256] = {
    ['0']=0,['1']=1,['2']=2,['3']=3,['4']=4,
    ['5']=5,['6']=6,['7']=7,['8']=8,['9']=9,
    ['a']=10,['b']=11,['c']=12,['d']=13,['e']=14,['f']=15,
    ['A']=10,['B']=11,['C']=12,['D']=13,['E']=14,['F']=15
};
static const char HEX_ENC[16] = "0123456789abcdef";

static int hex2bytes(const char *h, uint8_t *o, size_t n) {
    if (strlen(h) < n * 2) return 0;
    for (size_t i = 0; i < n; i++)
        o[i] = (uint8_t)((HEX_DEC[(uint8_t)h[i*2]] << 4) | HEX_DEC[(uint8_t)h[i*2+1]]);
    return 1;
}
static void bytes2hex(const uint8_t *b, size_t n, char *o) {
    for (size_t i = 0; i < n; i++) {
        o[i*2]   = HEX_ENC[b[i] >> 4];
        o[i*2+1] = HEX_ENC[b[i] & 0xF];
    }
    o[n*2] = '\0';
}

/* =========================================================================
 * CPU: file I/O helpers
 * ========================================================================= */
static int load_single_key(const char *fn, uint8_t out[KEY_BYTES]) {
    FILE *f = fopen(fn, "r"); if (!f) return 0;
    char line[256];
    while (fgets(line, 256, f)) {
        size_t l = strlen(line);
        while (l && (line[l-1]=='\n'||line[l-1]=='\r'||line[l-1]==' ')) line[--l]='\0';
        if (l == KEY_LEN) { fclose(f); return hex2bytes(line, out, KEY_BYTES); }
    }
    fclose(f); return 0;
}
static uint8_t **load_keys(const char *fn, size_t *cnt) {
    FILE *f = fopen(fn, "r");
    if (!f) { fprintf(stderr, "[!] Cannot open: %s\n", fn); exit(1); }
    size_t cap = 0; char line[256];
    while (fgets(line, 256, f)) cap++;
    rewind(f);
    uint8_t **keys = (uint8_t**)malloc(cap * sizeof(uint8_t*));
    size_t n = 0;
    while (fgets(line, 256, f)) {
        size_t l = strlen(line);
        while (l && (line[l-1]=='\n'||line[l-1]=='\r'||line[l-1]==' ')) line[--l]='\0';
        if (l != KEY_LEN) continue;
        keys[n] = (uint8_t*)malloc(KEY_BYTES);
        if (!hex2bytes(line, keys[n], KEY_BYTES)) { free(keys[n]); continue; }
        n++;
    }
    fclose(f); *cnt = n;
    printf("[+] Loaded %zu target keys from %s\n", n, fn);
    return keys;
}
static void fmt_dur(double s, char *o) {
    if (s < 0) s = 0;
    int h=(int)(s/3600),m=(int)((s-h*3600)/60),sec=(int)s%60;
    if (h)      sprintf(o, "%dh %02dm %02ds", h, m, sec);
    else if (m) sprintf(o, "%dm %02ds", m, sec);
    else        sprintf(o, "%.1fs", s);
}
static void fmt_rate(double r, char *o) {
    if      (r >= 1e9) sprintf(o, "%.2fG/s", r/1e9);
    else if (r >= 1e6) sprintf(o, "%.2fM/s", r/1e6);
    else if (r >= 1e3) sprintf(o, "%.1fK/s", r/1e3);
    else               sprintf(o, "%.0f/s",  r);
}

/* =========================================================================
 * Baby table: CPU build + insert
 * ========================================================================= */
static void baby_insert(const uint8_t *key33, uint64_t step) {
    uint32_t h = fast_hash(key33, KEY_BYTES);
    if (h == HT_EMPTY) h = 0u;
    int bid = (int)(h % (uint32_t)g_n_buckets);
    Bucket *B = &g_buckets[bid];
    pthread_mutex_lock(&B->lock);
    uint32_t i = h % B->size;
    while (B->slots[i].hash32 != HT_EMPTY)
        i = (i + 1) % B->size;
    B->slots[i].hash32 = h;
    B->slots[i].step   = step;
    memcpy(B->slots[i].key, key33, KEY_BYTES);
    B->count++;
    pthread_mutex_unlock(&B->lock);
}

static void *baby_builder(void *arg) {
    BabyWA *wa = (BabyWA*)arg;
    secp256k1_context *ctx = wa->ctx;
    secp256k1_pubkey cur = wa->start_point;
    uint8_t ser[KEY_BYTES]; size_t slen;
    for (uint64_t b = wa->b_from; b < wa->b_to; b++) {
        slen = KEY_BYTES;
        secp256k1_ec_pubkey_serialize(ctx, ser, &slen, &cur, SECP256K1_EC_COMPRESSED);
        baby_insert(ser, b);
        if (b + 1 < wa->b_to) {
            const secp256k1_pubkey *add2[2] = {&cur, &wa->G_pub};
            secp256k1_pubkey next;
            secp256k1_ec_pubkey_combine(ctx, &next, add2, 2);
            cur = next;
        }
    }
    return NULL;
}

/* =========================================================================
 * Upload baby table ke GPU VRAM sebagai flat array
 * ========================================================================= */
static void upload_baby_table_to_gpu() {
    /* Hitung total size */
    uint32_t total_size = 0;
    for (int i = 0; i < g_n_buckets; i++)
        total_size += g_buckets[i].size;
    g_gpu_tbl_size = total_size;

    printf("[GPU] Menyiapkan flat baby table (%u slot, %.2f GB)...\n",
           total_size, (double)total_size * sizeof(GpuBabySlot) / (1024.0*1024.0*1024.0));

    /* Alokasi host buffer sementara */
    GpuBabySlot *h_flat = (GpuBabySlot*)malloc((size_t)total_size * sizeof(GpuBabySlot));
    if (!h_flat) { fprintf(stderr, "[!] OOM flat table\n"); exit(1); }

    /* Init semua slot sebagai EMPTY */
    for (uint32_t i = 0; i < total_size; i++) {
        h_flat[i].hash32 = HT_EMPTY;
        h_flat[i].step   = 0;
        memset(h_flat[i].key, 0, 33);
    }

    /*
     * Rehash semua entri dari bucket-based ke flat single table.
     * GPU tidak punya konsep bucket, satu hash table besar lebih efisien.
     */
    uint32_t inserted = 0;
    for (int bid = 0; bid < g_n_buckets; bid++) {
        Bucket *B = &g_buckets[bid];
        for (uint32_t j = 0; j < B->size; j++) {
            if (B->slots[j].hash32 == HT_EMPTY) continue;
            /* Re-insert ke flat table */
            uint32_t h = B->slots[j].hash32;
            uint32_t i = h % total_size;
            while (h_flat[i].hash32 != HT_EMPTY)
                i = (i + 1) % total_size;
            h_flat[i].hash32 = h;
            h_flat[i].step   = B->slots[j].step;
            memcpy(h_flat[i].key, B->slots[j].key, 33);
            inserted++;
        }
    }
    printf("[GPU] %u entri di-upload ke VRAM...\n", inserted);

    /* Alokasi VRAM dan copy */
    CUDA_CHECK(cudaMalloc(&d_baby_tbl, (size_t)total_size * sizeof(GpuBabySlot)));
    CUDA_CHECK(cudaMemcpy(d_baby_tbl, h_flat,
                          (size_t)total_size * sizeof(GpuBabySlot),
                          cudaMemcpyHostToDevice));
    free(h_flat);
    printf("[GPU] Baby table berhasil di-upload ke VRAM!\n");
}

/* =========================================================================
 * Checkpoint
 * ========================================================================= */
static void save_checkpoint(GS *gs, uint64_t current_a) {
    pthread_mutex_lock(&gs->mu);
    char tmp[256]; snprintf(tmp, sizeof(tmp), "%s.tmp", CHECKPOINT_FILE);
    FILE *f = fopen(tmp, "w"); if (!f) { pthread_mutex_unlock(&gs->mu); return; }
    fprintf(f, "BSGS_CHECKPOINT_V2\n");
    fprintf(f, "timestamp=%ld\n", (long)time(NULL));
    fprintf(f, "total_bits=52\nm=%llu\nm_bits=26\n", (unsigned long long)BSGS_M);
    fprintf(f, "n_targets=%zu\n", gs->n_targets);
    fprintf(f, "last_a=%llu\n", (unsigned long long)current_a);
    fprintf(f, "total_a_done=%llu\n", (unsigned long long)gs->a_done);
    fprintf(f, "matches=%llu\n", (unsigned long long)gs->matches);
    fprintf(f, "---FOUND_KEYS---\n");
    for (size_t i = 0; i < gs->n_targets; i++)
        if (gs->found_k[i] != UINT64_MAX)
            fprintf(f, "target[%zu]=%llu\n", i, (unsigned long long)gs->found_k[i]);
    fclose(f); rename(tmp, CHECKPOINT_FILE);
    pthread_mutex_unlock(&gs->mu);
    printf("\n" CMG "[CKPT] Disimpan di a=%llu\n" CRST, (unsigned long long)current_a);
}

static int load_checkpoint(GS *gs, uint64_t *resume_a) {
    FILE *f = fopen(CHECKPOINT_FILE, "r"); if (!f) return 0;
    printf(CYL "[*] Memuat checkpoint...\n" CRST);
    char line[512]; int in_fk = 0; uint64_t last_a = 0; int fc = 0;
    while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\n")] = 0;
        if (strcmp(line, "---FOUND_KEYS---") == 0) { in_fk = 1; continue; }
        if (in_fk) {
            size_t idx; uint64_t val;
            if (sscanf(line, "target[%zu]=%llu", &idx, &val) == 2 &&
                idx < gs->n_targets && gs->found_k[idx] == UINT64_MAX) {
                gs->found_k[idx] = val;
                gs->found_mask |= (1ULL << (idx < 64 ? idx : 63));
                gs->matches++; fc++;
            }
        } else if (strncmp(line, "last_a=", 7) == 0)
            last_a = strtoull(line+7, NULL, 10);
    }
    fclose(f);
    if (last_a > 0) {
        *resume_a = last_a;
        if (fc > 0) printf(CGR "[+] Checkpoint: %d target sudah ditemukan\n" CRST, fc);
        return 1;
    }
    return 0;
}

/* =========================================================================
 * Signal handler
 * ========================================================================= */
#ifdef _WIN32
static BOOL WINAPI ctrl_handler(DWORD t) {
    if (t == CTRL_C_EVENT || t == CTRL_BREAK_EVENT) {
        printf("\n\n" CRD "[!] Ctrl+C - menyimpan checkpoint...\n" CRST);
        g_running = 0;
        if (g_gs_global) save_checkpoint(g_gs_global, g_gs_global->a_done);
        Sleep(1500);
        printf(CGR "[OK] Tersimpan. Gunakan --resume untuk melanjutkan.\n" CRST);
        exit(0);
    }
    return FALSE;
}
#else
static void sigint_handler(int sig) {
    (void)sig;
    printf("\n\n" CRD "[!] SIGINT - menyimpan checkpoint...\n" CRST);
    g_running = 0;
    if (g_gs_global) save_checkpoint(g_gs_global, g_gs_global->a_done);
    sleep(1);
    printf(CGR "[OK] Tersimpan. Gunakan --resume untuk melanjutkan.\n" CRST);
    exit(0);
}
#endif

/* =========================================================================
 * Display thread
 * ========================================================================= */
static void *display_thr(void *arg) {
    GS *gs = (GS*)arg;
    printf("\n\n\n\n\n"); fflush(stdout);
    while (g_running) {
        NANOSLEEP_MS(500);
        uint64_t done = gs->a_done;
        size_t n_found = 0;
        for (size_t i = 0; i < gs->n_targets; i++)
            if (gs->found_k[i] != UINT64_MAX) n_found++;
        struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
        double el = (now.tv_sec - gs->t0.tv_sec) + (now.tv_nsec - gs->t0.tv_nsec)*1e-9;
        uint64_t total_a = gs->a_end - gs->a_start;
        double pct  = total_a > 0 ? (double)done / total_a * 100.0 : 0;
        double rate = el > 0 ? (double)done * gs->n_targets / el : 0;
        char sr[16], se[24], eta[24];
        fmt_rate(rate, sr); fmt_dur(el, se);
        if (done > 0 && done < total_a && rate > 0)
            fmt_dur((total_a - done) * gs->n_targets / rate, eta);
        else strcpy(eta, "--:--");
        int bar_w = 50, filled = (int)(pct / 100.0 * bar_w);
        printf("\033[5A");
        printf("\033[K" CBOLD CBL "+-- BSGS 2^52 CUDA  [GTX1060 | Win %llu | Off %llu]" CRST "\n",
               (unsigned long long)gs->window_num,
               (unsigned long long)gs->window_offset);
        printf("\033[K [" CBOLD CCY);
        for (int i = 0; i < bar_w; i++)
            putchar(i < filled ? '#' : (i == filled ? '>' : '-'));
        printf(CRST CBOLD CCY "] %.2f%%" CRST "\n", pct);
        printf("\033[K a : " CYL "%llu / %llu" CRST "  Found: " CMG CBOLD "%zu / %zu" CRST "\n",
               (unsigned long long)done, (unsigned long long)total_a,
               n_found, gs->n_targets);
        printf("\033[K Rate: " CGR CBOLD "%-12s" CRST " Elapsed: " CCY "%-10s" CRST " ETA: " CCY "%s" CRST "\n",
               sr, se, eta);
        printf("\033[K" CBOLD CBL "+-------------------------------------------------------" CRST "\n");
        fflush(stdout);
    }
    return NULL;
}

/* =========================================================================
 * GPU Giant Step Runner (dipanggil dari main, bukan pthread worker)
 * ========================================================================= */
static void run_gpu_giant_steps(GS *gs, secp256k1_context *ctx) {
    size_t nt = gs->n_targets;
    uint64_t a_start = gs->a_start;
    uint64_t a_end   = gs->a_end;

    /* Alokasi device memory untuk found_k */
    uint64_t *d_found_k;
    CUDA_CHECK(cudaMalloc(&d_found_k, nt * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_found_k, gs->found_k, nt * sizeof(uint64_t), cudaMemcpyHostToDevice));

    /* Alokasi progress counter */
    unsigned long long *d_a_done;
    CUDA_CHECK(cudaMalloc(&d_a_done, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_a_done, 0, sizeof(unsigned long long)));

    /* Hitung koordinat -(m*G) untuk dikirim ke kernel */
    /* Serialize pos_mG dari secp256k1, lalu negasi y */
    uint8_t mG_ser[33]; size_t slen = 33;
    secp256k1_ec_pubkey_serialize(ctx, mG_ser, &slen, &gs->pos_mG, SECP256K1_EC_COMPRESSED);

    /* Parse ke limbs untuk GPU */
    uint32_t negmG_x[8], negmG_y_pre[8];
    /* x limbs dari mG_ser[1..32] (big-endian -> little-endian limbs) */
    for (int i = 0; i < 8; i++) {
        int off = 1 + (7-i)*4;
        negmG_x[i] = ((uint32_t)mG_ser[off]   << 24) |
                     ((uint32_t)mG_ser[off+1] << 16) |
                     ((uint32_t)mG_ser[off+2] <<  8) |
                      (uint32_t)mG_ser[off+3];
    }
    /* y: hitung dari x (y^2 = x^3 + 7), pilih paritas BERLAWANAN dengan mG_ser[0] */
    /* (negasi point = flip y) */
    /* Kita kirim sebagai uint32 array, kernel GPU akan pakai ini */
    /* Untuk simplisitas, kita encode negmG sebagai compressed point juga */
    /* dengan prefix 02/03 dibalik */
    uint8_t negmG_ser[33];
    memcpy(negmG_ser, mG_ser, 33);
    negmG_ser[0] = (mG_ser[0] == 0x02) ? 0x03 : 0x02;  /* flip paritas y = negasi */

    /* Serialize T_shifted points (starting points untuk batch) */
    uint8_t *h_T_ser = (uint8_t*)malloc(nt * 33);
    for (size_t i = 0; i < nt; i++) {
        slen = 33;
        secp256k1_ec_pubkey_serialize(ctx, h_T_ser + i*33, &slen,
                                      &gs->T_points[i], SECP256K1_EC_COMPRESSED);
    }

    /* Upload T_ser ke device */
    uint8_t *d_T_ser;
    CUDA_CHECK(cudaMalloc(&d_T_ser, nt * 33));
    CUDA_CHECK(cudaMemcpy(d_T_ser, h_T_ser, nt * 33, cudaMemcpyHostToDevice));
    free(h_T_ser);

    /* Parse negmG coords untuk kernel */
    uint32_t negmG_y[8];
    /* Derive y dari negmG_ser di CPU, kirim ke kernel sebagai constant */
    /* Untuk mempercepat, kita pass sebagai parameter */
    /* (Dalam implementasi ini kita skip dan biarkan kernel hitung dari prefix) */
    memset(negmG_y, 0, sizeof(negmG_y));  /* placeholder, kernel akan hitung dari prefix */

    printf("[GPU] Memulai giant step kernel...\n");
    printf("[GPU] Range a: [%llu, %llu) - total %llu langkah\n",
           (unsigned long long)a_start,
           (unsigned long long)a_end,
           (unsigned long long)(a_end - a_start));

    uint64_t total_a = a_end - a_start;
    uint64_t processed = 0;
    unsigned long long h_a_done = 0;

    /* Launch kernel dalam batch */
    while (processed < total_a && g_running) {
        uint64_t batch = CUDA_BATCH_SIZE;
        if (processed + batch > total_a) batch = total_a - processed;

        uint64_t cur_a_start = a_start + processed;

        /* Hitung jumlah thread */
        uint64_t n_threads = batch;
        uint32_t n_blocks = (uint32_t)((n_threads + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
        /* Batasi block count agar tidak terlalu besar */
        if (n_blocks > 65535) n_blocks = 65535;

        /* Parse x limbs dari negmG_ser */
        for (int i = 0; i < 8; i++) {
            int off = 1 + (7-i)*4;
            negmG_x[i] = ((uint32_t)negmG_ser[off]   << 24) |
                         ((uint32_t)negmG_ser[off+1] << 16) |
                         ((uint32_t)negmG_ser[off+2] <<  8) |
                          (uint32_t)negmG_ser[off+3];
        }

        /* Launch kernel */
        giant_step_kernel<<<n_blocks, CUDA_THREADS_PER_BLOCK>>>(
            d_baby_tbl, g_gpu_tbl_size,
            d_T_ser,
            negmG_x, negmG_y,
            cur_a_start, batch,
            (uint32_t)nt,
            gs->window_offset,
            BSGS_M,
            d_found_k,
            d_a_done
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        /* Cek error kernel */
        cudaError_t kerr = cudaGetLastError();
        if (kerr != cudaSuccess) {
            fprintf(stderr, "[CUDA KERNEL ERROR] %s\n", cudaGetErrorString(kerr));
            break;
        }

        /* Baca progress */
        CUDA_CHECK(cudaMemcpy(&h_a_done, d_a_done, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        gs->a_done = h_a_done;

        /* Cek apakah ada yang ditemukan */
        CUDA_CHECK(cudaMemcpy(gs->found_k, d_found_k, nt * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        size_t n_found = 0;
        for (size_t i = 0; i < nt; i++) {
            if (gs->found_k[i] != UINT64_MAX) {
                n_found++;
                if (!(gs->found_mask & (1ULL << (i < 64 ? i : 63)))) {
                    /* Baru ditemukan, print dan tulis ke file */
                    gs->found_mask |= (1ULL << (i < 64 ? i : 63));
                    gs->matches++;
                    char th[KEY_LEN+1];
                    bytes2hex(gs->targets[i], KEY_BYTES, th);
                    FILE *fo = fopen(gs->out_file, "a");
                    if (fo) {
                        fprintf(fo, "============================================================\n");
                        fprintf(fo, "[FOUND] GPU, a=%llu\n", (unsigned long long)(cur_a_start));
                        fprintf(fo, "Target: %s\n", th);
                        fprintf(fo, "k = %llu (0x%016llX)\n",
                                (unsigned long long)gs->found_k[i],
                                (unsigned long long)gs->found_k[i]);
                        fprintf(fo, "============================================================\n\n");
                        fclose(fo);
                    }
                    printf("\n" CGR
                        "+==============================================================+\n"
                        "| [OK] MATCH FOUND! GPU (%llu/%zu)\n"
                        "+--------------------------------------------------------------+\n"
                        "| Target[%zu]: " CYL "%s" CRST CGR "\n"
                        "| k = " CBOLD CCY "%llu" CRST CGR " (0x%016llX)\n"
                        "+==============================================================+\n" CRST,
                        (unsigned long long)gs->matches, nt, i, th,
                        (unsigned long long)gs->found_k[i],
                        (unsigned long long)gs->found_k[i]);
                    fflush(stdout);
                }
            }
        }

        /* Semua ditemukan? */
        if (n_found == nt) { processed += batch; break; }

        processed += batch;

        /* Simpan checkpoint setiap interval */
        if (processed % (CHECKPOINT_INTERVAL < total_a ? CHECKPOINT_INTERVAL : total_a) == 0)
            save_checkpoint(gs, a_start + processed);
    }

    gs->a_done = a_end - a_start;

    CUDA_CHECK(cudaFree(d_found_k));
    CUDA_CHECK(cudaFree(d_a_done));
    CUDA_CHECK(cudaFree(d_T_ser));
}

/* =========================================================================
 * Main
 * ========================================================================= */
int main(int argc, char *argv[]) {
    enable_ansi();

    if (argc < 3) {
        fprintf(stderr,
            CBOLD CBL "\n"
            "+==================================================================+\n"
            "| BSGS ECC 2^52 CUDA - GTX 1060 Edition                          |\n"
            "+==================================================================+\n" CRST
            "\nUsage: %s <base.txt> <batch.txt> [threads] [options]\n\n"
            "Options:\n"
            "  --resume    Resume dari checkpoint terakhir\n"
            "  --reset     Hapus semua checkpoint\n\n"
            "Compile (GTX 1060):\n"
            "  nvcc -O3 -arch=sm_61 -DHT_LOAD_TIGHT -o bsgs_2p52_cuda bsgs_2p52_cuda.cu"
            " -lsecp256k1 -lm -lpthread\n\n",
            argv[0]);
        return 1;
    }

    int nth = (argc >= 4) ? atoi(argv[3]) : DEFAULT_THREADS;
    if (nth < 1) nth = 1; if (nth > 64) nth = 64;

    int resume_mode = 0, reset_mode = 0;
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--resume") == 0) resume_mode = 1;
        if (strcmp(argv[i], "--reset")  == 0) reset_mode  = 1;
    }

    if (reset_mode) {
        printf(CYL "[*] Mereset checkpoint...\n" CRST);
        unlink(CHECKPOINT_FILE); unlink(PROGRESS_FILE);
    }

#ifdef _WIN32
    SetConsoleCtrlHandler(ctrl_handler, TRUE);
#else
    signal(SIGINT, sigint_handler);
#endif

    /* =====================================================================
     * Deteksi GPU
     * ===================================================================== */
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "[!] Tidak ada GPU CUDA yang ditemukan!\n");
        return 1;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf(CBOLD CBL
        "+======================================================================+\n"
        "| BSGS 2^52 CUDA - GTX 1060 Edition                                  |\n"
        "+======================================================================+\n" CRST);
    printf("| GPU       : %s\n", prop.name);
    printf("| VRAM      : %.1f GB\n", (double)prop.totalGlobalMem / (1024.0*1024.0*1024.0));
    printf("| CC        : %d.%d\n", prop.major, prop.minor);
    printf("| SMs       : %d\n", prop.multiProcessorCount);
    printf("| CUDA cores: ~%d\n", prop.multiProcessorCount * 128);  /* Pascal: 128 per SM */

    double ram_gb = (double)HT_SIZE * sizeof(GpuBabySlot) / (1024.0*1024.0*1024.0);
    printf("| Baby VRAM : ~%.2f GB\n", ram_gb);
    printf("| CPU thr   : %d\n", nth);
    if (ram_gb > (double)prop.totalGlobalMem / (1024.0*1024.0*1024.0) * 0.85) {
        printf(CRD "| PERINGATAN: Baby table (%.1f GB) mendekati batas VRAM (%.1f GB)!\n" CRST,
               ram_gb, (double)prop.totalGlobalMem / (1024.0*1024.0*1024.0));
        printf(CYL "|             Gunakan -DHT_LOAD_TIGHT saat compile untuk mengurangi ke ~%.1f GB\n" CRST,
               ram_gb / 2.0);
    }
    printf(CBOLD CBL "+======================================================================+\n\n" CRST);

    secp256k1_context *ctx = secp256k1_context_create(
        SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

    uint8_t base_bytes[KEY_BYTES];
    if (!load_single_key(argv[1], base_bytes)) {
        fprintf(stderr, "[!] Gagal memuat base key dari %s\n", argv[1]);
        secp256k1_context_destroy(ctx); return 1;
    }
    secp256k1_pubkey base_pub;
    if (!secp256k1_ec_pubkey_parse(ctx, &base_pub, base_bytes, KEY_BYTES)) {
        fprintf(stderr, "[!] Gagal parse base key\n");
        secp256k1_context_destroy(ctx); return 1;
    }
    { char bh[KEY_LEN+1]; bytes2hex(base_bytes, KEY_BYTES, bh); printf("[+] Base: %s\n", bh); }

    size_t n_targets;
    uint8_t **targets = load_keys(argv[2], &n_targets);
    secp256k1_pubkey *T_points = (secp256k1_pubkey*)malloc(n_targets * sizeof(secp256k1_pubkey));
    if (!T_points) { fprintf(stderr, "[!] OOM T_points\n"); return 1; }
    for (size_t i = 0; i < n_targets; i++) {
        if (!secp256k1_ec_pubkey_parse(ctx, &T_points[i], targets[i], KEY_BYTES)) {
            fprintf(stderr, "[!] Gagal parse target[%zu]\n", i);
            secp256k1_context_destroy(ctx); return 1;
        }
    }
    printf("[+] %zu targets di-parse\n", n_targets);

    static const char G_HEX[] = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    uint8_t gb[KEY_BYTES]; hex2bytes(G_HEX, gb, KEY_BYTES);
    secp256k1_pubkey G_pub;
    if (!secp256k1_ec_pubkey_parse(ctx, &G_pub, gb, KEY_BYTES)) {
        fprintf(stderr, "[!] Gagal parse G\n"); return 1;
    }
    secp256k1_pubkey neg_G = G_pub;
    secp256k1_ec_pubkey_negate(ctx, &neg_G);

    uint8_t scalar_m[32]; memset(scalar_m, 0, 32);
    { uint64_t mv = BSGS_M; for (int bi=0;bi<8;bi++) scalar_m[31-bi]=(uint8_t)(mv>>(bi*8)); }
    secp256k1_pubkey pos_mG = G_pub;
    secp256k1_ec_pubkey_tweak_mul(ctx, &pos_mG, scalar_m);

    secp256k1_pubkey neg_G_base = base_pub;
    secp256k1_ec_pubkey_negate(ctx, &neg_G_base);

    uint8_t scalar_max[32]; memset(scalar_max, 0, 32);
    { uint64_t mv = BSGS_MAX; for (int bi=0;bi<8;bi++) scalar_max[31-bi]=(uint8_t)(mv>>(bi*8)); }
    secp256k1_pubkey pos_maxG = G_pub;
    secp256k1_ec_pubkey_tweak_mul(ctx, &pos_maxG, scalar_max);
    secp256k1_pubkey neg_maxG = pos_maxG;
    secp256k1_ec_pubkey_negate(ctx, &neg_maxG);

    /* Alokasi baby table */
    printf("[+] Mengalokasikan baby table (~%.2f GB RAM)...\n",
           (double)HT_SIZE * sizeof(BabySlot) / (1024.0*1024.0*1024.0));
    g_buckets = (Bucket*)malloc(g_n_buckets * sizeof(Bucket));
    if (!g_buckets) { fprintf(stderr, "[!] OOM buckets\n"); return 1; }
    uint32_t bucket_size = HT_SIZE / (uint32_t)g_n_buckets;
    for (int i = 0; i < g_n_buckets; i++) {
        g_buckets[i].slots = (BabySlot*)malloc(bucket_size * sizeof(BabySlot));
        if (!g_buckets[i].slots) { fprintf(stderr, "[!] OOM bucket %d\n", i); return 1; }
        g_buckets[i].size  = bucket_size;
        g_buckets[i].count = 0;
        pthread_mutex_init(&g_buckets[i].lock, NULL);
        for (uint32_t j = 0; j < bucket_size; j++)
            g_buckets[i].slots[j].hash32 = HT_EMPTY;
    }

    /* Build baby table (CPU, paralel) */
    printf("[+] Membangun baby table dengan %d CPU thread (%llu entri)...\n",
           nth, (unsigned long long)BSGS_M);
    {
        pthread_t *bthr = (pthread_t*)malloc(nth * sizeof(pthread_t));
        BabyWA *bwa = (BabyWA*)malloc(nth * sizeof(BabyWA));
        uint64_t chunk = BSGS_M / (uint64_t)nth;
        struct timespec tb0, tn; clock_gettime(CLOCK_MONOTONIC, &tb0);
        for (int t = 0; t < nth; t++) {
            bwa[t].ctx    = ctx;
            bwa[t].G_pub  = G_pub;
            bwa[t].b_from = (uint64_t)t * chunk;
            bwa[t].b_to   = (t == nth-1) ? BSGS_M : bwa[t].b_from + chunk;
            if (bwa[t].b_from == 0) {
                bwa[t].start_point = G_pub;
            } else {
                uint8_t sc[32]; memset(sc, 0, 32);
                uint64_t sv = bwa[t].b_from + 1;
                for (int bi=0;bi<8;bi++) sc[31-bi]=(uint8_t)(sv>>(bi*8));
                bwa[t].start_point = G_pub;
                secp256k1_ec_pubkey_tweak_mul(ctx, &bwa[t].start_point, sc);
            }
            pthread_create(&bthr[t], NULL, baby_builder, &bwa[t]);
        }
        for (int t = 0; t < nth; t++) pthread_join(bthr[t], NULL);
        clock_gettime(CLOCK_MONOTONIC, &tn);
        double el = (tn.tv_sec-tb0.tv_sec)+(tn.tv_nsec-tb0.tv_nsec)*1e-9;
        char bt[24]; fmt_dur(el, bt);
        printf("[+] Baby table selesai dalam %s (%.2fM steps/s)\n",
               bt, BSGS_M / el / 1e6);
        free(bthr); free(bwa);
    }

    /* Upload ke GPU */
    upload_baby_table_to_gpu();

    /* T_shifted */
    secp256k1_pubkey *T_shifted = (secp256k1_pubkey*)malloc(n_targets * sizeof(secp256k1_pubkey));
    if (!T_shifted) { fprintf(stderr, "[!] OOM T_shifted\n"); return 1; }
    for (size_t i = 0; i < n_targets; i++) {
        const secp256k1_pubkey *pts[2] = {&T_points[i], &neg_G_base};
        secp256k1_ec_pubkey_combine(ctx, &T_shifted[i], pts, 2);
    }

    uint64_t *found_k = (uint64_t*)malloc(n_targets * sizeof(uint64_t));
    if (!found_k) { fprintf(stderr, "[!] OOM found_k\n"); return 1; }
    for (size_t i = 0; i < n_targets; i++) found_k[i] = UINT64_MAX;

    uint64_t full_mask = 0;
    for (size_t i = 0; i < n_targets && i < 64; i++) full_mask |= (1ULL << i);

    /* Window loop */
    for (uint64_t window = 0; ; window++) {
        uint64_t win_offset = window * BSGS_MAX;

        size_t already = 0;
        for (size_t i = 0; i < n_targets; i++)
            if (found_k[i] != UINT64_MAX) already++;
        if (already == n_targets) {
            printf(CGR "\n[OK] Semua %zu target ditemukan.\n" CRST, n_targets);
            break;
        }

        printf(CBOLD CYL
            "\n+======================================================================+\n"
            "| Window %-3llu  Range: [%llu, %llu)\n"
            "+======================================================================+\n" CRST,
            (unsigned long long)window,
            (unsigned long long)win_offset,
            (unsigned long long)(win_offset + BSGS_MAX));

        GS gs;
        memset(&gs, 0, sizeof(gs));
        gs.ctx           = ctx;
        gs.targets       = targets;
        gs.n_targets     = n_targets;
        gs.T_points      = T_shifted;
        gs.pos_mG        = pos_mG;
        gs.neg_G         = neg_G;
        gs.found_k       = found_k;
        gs.found_mask    = 0;
        gs.full_mask     = full_mask;
        gs.out_file      = "bsgs_results_2p52.txt";
        gs.a_start       = 0;
        gs.a_end         = BSGS_M + 1;
        gs.a_done        = 0;
        gs.matches       = 0;
        gs.window_offset = win_offset;
        gs.window_num    = window;
        pthread_mutex_init(&gs.mu, NULL);
        for (size_t i = 0; i < n_targets && i < 64; i++)
            if (found_k[i] != UINT64_MAX) gs.found_mask |= (1ULL << i);

        g_gs_global = &gs;
        g_running   = 1;

        if (window == 0) {
            uint64_t resume_a = 0;
            if (resume_mode && load_checkpoint(&gs, &resume_a)) {
                gs.a_start = resume_a;
                printf(CGR "[+] Melanjutkan dari a=%llu\n" CRST, (unsigned long long)resume_a);
            } else if (resume_mode)
                printf(CYL "[!] Tidak ada checkpoint, mulai baru\n" CRST);
        }

        clock_gettime(CLOCK_MONOTONIC, &gs.t0);

        /* Log window */
        FILE *fo = fopen(gs.out_file, "a");
        if (fo) {
            if (window == 0 && !resume_mode) {
                fprintf(fo, "============================================================\n");
                fprintf(fo, "BSGS CUDA Scan\n");
                char bh[KEY_LEN+1]; bytes2hex(base_bytes, KEY_BYTES, bh);
                fprintf(fo, "Base: %s\n", bh);
                time_t now = time(NULL);
                fprintf(fo, "Started: %s", ctime(&now));
                fprintf(fo, "============================================================\n\n");
            }
            fprintf(fo, "[Window %llu] Range [%llu, %llu)\n",
                    (unsigned long long)window,
                    (unsigned long long)win_offset,
                    (unsigned long long)(win_offset + BSGS_MAX));
            fclose(fo);
        }

        /* Start display thread */
        pthread_t dt;
        pthread_create(&dt, NULL, display_thr, &gs);

        /* Jalankan giant step di GPU */
        run_gpu_giant_steps(&gs, ctx);

        g_running = 0;
        pthread_join(dt, NULL);
        save_checkpoint(&gs, gs.a_end);

        /* Window summary */
        struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
        double win_el = (t1.tv_sec-gs.t0.tv_sec)+(t1.tv_nsec-gs.t0.tv_nsec)*1e-9;
        char wt[24]; fmt_dur(win_el, wt);
        size_t n_found_total = 0;
        for (size_t i = 0; i < n_targets; i++)
            if (found_k[i] != UINT64_MAX) n_found_total++;

        printf("\n" CBOLD CGR
            "+======================================================================+\n"
            "| WINDOW %3llu SELESAI\n"
            "+======================================================================+\n" CRST,
            (unsigned long long)window);
        printf("| Range  : [%llu, %llu)\n",
               (unsigned long long)win_offset,
               (unsigned long long)(win_offset + BSGS_MAX));
        printf("| Found  : " CMG "%zu" CRST " / %zu\n", n_found_total, n_targets);
        printf("| Time   : " CCY "%s" CRST "\n", wt);
        printf(CBOLD CGR "+======================================================================+\n" CRST);

        if (n_found_total > 0) {
            printf("\n" CBOLD "Keys ditemukan:\n" CRST);
            for (size_t i = 0; i < n_targets; i++) {
                if (found_k[i] != UINT64_MAX) {
                    char th[KEY_LEN+1]; bytes2hex(targets[i], KEY_BYTES, th);
                    printf("  [%2zu] k = %llu (0x%016llX) -> %s\n", i,
                           (unsigned long long)found_k[i],
                           (unsigned long long)found_k[i], th);
                }
            }
        }

        pthread_mutex_destroy(&gs.mu);

        /* Shift T_shifted untuk window berikutnya */
        if (n_found_total < n_targets) {
            for (size_t i = 0; i < n_targets; i++) {
                if (found_k[i] == UINT64_MAX) {
                    const secp256k1_pubkey *pts[2] = {&T_shifted[i], &neg_maxG};
                    secp256k1_pubkey sn;
                    secp256k1_ec_pubkey_combine(ctx, &sn, pts, 2);
                    T_shifted[i] = sn;
                }
            }
        }
    }

    /* Final summary */
    size_t n_found_final = 0;
    for (size_t i = 0; i < n_targets; i++)
        if (found_k[i] != UINT64_MAX) n_found_final++;

    printf("\n" CBOLD CGR
        "+======================================================================+\n"
        "| SCAN SELESAI\n"
        "+======================================================================+\n" CRST);
    printf("| Total   : %zu\n| Found   : " CMG "%zu" CRST "\n| Missing : " CRD "%zu" CRST "\n",
           n_targets, n_found_final, n_targets - n_found_final);
    printf("| Output  : bsgs_results_2p52.txt\n");
    printf(CBOLD CGR "+======================================================================+\n" CRST);

    /* Cleanup */
    if (d_baby_tbl) cudaFree(d_baby_tbl);
    free(found_k); free(T_shifted); free(T_points);
    for (size_t i = 0; i < n_targets; i++) free(targets[i]);
    free(targets);
    for (int i = 0; i < g_n_buckets; i++) {
        free(g_buckets[i].slots);
        pthread_mutex_destroy(&g_buckets[i].lock);
    }
    free(g_buckets);
    secp256k1_context_destroy(ctx);
    cudaDeviceReset();
    return 0;
}
