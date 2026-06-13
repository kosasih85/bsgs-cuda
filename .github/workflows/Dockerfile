FROM nvidia/cuda:11.8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    libsecp256k1-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY bsgs_2p52_cuda.cu .

RUN nvcc -O3 -arch=sm_61 -DHT_LOAD_TIGHT -o bsgs_2p52_cuda bsgs_2p52_cuda.cu -lsecp256k1 -lpthread -lm

CMD ["./bsgs_2p52_cuda"]
