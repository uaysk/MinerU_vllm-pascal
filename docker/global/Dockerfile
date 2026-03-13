FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12
ARG TORCH_CUDA_ARCH_LIST=6.1
ARG MAX_JOBS=1
ARG VLLM_PASCAL_REPO=https://github.com/uaysk/vllm-pascal.git
ARG VLLM_PASCAL_REF=main

ENV CUDA_HOME=/usr/local/cuda-12.2
ENV PATH=${CUDA_HOME}/bin:/usr/local/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CC=/usr/bin/gcc-12
ENV CXX=/usr/bin/g++-12
ENV CMAKE_ARGS=-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
ENV VLLM_TARGET_DEVICE=cuda
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV MAX_JOBS=${MAX_JOBS}
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        git \
        gcc-12 \
        g++-12 \
        cmake \
        ninja-build \
        fontconfig \
        fonts-noto-core \
        fonts-noto-cjk \
        libgl1 && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python /tmp/get-pip.py && \
    rm -f /tmp/get-pip.py && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && \
    python -m pip install \
        "setuptools>=77,<81" \
        setuptools_scm \
        wheel \
        packaging \
        cmake \
        ninja \
        jinja2 \
        regex \
        protobuf && \
    python -m pip install \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 \
        torchvision==0.20.1 \
        torchaudio==2.5.1

RUN git clone --depth 1 --branch ${VLLM_PASCAL_REF} ${VLLM_PASCAL_REPO} /opt/vllm-pascal && \
    python -m pip install /opt/vllm-pascal --no-build-isolation --no-deps

COPY . /opt/MinerU

RUN python -m pip install \
        "aiohttp>=3.13.3" \
        "anthropic>=0.71.0" \
        blake3 \
        cachetools \
        cbor2 \
        cloudpickle \
        "compressed-tensors==0.13.0" \
        "depyf==0.20.0" \
        "diskcache==5.6.3" \
        einops \
        "fastapi[standard]>=0.115.0" \
        "gguf>=0.17.0" \
        grpcio \
        grpcio-reflection \
        ijson \
        "lark==1.2.2" \
        "llguidance>=1.3.0,<1.4.0" \
        "lm-format-enforcer==0.11.3" \
        mcp \
        "mistral_common[image]>=1.9.1" \
        "model-hosting-container-standards>=0.1.13,<1.0.0" \
        msgspec \
        "openai>=1.99.1" \
        "openai-harmony>=0.0.3" \
        "opencv-python-headless>=4.13.0" \
        "opentelemetry-api>=1.27.0" \
        "opentelemetry-exporter-otlp>=1.27.0" \
        "opentelemetry-sdk>=1.27.0" \
        "opentelemetry-semantic-conventions-ai>=0.4.1" \
        "outlines_core==0.2.11" \
        partial-json-parser \
        "prometheus_client>=0.18.0" \
        "prometheus-fastapi-instrumentator>=7.0.0" \
        psutil \
        py-cpuinfo \
        pybase64 \
        "pydantic>=2.12.0" \
        python-json-logger \
        pyyaml \
        "pyzmq>=25.0.0" \
        sentencepiece \
        setproctitle \
        "tiktoken>=0.6.0" \
        "tokenizers>=0.21.1" \
        tqdm \
        "transformers>=4.56.0,<5" \
        watchfiles \
        "xgrammar==0.1.29" \
        "boto3>=1.28.43" \
        "click>=8.1.7" \
        "loguru>=0.7.2" \
        "numpy>=1.21.6" \
        "pdfminer.six>=20251230" \
        requests \
        httpx \
        "pillow>=11.0.0" \
        "pypdfium2>=4.30.0" \
        "pypdf>=5.6.0" \
        reportlab \
        "pdftext>=0.6.3" \
        "modelscope>=1.26.0" \
        "huggingface-hub>=0.32.4" \
        "json-repair>=0.46.2" \
        "opencv-python>=4.11.0.86" \
        "fast-langdetect>=0.2.3,<0.3.0" \
        "scikit-image>=0.25.0,<1.0.0" \
        "openai>=1.70.0,<3" \
        "beautifulsoup4>=4.13.5,<5" \
        "magika>=0.6.2,<1.1.0" \
        "mineru-vl-utils>=0.1.19.1,<1" \
        "qwen-vl-utils>=0.0.14,<1" \
        "accelerate>=1.5.1" \
        "matplotlib>=3.10,<4" \
        "ultralytics>=8.3.48,<9" \
        "doclayout_yolo==0.0.4" \
        "dill>=0.3.8,<1" \
        "ftfy>=6.3.1,<7" \
        "shapely>=2.0.7,<3" \
        "pyclipper>=1.3.0,<2" \
        "omegaconf>=2.3.0,<3" \
        "onnxruntime>1.17.0" \
        fastapi \
        python-multipart \
        uvicorn && \
    python -m pip install /opt/MinerU --no-deps

RUN /bin/bash -lc "MINERU_MODEL_SOURCE=huggingface mineru-models-download -s huggingface -m all"

ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]
