# hash:sha256:8d5d3da8954239b6d5b596e398a691544b9471cecd013aadb57a59f47ce13dd2
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/cuda-miniconda-jupyterlab:latest

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        libssl-dev \
        pandoc \
        pkg-config \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-ephys-utils==0.0.15 \
    hdmf-zarr==0.11.0 \
    ipympl==0.9.3 \
    matplotlib==3.9.1 \
    numpy==2.1.3 \
    open-ephys-python-tools==0.1.10 \
    pynwb==2.8.3 \
    pypdf2==3.0.1 \
    spikeinterface[full]==0.102.1 \
    statsmodels==0.14.2 \
    wavpack-numcodecs==0.2.2 \
    natsort \
    harp-python@git+https://github.com/jsiegle/harp-python@decode-clock \
    aind-dynamic-foraging-data-utils==0.1.4 \
    codeocean==0.2.0 \
    requests==2.32.3 \
    pdf2image \
    seaborn \
    trimesh

COPY postInstall /
RUN /postInstall
