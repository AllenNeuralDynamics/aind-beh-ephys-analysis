# hash:sha256:8d5d3da8954239b6d5b596e398a691544b9471cecd013aadb57a59f47ce13dd2
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.8ubuntu1.1 \
        libgl1-mesa-glx=21.2.6-0ubuntu0.1~20.04.2 \
        libssl-dev=1.1.1f-1ubuntu2.22 \
        pandoc=2.5-3build2 \
        pkg-config=0.29.1-0ubuntu4 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        pandas==2.2.2 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aind-ephys-utils==0.0.15 \
    hdmf-zarr==0.6.0 \
    ipympl==0.9.3 \
    matplotlib==3.9.1 \
    numpy==1.26.4 \
    open-ephys-python-tools==0.1.10 \
    pynwb==2.6.0 \
    pypdf2==3.0.1 \
    scipy==1.13.0 \
    seaborn==0.13.2 \
    spikeinterface[full]==0.100.0 \
    statsmodels==0.14.2 \
    wavpack-numcodecs==0.1.5 \
    zarr==2.18.2 \
    harp-python@git+https://github.com/jsiegle/harp-python@decode-clock

COPY postInstall /
RUN /postInstall
