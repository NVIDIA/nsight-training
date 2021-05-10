FROM nvcr.io/nvidia/cuda:11.2.0-devel-ubuntu18.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
	software-properties-common \
	apt-transport-https \
    wget \
    ca-certificates \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    add-apt-repository -y ppa:graphics-drivers/ppa && \
    add-apt-repository -y ppa:ettusresearch/uhd &&\
    add-apt-repository -y ppa:srslte/releases && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin -O /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    add-apt-repository -y "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
	gcc-9 \
	g++-9 \
	build-essential \
	cmake \
	make \
	git \
	gdb \
	nano \
	mc \
	screen \
    libboost-all-dev \
    libfftw3-dev \
    libmbedtls-dev \
    libconfig++-dev \
    libsctp-dev \
    libuhd-dev \
    uhd-host \
    libusb-1.0-0-dev \
    python3-mako \
    dpdk \
    dpdk-dev \
    libdpdk-dev \
    qtbase5-dev \
    libqt5charts5-dev \
    srslte-dev \
    libliquid-dev \
    libgtest-dev \
    doxygen \
    graphviz \
    vim \
    git-lfs \
    python3-pip \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-9 && \
    update-alternatives --config gcc

RUN pip3 install \
    jupyterlab


RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
	cuda-gdb-11-2 \
	cuda-sanitizer-11-2 \
    cuda-nvcc-11-2 \
	libpng-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-sanitizer-11-3_11.3.58-1_amd64.deb .
RUN apt install -y ./cuda-sanitizer-11-3_11.3.58-1_amd64.deb
ENV PATH=/usr/local/cuda-11.3/bin:/usr/local/cuda-11.3/bin/compute-sanitizer:/usr/local/cuda-11.2/bin:${PATH}
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-11.2/lib64
ENV CUDACXX=/usr/local/cuda/bin/nvcc

COPY jupyter/ /root/.jupyter/

WORKDIR /dli/task
ADD task .

# Jupyter listens on 8888
EXPOSE 8888

ENV SHELL="/bin/bash"

ADD entrypoint.sh /usr/local/bin
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
