FROM nvcr.io/nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04
RUN apt update -y && \
    apt install -y \
    software-properties-common \
    wget

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN add-apt-repository -y ppa:graphics-drivers/ppa
RUN add-apt-repository -y ppa:ettusresearch/uhd
RUN add-apt-repository -y ppa:srslte/releases

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/devtools/repos/ubuntu1804/amd64/nvidia.pub
RUN add-apt-repository -y "deb https://developer.download.nvidia.com/devtools/repos/ubuntu1804/amd64/ /"

RUN DEBIAN_FRONTEND=noninteractive \
    apt update -y && \
    apt install -y gcc-9 g++-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-9
RUN update-alternatives --config gcc

RUN DEBIAN_FRONTEND=noninteractive \
apt install -y \
build-essential cmake make git gdb nano mc screen \
libboost-all-dev libfftw3-dev libmbedtls-dev libconfig++-dev libsctp-dev \
libuhd-dev uhd-host libusb-1.0-0-dev python3-mako \
dpdk dpdk-dev libdpdk-dev \
qtbase5-dev libqt5charts5-dev \
srslte-dev \
libliquid-dev \
libgtest-dev \
doxygen graphviz \
vim \
git-lfs

RUN DEBIAN_FRONTEND=noninteractive \
apt install -y python3-pip


# Install updated nsight-compute and nsight-systems and add to PATH
# Note: when changed, update in nsight container, too
WORKDIR /build
RUN rm -rf /opt/nvidia/nsight-compute /opt/nvidia/nsight-systems
ADD https://developer.download.nvidia.com/devtools/repos/ubuntu1804/amd64/nsight-compute-2021.2.1_2021.2.1.2-1_amd64.deb .
ADD https://developer.download.nvidia.com/devtools/repos/ubuntu1804/amd64/NsightSystems-linux-public-2021.4.1.73-08591f7.deb .
RUN apt install -y ./nsight-compute-2021.2.1_2021.2.1.2-1_amd64.deb
RUN apt install -y ./NsightSystems-linux-public-2021.4.1.73-08591f7.deb

ENV PATH=/opt/nvidia/nsight-systems/2021.4.1/bin:/opt/nvidia/nsight-compute/2021.2.1:/usr/local/cuda-11.2/bin:${PATH}

RUN apt clean -y
RUN apt autoclean -y
RUN apt autoremove -y --purge
RUN dpkg -l | grep 'nvidia-cuda-toolkit\|qtbase5-dev\|libuhd-dev\|srslte-dev\|libliquid-dev'

# Install Pytorch dependencies for nsys
RUN python3 -m pip install pip==19.3.1
RUN python3 -m pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
# Setup config file option for nsys to avoid diagnostic error on events lost
RUN mkdir -p "$(dirname "$(nsys -z)")"
RUN echo 'SampleSoftwareFrequency=1000' >>"$(nsys -z)"

# juperlab part
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install jupyterlab==3.0.12

# Load saved JupyterLab workspace
COPY jupyter/ /root/.jupyter/

# Create workshop working directory with content (full repo)
WORKDIR /dli/task
ADD task .

# Jupyter listens on 8888
EXPOSE 8888

ENV CUDACXX="/usr/local/cuda/bin/nvcc"
ENV SHELL="/bin/bash"

# copy entrypoint script to /usr/local/bin
ADD entrypoint.sh /usr/local/bin
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
