FROM nvcr.io/nvidia/pytorch:25.12-py3

# Install basic packages first.
WORKDIR /tmp
RUN apt update -y && apt install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    python3-pip \
    python3-venv \
    software-properties-common \
    unzip \
    wget \
    yasm \
    && rm -rf /var/lib/apt/lists/*

# Juperlab and its modules
RUN python3 -m pip install --upgrade pip
RUN pip3 install torch transformers accelerate sentencepiece protobuf


# Install updated nsight tools and add to PATH
# Note: when changed, update in nsight container, too
WORKDIR /build
RUN rm -rf /opt/nvidia/nsight-compute
RUN wget -O nsight-compute-2026.1.0.deb https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/nsight-compute-2026.1.0_2026.1.0.9-1_amd64.deb
RUN dpkg-deb -x nsight-compute-2026.1.0.deb /
RUN rm -f nsight-compute-2026.1.0.deb
RUN test -d /opt/nvidia/nsight-compute/2026.1.0
ENV PATH=/opt/nvidia/nsight-compute/2026.1.0:/usr/local/cuda/bin:${PATH}


# Add nsight extensions for Jupyterlab cell profiling
ADD jupyterlab_nvidia_nsight-0.8.2-py3-none-any.whl .
RUN python3 -m pip install jupyterlab_nvidia_nsight-0.8.2-py3-none-any.whl


# Add nsight python and nsight report interface modules
RUN python3 -m pip install nsight-python ncu-report


# Load saved JupyterLab workspace
COPY jupyter/ /root/.jupyter/


# Jupyter listens on this port
EXPOSE 8888


# Create workshop working directory with content (full repo)
WORKDIR /dli/task
COPY task/images images
COPY task/ncu ncu
COPY task/lab.ipynb lab.ipynb


ENV CUDACXX="/usr/local/cuda/bin/nvcc"
ENV SHELL="/bin/bash"


# Copy entrypoint script to /usr/local/bin
ADD entrypoint.sh /usr/local/bin
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
