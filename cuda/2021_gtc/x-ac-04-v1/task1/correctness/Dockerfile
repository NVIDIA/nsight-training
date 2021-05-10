FROM  nvcr.io/nvidia/cudagl:11.2.0-base-ubuntu18.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
	software-properties-common \
	apt-transport-https \
    ca-certificates \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN dpkg --add-architecture i386 && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    echo ttf-mscorefonts-installer \
        msttcorefonts/accepted-mscorefonts-eula \
        select true \
        | debconf-set-selections && \
    apt-get update && \
    apt-get -y upgrade && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    build-essential \
    gcc-multilib \
    pkg-config \
    libxau-dev \
    libxdmcp-dev \
    libxcb1-dev \
    libxext-dev \
    libxext-dev:i386 \
    libx11-dev \
    libx11-dev:i386 \
	gcc-9 \
	g++-9 \
	libliquid-dev \
    xfce4 \
    curl \
    libc6-dev \
    libglu1 \
    libglu1:i386 \
    libsm6 \
    libxv1 \
    libxv1:i386 \
    make \
    python \
    python-numpy \
    x11-xkb-utils \
    xauth \
    xfonts-base \
    xkb-data \
    git \
    nano \
    vim \
    emacs \
    mousepad \
    xfce4-terminal \
    libxkbcommon-x11-0 \
	msttcorefonts \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG TURBOVNC_VERSION=2.2.1
ARG VIRTUALGL_VERSION=2.6.1
ARG LIBJPEG_VERSION=1.5.2
RUN cd /tmp && \
    curl -fsSL \
    -O https://sourceforge.net/projects/turbovnc/files/${TURBOVNC_VERSION}/turbovnc_${TURBOVNC_VERSION}_amd64.deb \
    -O https://sourceforge.net/projects/libjpeg-turbo/files/${LIBJPEG_VERSION}/libjpeg-turbo-official_${LIBJPEG_VERSION}_amd64.deb \
    -O https://sourceforge.net/projects/virtualgl/files/${VIRTUALGL_VERSION}/virtualgl_${VIRTUALGL_VERSION}_amd64.deb \
    -O https://sourceforge.net/projects/virtualgl/files/${VIRTUALGL_VERSION}/virtualgl32_${VIRTUALGL_VERSION}_amd64.deb \
    && \
    dpkg -i *.deb && \
    rm *.deb && \
    sed -i 's/$host:/unix:/g' /opt/TurboVNC/bin/vncserver
ENV PATH ${PATH}:/opt/VirtualGL/bin:/opt/TurboVNC/bin

ARG WEBSOCKIFY_VERSION=0.8.0
ARG NOVNC_VERSION=1.0.0
RUN curl -fsSL https://github.com/novnc/noVNC/archive/v${NOVNC_VERSION}.tar.gz | tar -xzf - -C /opt && \
    curl -fsSL https://github.com/novnc/websockify/archive/v${WEBSOCKIFY_VERSION}.tar.gz | tar -xzf - -C /opt && \
    mv /opt/noVNC-${NOVNC_VERSION} /opt/noVNC && \
    mv /opt/websockify-${WEBSOCKIFY_VERSION} /opt/websockify && \
    cd /opt/websockify && \
    make


RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
	cuda-gdb-11-2 \
	cuda-sanitizer-11-2 \
    cuda-nvcc-11-2 \
    cuda-nsight-11-2 \
    cuda-samples-11-2 \
	libpng-dev \
	eog \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN cd /tmp && \
    curl -fsSL -O https://mirror.dkm.cz/eclipse/technology/epp/downloads/release/2020-12/R/eclipse-cpp-2020-12-R-linux-gtk-x86_64.tar.gz && \
    tar xf eclipse-cpp-2020-12-R-linux-gtk-x86_64.tar.gz && \
    mv eclipse /usr/local/ && \
    rm eclipse-cpp-2020-12-R-linux-gtk-x86_64.tar.gz
RUN /usr/local/eclipse/eclipse -nosplash -application org.eclipse.equinox.p2.director -repository jar:file:///usr/local/cuda-11.2/nsightee_plugins/com.nvidia.cuda.repo-1.0.0-SNAPSHOT.zip!/ \
    -installIU com.nvidia.cuda.core \
    -installIU com.nvidia.cuda.debug.core \
    -installIU com.nvidia.cuda.debug.ui \
    -installIU com.nvidia.cuda.feature.group \
    -installIU com.nvidia.cuda.feature.jar \
    -installIU com.nvidia.cuda.profile.ui \
    -installIU com.nvidia.cuda.toolchain \
    -installIU com.nvidia.cuda.toolchain.doc.user \
    -installIU com.nvidia.cuda.ui
RUN echo '-Dorg.eclipse.oomph.setup.donate=false' >> /usr/local/eclipse/eclipse.ini
ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-sanitizer-11-3_11.3.58-1_amd64.deb .
RUN apt install -y ./cuda-sanitizer-11-3_11.3.58-1_amd64.deb
ENV PATH=/usr/local/cuda-11.3/bin:/usr/local/cuda-11.3/bin/compute-sanitizer:/usr/local/cuda-11.2/bin:${PATH}
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-11.2/lib64
ENV CUDACXX=/usr/local/cuda/bin/nvcc

ENV DISPLAY :1
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,display

COPY fs /
RUN cp /etc/xdg/xfce4/panel/default.xml /etc/xdg/xfce4/xfconf/xfce-perchannel-xml/xfce4-panel.xml

RUN groupadd -g 1000 nvidia && \
    useradd -g nvidia -u 1000 nvidia && \
    chown -R nvidia:nvidia /home/nvidia

WORKDIR /home/nvidia
USER nvidia

ARG VNCPASSWORD=nvidia
RUN mkdir -p .vnc && \
    echo "${VNCPASSWORD}" | vncpasswd -f > .vnc/passwd && \
    chmod 0600 .vnc/passwd

EXPOSE 5901

ENV SHELL="/bin/bash"

ENTRYPOINT ["/opt/websockify/run", "5901", "--web=/opt/noVNC", "--", "vncserver", "-fg", ":1"]
