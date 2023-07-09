FROM robotlocomotion/drake:focal-1.18.0

ARG DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash

# set eviroment variables
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV LFD_PKG_PATH "/root/lfd_ws"

# Open ports for meshcat and Jupyter:
EXPOSE 7000-7099/tcp
EXPOSE 8888/tcp

# copy repository
RUN mkdir -p ${LFD_PKG_PATH}
COPY . ${LFD_PKG_PATH}

# Install python and apt dependencies
RUN . ${LFD_PKG_PATH}/docker/install_deps.bash

WORKDIR ${LFD_PKG_PATH}
