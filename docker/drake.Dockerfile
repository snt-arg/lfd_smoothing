FROM robotlocomotion/drake:focal-1.18.0

# set eviroment variables
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV LFD_PKG_PATH "/root/lfd_ws"

# copy repository
RUN mkdir -p ${LFD_PKG_PATH}
COPY . ${LFD_PKG_PATH}

# Install python and apt dependencies
RUN . ${LFD_PKG_PATH}/docker/install_deps.bash

WORKDIR ${LFD_PKG_PATH}
