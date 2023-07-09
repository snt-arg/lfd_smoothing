FROM robotlocomotion/drake:focal-1.18.0

# set eviroment variables
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV LFDS_PKG_PATH "/root/lsdf_ws"

# copy repository
RUN mkdir -p ${LFDS_PKG_PATH}
COPY . ${LFDS_PKG_PATH}

# Install python and apt dependencies
RUN . ${LFDS_PKG_PATH}/docker/install_deps.bash

# Install lfd_smoothing dependecy
RUN pip3 install -e ${LFDS_PKG_PATH}/src/lfd_smoother

WORKDIR ${LFDS_PKG_PATH}
