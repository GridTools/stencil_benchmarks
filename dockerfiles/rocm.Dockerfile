ARG ROCM_VERSION=3.7
FROM rocm/dev-ubuntu-18.04:${ROCM_VERSION}
LABEL maintainer="Felix Thaler <thaler@cscs.ch>"

RUN apt-get update -qq && \
    apt-get install -qq -y \
    file \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

COPY . /stencil_benchmarks
RUN pip3 install /stencil_benchmarks

RUN /bin/bash -c 'for c in $(compgen -c sbench); do cup=${c^^} && echo "eval \"$""(_${cup//-/_}_COMPLETE=source $c)\""; done >> ~/.bashrc'
