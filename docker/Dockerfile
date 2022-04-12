# This image is pushed to heltai/dealii:vscode
FROM dealii/dealii:master-focal

LABEL maintainer="luca.heltai@gmail.com"

USER root

RUN apt-get update \
    && apt-get install -y \
    biber \
    cmake-curses-gui \
    doxygen \
    gdb \
    graphviz \
    libcgal-dev \
    linux-tools-common \
    linux-tools-generic \
    valgrind \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.vscode-server/extensions \
    /root/.vscode-server-insiders/extensions 

ENV OMPI_ALLOW_RUN_AS_ROOT 1

ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM 1

ENV HOME /root

RUN mkdir /root/.ssh

ADD dotgdbinit.py /root/.gdbinit

RUN mv /usr/local/bin/clang-format /usr/bin/clang-format

# Now install parmoonlith
RUN cd /usr/src/ && \
    git clone https://bitbucket.org/zulianp/par_moonolith.git && \
    cd par_moonolith && \
    mkdir build && \
    cd build && \
    cmake .. -G Ninja \
        -DBUILD_SHARED_LIBS=ON && \
    ninja && \
    ninja install && \
    cd ../ && rm -rf build
