# This image is pushed to heltai/dealii:vscode
FROM dealii/dealii:master-jammy

LABEL maintainer="luca.heltai@gmail.com"

USER root

RUN apt-get update \
    && apt-get install -y \
    biber \
    cmake-curses-gui \
    doxygen \
    gdb \
    graphviz \
    linux-tools-common \
    linux-tools-generic \
    valgrind \
    vim \
    wdiff \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Now install parmoonlith
RUN cd /usr/src/ && \
    git clone https://bitbucket.org/zulianp/par_moonolith.git && \
    cd par_moonolith && \
    mkdir build && \
    cd build && \
    cmake .. -G Ninja \
        -DBUILD_SHARED_LIBS=ON && \
    sed -i '/#include <vector>/a #include <limits>' _deps/benchmark-src/src/benchmark_register.h &&\
    ninja && \
    ninja install && \
    cd ../ && rm -rf build

RUN mkdir -p /root/.vscode-server/extensions \
    /root/.vscode-server-insiders/extensions 

ENV OMPI_ALLOW_RUN_AS_ROOT 1

ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM 1

ADD dotgdbinit.py /root/.gdbinit

RUN mv /usr/local/bin/clang-format /usr/bin/clang-format

ADD  	\
	build.sh \
	copy_output.sh \
	copy_test.sh \
	d2check.sh \
    cwdiff \
    newdocentry.sh \
    work_on_test.sh \
    /usr/bin/

