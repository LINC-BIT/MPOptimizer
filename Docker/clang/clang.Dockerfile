# ubuntu 22.04
FROM ubuntu:jammy-20220404

ARG clang_version=14

# docker build -t clang:xxx -f clang.Dockerfile .

RUN set -ex && \
    mkdir -p /opt/workplace && \
    apt-get update 

RUN apt-get install -y apt-utils curl wget unzip \
        lsb-release software-properties-common && \
    rm /bin/sh && \
    ln -sv /bin/bash /bin/sh && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd 

WORKDIR /opt/workplace 

# llvm
RUN wget https://apt.llvm.org/llvm.sh && \ 
    chmod +x llvm.sh 

RUN ./llvm.sh ${clang_version} all && \ 
    rm llvm.sh 

# A test program
COPY cpp20example.cpp ./cpp20example.cpp

RUN clang++-${clang_version} -std=c++20 -stdlib=libc++ cpp20example.cpp -o cpp20example.exe && \
    ./cpp20example.exe && \ 
    rm ./cpp20example.exe 