ARG libtorch_version=1.10.2
FROM libtorch:${libtorch_version}

ARG clang_version=14

RUN set -ex && \
    apt-get -y install libboost-all-dev

COPY example_boost.cpp ./example_boost.cpp

RUN clang++-${clang_version} -std=c++20 -stdlib=libc++ example_boost.cpp -o example_boost.exe && \
    ./example_boost.exe && \ 
    rm ./example_boost.exe 