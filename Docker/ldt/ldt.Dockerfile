ARG boost_version=1.74
FROM boost:${boost_version}

ARG clang_version=14

#整理libtorch资源,清理空间
RUN set -ex && \
    mkdir libtorch && \
    cp -r build_libtorch/build/bin libtorch/ && \
    cp -r build_libtorch/build/lib libtorch/ && \
    cp -r pytorch/torch/include/ libtorch/ && \
    cp -r pytorch/torch/share libtorch/ && \
    rm -rf build_libtorch && \
    rm -rf pytorch
    
#需要拉取最新的ldt-cpp文件
COPY ldt-cpp ./ldt-cpp

RUN set -ex && \
    cd ldt-cpp && \
    mkdir build && \
    cd build && \
    export Torch_DIR=$PWD/../../libtorch/share/cmake/Torch && \
    export CC=clang-${clang_version} && \
    export CXX=clang++-${clang_version} && \
    cmake -DCMAKE_PREFIX_PATH=$PWD/../../libtorch ..