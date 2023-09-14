ARG clang_version=14
FROM clang:${clang_version}

ARG clang_version=14

RUN set -ex && \
    apt-get install -y git make cmake 

ARG torch_version=1.10.2
ARG torch_version_string=v${torch_version}

# RUN set -ex && \
#     env && \
#     echo "" && sleep 1 && \
#     git clone -b ${torch_version_string} --recurse-submodule https://github.com/pytorch/pytorch.git

# RUN wget -O pytorch_src-${torch_version}.zip \
#         https://codeload.github.com/pytorch/pytorch/zip/refs/tags/${torch_version_string} 

# RUN unzip pytorch_src-${torch_version}.zip  && \ 
#     mv pytorch-${torch_version} pytorch && \
#     ls pytorch 

RUN set -ex && \
    apt-get install -y python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev

# setuptools 版本过高(59.6.0)会出问题：
# AttributeError: module 'distutils' has no attribute 'version'
# 而且网络不好，翻墙也解决不了。
# RUN set -ex && \
#     pip install setuptools==59.5.0 

COPY setuptools-59.5.0-py3-none-any.whl setuptools-59.5.0-py3-none-any.whl

RUN set -ex && ls && \
    pip install setuptools-59.5.0-py3-none-any.whl && \
    rm setuptools-59.5.0-py3-none-any.whl

# RUN set -ex && \
#     pip install typing_extensions

RUN set -ex && \
    env && \
    pip3 install future && \
    pip3 install -U --user wheel mock pillow && \
    pip3 install testresources && \
    pip3 install Cython

#下载速度过慢，直接导入下载好的pytorch轮子
RUN set -ex && \
    env && \
    echo "" && sleep 1 && \
    git clone -b ${torch_version_string} --recurse-submodule https://github.com/pytorch/pytorch.git

# # 我需要指定编译器为 clang, 下面的 build 方式不知道如何指定编译器。
# # 还是得直接用 cmake 构建。
# RUN set -ex && \
#     mkdir pytorch-build && \
#     cd pytorch-build && \
#     python3 ../pytorch/tools/build_libtorch.py 


# CMake Warning at CMakeLists.txt:36 (message):
#   C++ standard version definition detected in environment variable.PyTorch
#   requires -std=c++14.  Please remove -std=c++ settings in your environment.
Run set -ex && \
    cd pytorch && \
    pip3 install -r requirements.txt

#   构建完整的libtorch需要超过4G的RAM，jetson nano上只提供2GB的交换空间，所以必须安装dphys-swapfile来临时从SD卡中获取额外空间
# Run set -ex && \
#     apt-get install dphys-swapfile && \
#     apt-get install nano

RUN set -ex && \
    mkdir build_libtorch && \
    cd build_libtorch && \
    export BUILD_PYTHON=OFF && \
    export BUILD_CAFFE2_OPS=OFF && \
    export USE_FBGEMM=OFF && \
    export USE_FAKELOWP=OFF && \
    export BUILD_TEST=OFF && \
    export USE_MKLDNN=OFF && \
    export USE_NNPACK=OFF && \
    export USE_XNNPACK=OFF && \
    export USE_QNNPACK=OFF && \
    export USE_PYTORCH_QNNPACK=OFF && \
    # export MAX_JOBS=4 && \
    export USE_BREAKPAD=0 && \
    export USE_NCCL=OFF && \
    export USE_OPENCV=OFF && \
    export USE_SYSTEM_NCCL=OFF && \
    export BUILD_SHARED_LIBS=ON && \
    PATH=/usr/lib/ccache:$PATH && \
    export CC=clang-14 && \
    export CXX=clang++-14 && \
    export DCMAKE_CXX_FLAGS="-Xclang -std=c++20 -stdlib=libc++" && \
    python3 ../pytorch/tools/build_libtorch.py

#整理libtorch资源,清理空间
RUN set -ex && \
    mkdir libtorch && \
    cp -r build_libtorch/build/bin libtorch/ && \
    cp -r build_libtorch/build/lib libtorch/ && \
    cp -r pytorch/torch/include/ libtorch/ && \
    cp -r pytorch/torch/share libtorch/ && \
    rm -rf build_libtorch && \
    rm -rf pytorch

#多阶段构建减少占用
FROM clang:${clang_version}

ARG clang_version=14

RUN set -ex && \
    apt-get install -y git make cmake python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev

COPY --from=0 /opt/workplace .

CMD ["/bin/bash"]