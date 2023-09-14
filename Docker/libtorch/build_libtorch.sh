LIBTORCH_VERSION=1.10.2
CLANG_VERSION=14
docker build \
    --build-arg https_proxy=$HTTP_PROXY --build-arg http_proxy=$HTTP_PROXY \
    --build-arg HTTP_PROXY=$HTTPS_PROXY --build-arg HTTPS_PROXY=$HTTPS_PROXY \
    --build-arg NO_PROXY=$NO_PROXY --build-arg no_proxy=$NO_PROXY \
    --build-arg libtorch_version=${LIBTORCH_VERSION} \
    --build-arg clang_version=${CLANG_VERSION} \
    -t libtorch:${LIBTORCH_VERSION} -f libtorch.Dockerfile .
