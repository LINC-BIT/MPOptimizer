LIBTORCH_VERSION=1.10.2
CLANG_VERSION=14
docker build \
    --build-arg libtorch_version=${LIBTORCH_VERSION} \
    --build-arg clang_version=${CLANG_VERSION} \
    -t ldt:1.00 -f ldt.Dockerfile .