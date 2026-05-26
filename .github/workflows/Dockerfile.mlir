FROM quay.io/pypa/manylinux_2_28_x86_64:2025.05.16-1

RUN rm -rf /opt/_internal/pipx/venvs/cmake \
    && dnf install -y \
        ca-certificates \
        cmake \
        curl \
        dnf-plugins-core \
        libxml2-devel \
        libzstd-devel \
        ninja-build \
        pkgconf-pkg-config \
        zlib-devel \
    && dnf config-manager --add-repo \
        https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf install -y cuda-toolkit-13-2 \
    && dnf clean all \
    && rm -rf /var/cache/dnf

RUN /opt/python/cp312-cp312/bin/python -m pip install --no-cache-dir lit \
    && ln -s /opt/python/cp312-cp312/bin/lit /usr/local/bin/lit

ENV CUDA_PATH=/usr/local/cuda-13.2
ENV CUDA_HOME=/usr/local/cuda-13.2
ENV CUDAToolkit_ROOT=/usr/local/cuda-13.2
ENV CMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
ENV PATH=/usr/local/cuda-13.2/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:/usr/local/cuda-13.2/lib64/stubs

RUN ln -sf libcuda.so /usr/local/cuda-13.2/lib64/stubs/libcuda.so.1
