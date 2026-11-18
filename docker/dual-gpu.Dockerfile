# syntax=docker/dockerfile:1
# Opt-in dual-card (RTX 5090 + Radeon AI PRO R9700) builder: the CUDA builder image plus a ROCm
# tree. Built by scripts/dev-setup-dual-gpu.sh with --build-context rocm=<host ROCm tree>.
ARG BASE_IMAGE=local/ninfer-builder:5090
FROM ${BASE_IMAGE} AS dual-gpu

RUN apt-get update && apt-get install --yes --no-install-recommends \
    libnuma1 libdrm2 libelf1 libtinfo6 libexpat1 libffi8 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=rocm / /opt/rocm/core-10.0/
ENV ROCM_PATH=/opt/rocm/core-10.0
ENV HIP_PATH=/opt/rocm/core-10.0
ENV PATH=/opt/rocm/core-10.0/bin:/opt/rocm/core-10.0/lib/llvm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib:/opt/rocm/core-10.0/lib/llvm/lib:/opt/rocm/core-10.0/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH}
LABEL org.ninfer.platform=sm_120a+gfx1201
WORKDIR /src
CMD ["sleep", "infinity"]
