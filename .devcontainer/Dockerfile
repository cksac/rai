# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:24.01-py3

ARG LLVM_VERSION=17
ARG IREE_VERSION=20231218.742
ARG LIB_IREE_COMPILER=/usr/local/lib/python3.10/dist-packages/iree/compiler/_mlir_libs/

# install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain stable -y

RUN apt update -y

RUN apt-get install -y libzstd-dev

# llvm
RUN apt install -y lsb-release wget software-properties-common gnupg
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh
RUN ./llvm.sh ${LLVM_VERSION} all

# mlir
RUN apt-get install -y libmlir-${LLVM_VERSION}-dev mlir-${LLVM_VERSION}-tools

# llvm-config symlink
RUN ln -s /bin/llvm-config-${LLVM_VERSION} /bin/llvm-config

# iree
RUN pip install iree-compiler==${IREE_VERSION} iree-runtime==${IREE_VERSION}

ENV LIB_IREE_COMPILER ${LIB_IREE_COMPILER}
ENV PATH=/root/.cargo/bin:$PATH

