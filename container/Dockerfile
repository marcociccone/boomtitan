# Use the official NVIDIA CUDA devel image as the base
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Set environment variables to prevent interactive prompts during installation
ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12, pip, git, and venv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    python3.12-venv \
    python3.12-dev \
    git \
    vim \
    wget \
    gcc-11 \
    g++-11 \
    build-essential \
    ca-certificates \
    curl \
    git \
    wget \
    sudo \
    vim \
    jq \
    vim \
    unzip \
    gdb \
    rsync \
    libssl-dev \
    zip \
    unzip \
    parallel \
    cmake \
    iperf3 \
    pciutils \
    protobuf-compiler \
    libprotobuf-dev \
    tmux \
    net-tools \
    && \
    rm -rf /var/lib/apt/lists/*

# Set gcc-11/g++-11 as the default compilers to ensure compatibility with PyTorch/Triton
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# Explicitly set the C and C++ compilers to be used by build systems
ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11

# Create a symbolic link so that `python` points to `python3`
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/python3.12-config /usr/bin/python3-config
RUN ln -s /usr/bin/python3.12-config /usr/bin/python-config

RUN python3 -m venv /opt/titan
ENV PATH="/opt/titan/bin:$PATH"

# -- Install torch
# RUN pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# -- Use nightly pytorch 
# we need this fix https://github.com/pytorch/pytorch/commit/7f4c3e7d2f3989115474d77936915fab6d6e0d93
# torchft triggers an error as some tensors sometimes end up as size zero. It happens when number or ranks is not power of 2
RUN pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
RUN pip install torchdata datasets tomli tensorboard tabulate wandb fsspec tyro tokenizers safetensors pybind11
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Install torchao for optimizations (float8, etc.)
RUN pip install torchao

# -- TORCHFT 
# Install Rust for installing torchft
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# Add Rust to PATH
ENV PATH="/root/.cargo/bin:${PATH}"
# Verify installation
RUN rustc --version
RUN rustup default stable
# Install torchft 
# RUN python -m pip install git+https://github.com/meta-pytorch/torchft.git@73dafea422116b2a713542719c8454efe5444c6c
# -- Use my fork to revert last commit that breaks torchft 
RUN python -m pip install git+https://github.com/marcociccone/torchft.git@96bff1c9b2e035a45bd7df2b1d81f1519442f015
RUN python -c "import torchft; print('torchft installed successfully')"

# Copy dotfiles to the user directory (nice to have for debugging inside the container)
RUN git clone https://github.com/marcociccone/dotfiles.git /users/mciccone/dotfiles
RUN cp -r /users/mciccone/dotfiles/.* /users/mciccone/ 2>/dev/null || true

# Install Python 3.12, pip, git, and venv
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     && \
#     rm -rf /var/lib/apt/lists/*

# Set a working directory inside the container
WORKDIR /workspace

# Set the default command to launch a bash shell
CMD ["/bin/bash"]

