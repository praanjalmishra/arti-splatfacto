

# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/yourusername/arti-splatfacto"
LABEL org.opencontainers.image.description="ArtiSplatfacto - Articulated 3D Gaussian Splatting"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Build arguments
ARG CUDA_ARCHITECTURES="90;89;86;80;75;70;61"
ARG PYTHON_VERSION=3.10

ENV DEBIAN_FRONTEND=noninteractive
ENV TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
ENV PYTHONUNBUFFERED=1

# ============================================================================
# System Dependencies
# ============================================================================
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libffi-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Make python default
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ============================================================================
# PyTorch 2.1.2 + CUDA 11.8
# ============================================================================
RUN pip install --no-cache-dir \
    torch==2.1.2+cu118 \
    torchvision==0.16.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Pin NumPy < 2.0 immediately
RUN pip install --no-cache-dir "numpy<2.0.0"

# ============================================================================
# PyTorch3D
# ============================================================================
RUN pip install --no-cache-dir fvcore iopath && \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" \
    MAX_JOBS=4 \
    CXXFLAGS="-std=c++17" \
    pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# ============================================================================
# tiny-cuda-nn
# ============================================================================
RUN pip install --no-cache-dir ninja && \
    TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
    pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch"

# ============================================================================
# Nerfstudio (from main branch for latest features)
# ============================================================================
RUN pip install --no-cache-dir \
    "git+https://github.com/nerfstudio-project/nerfstudio.git@main" && \
    pip install --no-cache-dir --force-reinstall "numpy<2.0.0"



# Suppress non-critical warnings
RUN echo 'export PYMESHLAB_DISABLE_PLUGIN_WARNINGS=1' >> /root/.bashrc && \
    echo 'export PYTHONWARNINGS="ignore"' >> /root/.bashrc
# ============================================================================
# ArtiSplatfacto
# ============================================================================
WORKDIR /workspace

# Copy source code
COPY . /workspace/arti-splatfacto

# Install ArtiSplatfacto
WORKDIR /workspace/arti-splatfacto
RUN pip install --no-cache-dir -e . --no-deps && \
    pip install --no-cache-dir --force-reinstall "numpy<2.0.0"

# Install CLI completions
RUN ns-install-cli --mode install 2>/dev/null || true

# ============================================================================
# Verification
# ============================================================================

RUN python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" && \
    python -c "import pytorch3d; print('✓ PyTorch3D')" && \
    python -c "from arti_splatfacto.config import arti_splatfacto_config; print('✓ ArtiSplatfacto')" && \
    echo "✅ Build verification passed. GPU will be checked at runtime."

# ============================================================================
# Runtime Configuration
# ============================================================================
WORKDIR /workspace

# Expose viewer port
EXPOSE 7007

# Default command
CMD ["/bin/bash"]



# FROM nerfstudio-base:latest

# # Mount point for your code
# WORKDIR /workspace

# # Install ArtiSplatfacto at runtime (not build time)
# CMD ["/bin/bash"]