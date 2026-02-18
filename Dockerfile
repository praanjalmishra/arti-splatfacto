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

# UID/GID mapping
ARG DOCKER_UID=1000
ARG DOCKER_GID=1000

RUN groupadd -g ${DOCKER_GID} appgroup && \
    useradd -m -u ${DOCKER_UID} -g ${DOCKER_GID} -s /bin/bash appuser

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
    libopengl0 \
    libegl1 \
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

# ============================================================================
# Stable NLP / SAM2 / LightGlue Dependency Block
# (Torch 2.1.2 + CUDA 11.8 Compatible)
# ============================================================================

# --- Core runtime deps ---
RUN pip install --no-cache-dir \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    timm==0.9.12 \
    pillow \
    matplotlib \
    opencv-python \
    regex \
    safetensors \
    --no-deps

# --- Transformers stack (Pinned for Torch 2.1.2) ---
RUN pip install --no-cache-dir \
    transformers==4.38.2 \
    tokenizers==0.15.2 \
    huggingface-hub==0.20.3 \
    --no-deps

# --- LightGlue (no dependency override) ---
RUN pip install --no-cache-dir \
    git+https://github.com/cvg/LightGlue@035612541779b17897aa06d6ff19cb4060111616 \
    --no-deps

# --- SAM2 (install normally but ignore its torch requirements) ---
WORKDIR /workspace/arti-splatfacto/third_party/sam2
RUN pip install --no-cache-dir . --no-deps

WORKDIR /workspace/arti-splatfacto/qed-splatter
RUN pip install --no-cache-dir . --no-deps


RUN pip install --no-cache-dir antlr4-python3-runtime==4.9.3 --no-deps
pip install kornia==0.7.2 kornia-rs==0.1.9 --no-deps


# Install CLI completions
RUN ns-install-cli --mode install 2>/dev/null || true

# ============================================================================
# Verification
# ============================================================================
RUN python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" && \
    python -c "import pytorch3d; print('✓ PyTorch3D')" && \
    python -c "from arti_splatfacto.config import arti_splatfacto_config; print('✓ ArtiSplatfacto')" && \
    echo "✅ Build verification passed. GPU will be checked at runtime."



# Create cache directories with proper permissions
RUN mkdir -p /home/appuser/.cache/torch_extensions /home/appuser/.cache/torch && \
    chown -R appuser:appgroup /home/appuser/.cache && \
    chown -R appuser:appgroup /workspace

# Suppress non-critical warnings in appuser's bashrc
RUN echo 'export PYMESHLAB_DISABLE_PLUGIN_WARNINGS=1' >> /home/appuser/.bashrc && \
    echo 'export PYTHONWARNINGS="ignore"' >> /home/appuser/.bashrc

# ============================================================================
# Switch to non-root user
# ============================================================================
USER appuser

# ============================================================================
# Runtime Configuration
# ============================================================================
WORKDIR /workspace

# Expose viewer port
EXPOSE 7007

# Default command
CMD ["/bin/bash"]