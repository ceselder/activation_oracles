#!/usr/bin/env bash
# Setup script for RTX 5090 (Blackwell / sm_120) GPUs.
# Requires CUDA 12.8+ driver installed on the host.
#
# Usage:
#   export HF_TOKEN="hf_..."
#   export WANDB_KEY="..."
#   bash setup_5090.sh

set -euo pipefail

apt-get update -y
apt-get install -y build-essential unzip zip nvtop tmux
export CC=gcc

pip install uv
uv venv --python 3.12
source .venv/bin/activate

# 1. Install PyTorch with CUDA 12.8 (sm_120 support)
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. Install flash-attn (FA2 builds with sm_120 when CUDA 12.8+ is present)
#    If the pre-built wheel fails, uncomment the next line to build from source:
#    TORCH_CUDA_ARCH_LIST="12.0" uv pip install flash-attn --no-build-isolation
uv pip install flash-attn --no-build-isolation

# 3. Install the project and remaining deps
uv pip install -e .

# 4. (Optional) Install vllm for evals — use latest nightly for Blackwell support
# uv pip install vllm --index-url https://download.pytorch.org/whl/cu128

# 5. Login to services
if [ -n "${WANDB_KEY:-}" ]; then
    wandb login "$WANDB_KEY"
fi

if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN"
fi

echo "Setup complete. Activate with: source .venv/bin/activate"
