#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="embodied"
PYTHON_VERSION="3.12"

echo "======================================"
echo " Embodied AI Linux Environment Setup"
echo "======================================"

echo "[1/6] Installing system packages..."

sudo apt update

sudo apt install -y \
    wget \
    git \
    build-essential \
    cmake \
    pkg-config \
    libvulkan1 \
    vulkan-tools


echo "[2/6] Checking conda..."

if ! command -v conda >/dev/null 2>&1; then

    echo "Conda not found."
    echo "Please install Miniforge first."

    exit 1

fi


source "$(conda info --base)/etc/profile.d/conda.sh"


echo "[3/6] Creating conda environment..."

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then

    echo "Environment already exists."

else

    conda create \
        -n "$ENV_NAME" \
        python="$PYTHON_VERSION" \
        -y

fi


conda activate "$ENV_NAME"


echo "[4/6] Updating Python tools..."

python -m pip install --upgrade \
    pip \
    setuptools \
    wheel


echo "[5/6] Installing PyTorch CUDA 12.8..."

python -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0 \
    torchvision


echo "[6/6] Installing ManiSkill..."

python -m pip install \
    mani_skill


echo "======================================"
echo " Verification"
echo "======================================"

python - <<'PY'

import torch
import mani_skill

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("ManiSkill: OK")

PY


echo "Setup completed."

