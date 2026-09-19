#!/usr/bin/env bash

set -euo pipefail


ENV_NAME="embodied"
PYTHON_VERSION="3.12"


echo "======================================"
echo " Embodied AI V3 Environment Setup"
echo "======================================"


echo "[1/7] Installing system packages..."

sudo apt update

sudo apt install -y \
    wget \
    git \
    build-essential \
    cmake \
    pkg-config \
    libvulkan1 \
    vulkan-tools \
    ffmpeg


echo "[2/7] Checking conda..."

if ! command -v conda >/dev/null 2>&1
then
    echo "Conda not found."
    echo "Please install Miniforge first."
    exit 1
fi


source "$(conda info --base)/etc/profile.d/conda.sh"


echo "[3/7] Creating conda environment..."


if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"
then
    echo "Environment already exists."

else

    conda create \
        -n "$ENV_NAME" \
        python="$PYTHON_VERSION" \
        -y

fi


conda activate "$ENV_NAME"



echo "[4/7] Updating Python tools..."


python -m pip install --upgrade \
    pip \
    setuptools \
    wheel



echo "[5/7] Installing PyTorch CUDA 12.8..."


python -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0 \
    torchvision



echo "[6/7] Installing Embodied AI stack..."


# Simulation
python -m pip install \
    mani_skill


# Robot learning dataset format
python -m pip install \
    lerobot==0.6.1


# Dataset processing
python -m pip install \
    h5py \
    numpy \
    pandas \
    pyarrow



# Development tools
python -m pip install \
    jupyter \
    ipykernel



echo "[7/7] Verification"

python - <<'PY'

import torch
import mani_skill
import lerobot
import h5py
import pandas


print("==============================")
print("Environment Check")
print("==============================")

print("PyTorch:",
      torch.__version__)

print("CUDA runtime:",
      torch.version.cuda)

print("CUDA available:",
      torch.cuda.is_available())


if torch.cuda.is_available():
    print("GPU:",
          torch.cuda.get_device_name(0))


print("ManiSkill: OK")

print("LeRobot:",
      lerobot.__version__)

print("h5py: OK")

print("pandas: OK")


PY


echo "======================================"
echo " Embodied AI V3 Setup Completed"
echo "======================================"