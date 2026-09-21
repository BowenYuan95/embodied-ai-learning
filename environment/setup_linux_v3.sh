#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="${ENV_NAME:-embodied}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.11.0}"
CUDA_WHEEL="${CUDA_WHEEL:-cu128}"
LEROBOT_VERSION="${LEROBOT_VERSION:-0.6.1}"
MANISKILL_VERSION="${MANISKILL_VERSION:-3.0.1}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
PROJECT_REPO_URL="${PROJECT_REPO_URL:-}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/Projects/embodied-ai-learning}"

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || fail "This installer supports Linux only."

log "Installing Ubuntu system dependencies"
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential ca-certificates cmake curl ffmpeg git git-lfs \
    libegl1 libgl1 libglib2.0-0 libusb-1.0-0 pkg-config \
    udev unzip wget
git lfs install --skip-repo

log "Checking NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi was not found. Install the Ubuntu NVIDIA driver and reboot, then rerun this script."
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

log "Installing or loading Miniforge"
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -x "$MINIFORGE_DIR/bin/conda" ]]; then
    CONDA_BASE="$MINIFORGE_DIR"
else
    installer_path="$(mktemp --suffix=.sh)"
    machine="$(uname -m)"
    case "$machine" in x86_64|aarch64) ;; *) fail "Unsupported CPU architecture: $machine" ;; esac
    curl -fL --retry 3 \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${machine}.sh" \
        -o "$installer_path"
    bash "$installer_path" -b -p "$MINIFORGE_DIR"
    rm -f "$installer_path"
    CONDA_BASE="$MINIFORGE_DIR"
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

log "Creating conda environment: $ENV_NAME"
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Environment already exists; updating it in place."
else
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"

log "Installing FFmpeg inside the environment"
conda install -y -c conda-forge "ffmpeg=7.1.1"

log "Updating Python packaging tools"
python -m pip install --upgrade pip setuptools wheel

log "Installing PyTorch $PYTORCH_VERSION ($CUDA_WHEEL)"
python -m pip install --index-url "https://download.pytorch.org/whl/$CUDA_WHEEL" \
    "torch==$PYTORCH_VERSION" torchvision torchaudio

log "Installing LeRobot, SO-101, dataset, training and notebook support"
python -m pip install \
    "lerobot[dataset,training,feetech,notebook]==$LEROBOT_VERSION"

log "Installing ManiSkill and analysis dependencies"
python -m pip install \
    "mani_skill==$MANISKILL_VERSION" \
    h5py matplotlib pandas pyarrow scipy seaborn tqdm ipywidgets nbformat

log "Registering Jupyter kernel"
python -m ipykernel install --user --name "$ENV_NAME" \
    --display-name "Python ($ENV_NAME)"

log "Configuring serial-port access for SO-101"
if getent group dialout >/dev/null 2>&1; then
    sudo usermod -aG dialout "$USER"
fi
sudo tee /etc/udev/rules.d/99-lerobot.rules >/dev/null <<'RULES'
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", MODE="0660", GROUP="dialout"
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", MODE="0660", GROUP="dialout"
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", MODE="0660", GROUP="dialout"
RULES
sudo udevadm control --reload-rules
sudo udevadm trigger

if [[ -n "$PROJECT_REPO_URL" ]]; then
    log "Deploying project repository"
    mkdir -p "$(dirname "$PROJECT_DIR")"
    if [[ -d "$PROJECT_DIR/.git" ]]; then
        git -C "$PROJECT_DIR" pull --ff-only
    elif [[ -e "$PROJECT_DIR" ]]; then
        fail "PROJECT_DIR exists but is not a git repository: $PROJECT_DIR"
    else
        git clone "$PROJECT_REPO_URL" "$PROJECT_DIR"
    fi
else
    log "PROJECT_REPO_URL is empty; repository clone skipped"
fi

log "Running verification"
python - <<'PY'
from importlib.metadata import version
import shutil
import subprocess
import torch
import mani_skill
import lerobot
import h5py
import pandas
import pyarrow
import serial

print("Python dependencies: PASS")
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA verification failed")
print("GPU:", torch.cuda.get_device_name(0))
print("ManiSkill:", version("mani-skill"))
print("LeRobot:", lerobot.__version__)
print("PySerial:", serial.__version__)
print("FFmpeg:", subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0])
print("Jupyter:", shutil.which("jupyter"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
print("LeRobotDataset import: PASS")
PY

python -m pip check

cat <<EOF

======================================
 Embodied AI V3 setup completed
======================================
Environment: $ENV_NAME
Activate:    conda activate $ENV_NAME
Notebook:    jupyter lab

IMPORTANT: log out and back in before using SO-101 serial ports so the
new dialout-group membership takes effect.
EOF
