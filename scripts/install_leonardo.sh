#!/bin/bash
# =============================================================================
# install_leonardo.sh — one-time setup for boomtitan on Leonardo @ CINECA
#
# Run once from the login node:
#   bash scripts/install_leonardo.sh
# =============================================================================

set -euo pipefail

VENV_PATH="/leonardo_scratch/fast/IscrB_Decentro/mciccone/envs/boom"
BOOMTITAN_ROOT="/leonardo/home/userexternal/mciccone/exp/boomtitan"
TORCHFT_ROOT="/leonardo/home/userexternal/mciccone/exp/torchft_boom"

# ---- 1. Modules -------------------------------------------------------------
# Note: no cuda module — PyTorch bundles its own CUDA runtime.
# Loading cuda/12.x causes NVML driver/library mismatch on nodes with older drivers.
module load python/3.11.7

# ---- 2. Activate existing venv ----------------------------------------------
source "${VENV_PATH}/bin/activate"
echo "Activated venv at ${VENV_PATH}"

# ---- 3. Upgrade pip + build tools ------------------------------------------
pip install --upgrade pip wheel setuptools --quiet

# ---- 4. PyTorch (CUDA 12.6) -------------------------------------------------
echo "Installing PyTorch (cu126)..."
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126 \
    --quiet

# ---- 5. Base requirements ---------------------------------------------------
echo "Installing base requirements..."
pip install \
    torchdata \
    "datasets>=3.6.0" \
    tomli \
    tensorboard \
    tabulate \
    wandb \
    fsspec \
    tyro \
    "tokenizers>=0.15.0" \
    safetensors \
    pybind11 \
    --quiet

# ---- 6. torchao (float8 support, cu126 wheel) -------------------------------
echo "Installing torchao..."
pip install torchao \
    --index-url https://download.pytorch.org/whl/cu126 \
    --quiet

# ---- 7. Rust (required for torchft) ----------------------------------------
if ! command -v cargo &>/dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh -s -- -y --quiet
fi
source "${HOME}/.cargo/env"

# ---- 8. torchft from source -------------------------------------------------
if [[ ! -d "${TORCHFT_ROOT}" ]]; then
    echo "Cloning torchft..."
    git clone https://github.com/pytorch/torchft.git "${TORCHFT_ROOT}"
fi
echo "Installing torchft..."
pip install -e "${TORCHFT_ROOT}" --quiet

# ---- 9. boomtitan (editable install) ----------------------------------------
echo "Installing boomtitan..."
pip install -e "${BOOMTITAN_ROOT}" --quiet

# ---- Done -------------------------------------------------------------------
echo ""
echo "Installation complete. Verifying..."
python -c "import torch, torchft, torchao, torchtitan; print(f'torch={torch.__version__}  torchao={torchao.__version__}  torchft OK  torchtitan OK')"
echo ""
echo "To activate in future sessions:"
echo "  source ${VENV_PATH}/bin/activate"
