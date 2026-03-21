#!/bin/bash
# =============================================================================
# setup_env_cscs.sh  —  source this at the top of every SLURM script
# CSCS Alps (Titan node)
#
# NOTE: CSCS uses a container image for CUDA; add this to the job script header:
#   #SBATCH --environment=/iopsstor/scratch/cscs/mciccone/cuda128/cuda1281.toml
# This line cannot be set dynamically, so add it manually when targeting CSCS.
# =============================================================================

# ---- Python environment (provided by container) ----------------------------
source /opt/titan/bin/activate

# ---- Working directory -----------------------------------------------------
export BOOMTITAN_ROOT="${BOOMTITAN_ROOT:-/iopsstor/scratch/cscs/mciccone/exp/boomtitan}"
cd "${BOOMTITAN_ROOT}"

# ---- Paths -----------------------------------------------------------------
export HF_HOME="${HF_HOME:-${SCRATCH}/huggingface}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-${SCRATCH}/checkpoints}"
export LOG_PATH="${LOG_PATH:-${SCRATCH}/logs}"

# ---- W&B -------------------------------------------------------------------
export WANDB_PROJECT="${WANDB_PROJECT:-boomtitan}"
export WANDB_ENTITY="${WANDB_ENTITY:-mciccone}"
# CSCS has internet access — leave WANDB_MODE unset (online by default)

# ---- SLURM account / partition ---------------------------------------------
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-infra01}"
export SLURM_PARTITION="${SLURM_PARTITION:-normal}"

# ---- NCCL (CSCS uses NVLink/RoCE, not InfiniBand) -------------------------
export NCCL_P2P_LEVEL=SYS
export NCCL_DEBUG=INFO

# ---- Misc ------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONFAULTHANDLER=1
