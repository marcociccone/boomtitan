#!/bin/bash
#SBATCH --job-name=boomtitan-train
#SBATCH -A IscrB_Decentro
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR2@120
#SBATCH --open-mode=append
#SBATCH --distribution=block
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# =============================================================================
# boomtitan single-replica multi-node training (no TorchFT)
#
# Submit (Leonardo, 2 nodes x 4 GPUs):
#   sbatch scripts/slurm/train.sh
#
# Override nodes/GPUs:
#   sbatch --nodes=4 --gres=gpu:4 scripts/slurm/train.sh
#
# Override config and batch size:
#   CONFIG_FILE=./torchtitan/models/boom/train_configs/fw-edu-local/local_fw_edu_smollm3B_leonardo.toml \
#   NUM_NODES=4 NUM_GPUS=4 LOCAL_BATCH_SIZE=4 \
#   sbatch --nodes=4 --gres=gpu:4 scripts/slurm/train.sh
#
# CSCS (add --environment line to header first):
#   CLUSTER=cscs sbatch --account=infra01 --partition=normal \
#     --nodes=4 --gres=gpu:4 scripts/slurm/train.sh
#
# Pass extra torchtitan CLI overrides at the end:
#   sbatch scripts/slurm/train.sh --training.steps 500 --checkpoint.enable_checkpoint true
# =============================================================================

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/slurm"
source "${SCRIPT_DIR}/setup_env_${CLUSTER:-leonardo}.sh"

# ---- Cluster topology -------------------------------------------------------
NUM_NODES="${SLURM_NNODES:-2}"
NUM_GPUS="${NUM_GPUS:-4}"
RDZV_PORT="${RDZV_PORT:-29500}"

# Resolve head node IP (hostname may not be routable; IP always is)
head_node=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "${head_node}" hostname --ip-address)
echo "Head node: ${head_node} (${head_node_ip}:${RDZV_PORT})"

# ---- Training config --------------------------------------------------------
CONFIG_FILE="${CONFIG_FILE:-./torchtitan/models/boom/train_configs/sl3-datamix/smollm3B.toml}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-2}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"
GLOBAL_BATCH_SIZE=$(( LOCAL_BATCH_SIZE * NUM_GPUS * NUM_NODES * GRAD_ACC_STEPS ))

# ---- Output paths -----------------------------------------------------------
DUMP_FOLDER="${BASE_CHECKPOINT_PATH}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
LOG_DIR="${LOG_PATH}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${DUMP_FOLDER}" "${LOG_DIR}"

echo "================================================"
echo "boomtitan train"
echo "  Cluster:     ${CLUSTER:-leonardo}"
echo "  Config:      ${CONFIG_FILE}"
echo "  Nodes:       ${NUM_NODES} x ${NUM_GPUS} GPUs"
echo "  local_bs:    ${LOCAL_BATCH_SIZE}  grad_acc: ${GRAD_ACC_STEPS}  global_bs: ${GLOBAL_BATCH_SIZE}"
echo "  Dump folder: ${DUMP_FOLDER}"
echo "================================================"

srun --kill-on-bad-exit=1 \
    --output="${LOG_DIR}/out.%N.%j.%t" \
    --error="${LOG_DIR}/err.%N.%j.%t" \
  torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes="${NUM_NODES}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${head_node_ip}:${RDZV_PORT}" \
    --role=rank \
    --tee=3 \
    -m torchtitan.train \
    --job.config_file "${CONFIG_FILE}" \
    --job.dump_folder "${DUMP_FOLDER}" \
    --training.local_batch_size "${LOCAL_BATCH_SIZE}" \
    --training.global_batch_size "${GLOBAL_BATCH_SIZE}" \
    "$@"
