#!/bin/bash
#SBATCH --job-name=boomtitan-replica
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
#SBATCH --output=%x-%j-replica${REPLICA_ID}.out
#SBATCH --error=%x-%j-replica${REPLICA_ID}.err

# =============================================================================
# boomtitan multi-replica TorchFT training (diloco / local_sgd / hsdp)
#
# Run lighthouse.sh first, then submit one job per replica:
#   sbatch scripts/slurm/lighthouse.sh
#   REPLICA_ID=0 GROUP_SIZE=2 SYNC_METHOD=diloco sbatch scripts/slurm/replica.sh
#   REPLICA_ID=1 GROUP_SIZE=2 SYNC_METHOD=diloco sbatch scripts/slurm/replica.sh
#
# Override nodes/GPUs per replica:
#   REPLICA_ID=0 GROUP_SIZE=2 sbatch --nodes=4 --gres=gpu:4 scripts/slurm/replica.sh
#
# Set lighthouse address explicitly (skips auto-discovery):
#   TORCHFT_LIGHTHOUSE=http://<host>:29510 REPLICA_ID=0 sbatch scripts/slurm/replica.sh
#
# CSCS (add --environment line to header first):
#   CLUSTER=cscs REPLICA_ID=0 GROUP_SIZE=2 \
#     sbatch --account=infra01 --partition=normal scripts/slurm/replica.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/slurm"
source "${SCRIPT_DIR}/setup_env_${CLUSTER:-leonardo}.sh"

# ---- Replica identity -------------------------------------------------------
REPLICA_ID="${REPLICA_ID:-0}"
GROUP_SIZE="${GROUP_SIZE:-2}"
SYNC_METHOD="${SYNC_METHOD:-local_sgd}"
SYNC_STEPS="${SYNC_STEPS:-30}"

# Auto-discover lighthouse if not set explicitly
if [[ -z "${TORCHFT_LIGHTHOUSE:-}" ]]; then
    LIGHTHOUSE_ADDR_FILE="${BASE_CHECKPOINT_PATH}/lighthouse_addr.txt"
    if [[ -f "${LIGHTHOUSE_ADDR_FILE}" ]]; then
        export TORCHFT_LIGHTHOUSE=$(cat "${LIGHTHOUSE_ADDR_FILE}")
        echo "Auto-discovered lighthouse at: ${TORCHFT_LIGHTHOUSE}"
    else
        echo "ERROR: TORCHFT_LIGHTHOUSE not set and ${LIGHTHOUSE_ADDR_FILE} not found."
        echo "Start lighthouse.sh first, or set TORCHFT_LIGHTHOUSE manually."
        exit 1
    fi
fi

# ---- Cluster topology -------------------------------------------------------
NUM_NODES="${SLURM_NNODES:-2}"
NUM_GPUS="${NUM_GPUS:-4}"
# Different rdzv port per replica to avoid collisions on shared clusters
RDZV_PORT=$(( 29500 + REPLICA_ID ))

# Resolve head node IP
head_node=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "${head_node}" hostname --ip-address)
echo "Head node: ${head_node} (${head_node_ip}:${RDZV_PORT})"

# ---- Training config --------------------------------------------------------
CONFIG_FILE="${CONFIG_FILE:-./torchtitan/models/boom/train_configs/sl3-datamix/smollm3B.toml}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-2}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"
DP_PARALLEL_SHARD_DEGREE=$(( NUM_NODES * NUM_GPUS ))
REPLICA_BATCH_SIZE=$(( LOCAL_BATCH_SIZE * NUM_GPUS * NUM_NODES * GRAD_ACC_STEPS ))
GLOBAL_BATCH_SIZE=$(( REPLICA_BATCH_SIZE * GROUP_SIZE ))

# ---- Output paths -----------------------------------------------------------
# CHECKPOINT_OVERRIDE allows launch_replicas.sh to set a stable named path.
DUMP_FOLDER="${CHECKPOINT_OVERRIDE:-${BASE_CHECKPOINT_PATH}/${SLURM_JOB_NAME}_replica${REPLICA_ID}}"
LOG_DIR="${LOG_PATH}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_replica${REPLICA_ID}"
mkdir -p "${DUMP_FOLDER}" "${LOG_DIR}"

# wandb run name — set WANDB_NAME env var (picked up automatically by wandb)
[[ -n "${WANDB_RUN_NAME:-}" ]] && export WANDB_NAME="${WANDB_RUN_NAME}"

# ---- TorchFT timeouts -------------------------------------------------------
export TORCHFT_QUORUM_TIMEOUT_SEC="${TORCHFT_QUORUM_TIMEOUT_SEC:-900}"
export TORCHFT_TIMEOUT_SEC="${TORCHFT_TIMEOUT_SEC:-600}"
export TORCHFT_WATCHDOG_TIMEOUT_SEC="${TORCHFT_WATCHDOG_TIMEOUT_SEC:-300}"

echo "================================================"
echo "boomtitan replica ${REPLICA_ID} / ${GROUP_SIZE}"
echo "  Cluster:     ${CLUSTER:-leonardo}"
echo "  Lighthouse:  ${TORCHFT_LIGHTHOUSE}"
echo "  Sync method: ${SYNC_METHOD}  (sync_steps=${SYNC_STEPS})"
echo "  Config:      ${CONFIG_FILE}"
echo "  Nodes:       ${NUM_NODES} x ${NUM_GPUS} GPUs  DP_shard=${DP_PARALLEL_SHARD_DEGREE}"
echo "  local_bs:    ${LOCAL_BATCH_SIZE}  replica_bs: ${REPLICA_BATCH_SIZE}  global_bs: ${GLOBAL_BATCH_SIZE}"
echo "  Dump folder: ${DUMP_FOLDER}"
echo "================================================"

# ---- Base torchrun command --------------------------------------------------
TORCHRUN_CMD=(
    torchrun
    --nproc_per_node="${NUM_GPUS}"
    --nnodes="${NUM_NODES}"
    --rdzv_id="${SLURM_JOB_ID}"
    --rdzv_backend=c10d
    --rdzv_endpoint="${head_node_ip}:${RDZV_PORT}"
    --role=rank
    --tee=3
    -m torchtitan.train
    --job.config_file "${CONFIG_FILE}"
    --job.dump_folder "${DUMP_FOLDER}"
    --training.local_batch_size "${LOCAL_BATCH_SIZE}"
    --training.global_batch_size "${GLOBAL_BATCH_SIZE}"
    --parallelism.data_parallel_replicate_degree 1
    --parallelism.data_parallel_shard_degree "${DP_PARALLEL_SHARD_DEGREE}"
    --fault_tolerance.enable
    --fault_tolerance.group_size "${GROUP_SIZE}"
    --fault_tolerance.replica_id "${REPLICA_ID}"
    --fault_tolerance.process_group_timeout_ms 300000
    --comm.train_timeout_seconds 300
)

if [[ "${SYNC_METHOD}" == "hsdp" ]]; then
    : # hsdp uses default inner optimizer sync — no extra flags needed
elif [[ "${SYNC_METHOD}" == "local_sgd" || "${SYNC_METHOD}" == "diloco" ]]; then
    TORCHRUN_CMD+=(
        --fault_tolerance.sync_steps "${SYNC_STEPS}"
        --fault_tolerance.semi_sync_method "${SYNC_METHOD}"
    )
else
    echo "ERROR: Unsupported SYNC_METHOD '${SYNC_METHOD}'. Use: hsdp, local_sgd, diloco"
    exit 1
fi

srun --kill-on-bad-exit=1 \
    --output="${LOG_DIR}/out.%N.%j.%t" \
    --error="${LOG_DIR}/err.%N.%j.%t" \
    "${TORCHRUN_CMD[@]}" \
    "$@"
