#!/bin/bash
# Launch all TorchFT replicas for a multi-replica training run.
#
# Computes LOCAL_BATCH_SIZE from GLOBAL_BATCH_SIZE / replica topology,
# optionally submits a lighthouse job, then submits one replica job per replica.
#
# Usage:
#   GLOBAL_BATCH_SIZE=512 NUM_REPLICAS=2 NUM_NODES=2 NUM_GPUS=4 \
#     bash scripts/slurm/launch_replicas.sh --name my_exp
#
#   bash scripts/slurm/launch_replicas.sh --resume my_exp   # continue existing run
#
# Skip lighthouse (already running):
#   START_LIGHTHOUSE=0 ... bash scripts/slurm/launch_replicas.sh --name my_exp
#
# Extra sbatch flags or torchtitan CLI overrides:
#   EXTRA_SBATCH_ARGS="--time=04:00:00" \
#   EXTRA_TRAIN_ARGS="--training.steps 500" \
#     bash scripts/slurm/launch_replicas.sh --name my_exp

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Parse args -------------------------------------------------------------
RUN_NAME=""
RESUME=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)    RUN_NAME="$2"; shift 2 ;;
        --resume)  RESUME=true; RUN_NAME="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash launch_replicas.sh [--name NAME | --resume NAME]"
            exit 1 ;;
    esac
done

# ---- Required parameter -----------------------------------------------------
if [[ -z "${GLOBAL_BATCH_SIZE:-}" ]]; then
    echo "ERROR: GLOBAL_BATCH_SIZE is required."
    echo "  Example: GLOBAL_BATCH_SIZE=512 NUM_REPLICAS=2 bash $0 --name my_exp"
    exit 1
fi

# ---- Topology ---------------------------------------------------------------
NUM_REPLICAS="${NUM_REPLICAS:-2}"
NUM_NODES="${NUM_NODES:-2}"
NUM_GPUS="${NUM_GPUS:-4}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"

# ---- Training config --------------------------------------------------------
SYNC_METHOD="${SYNC_METHOD:-local_sgd}"
SYNC_STEPS="${SYNC_STEPS:-30}"
CONFIG_FILE="${CONFIG_FILE:-}"    # empty = replica.sh uses its own default

# ---- Submission config ------------------------------------------------------
START_LIGHTHOUSE="${START_LIGHTHOUSE:-1}"
CLUSTER="${CLUSTER:-leonardo}"
EXTRA_SBATCH_ARGS="${EXTRA_SBATCH_ARGS:-}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

# ---- Batch size math --------------------------------------------------------
COMPUTE_MULTIPLIER=$(( NUM_NODES * NUM_GPUS * GRAD_ACC_STEPS ))

if (( GLOBAL_BATCH_SIZE % NUM_REPLICAS != 0 )); then
    echo "ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} is not divisible by NUM_REPLICAS=${NUM_REPLICAS}"
    exit 1
fi
REPLICA_BATCH_SIZE=$(( GLOBAL_BATCH_SIZE / NUM_REPLICAS ))

if (( REPLICA_BATCH_SIZE % COMPUTE_MULTIPLIER != 0 )); then
    echo "ERROR: REPLICA_BATCH_SIZE=${REPLICA_BATCH_SIZE} is not divisible by"
    echo "  NUM_NODES * NUM_GPUS * GRAD_ACC_STEPS = ${COMPUTE_MULTIPLIER}"
    exit 1
fi
LOCAL_BATCH_SIZE=$(( REPLICA_BATCH_SIZE / COMPUTE_MULTIPLIER ))

# ---- Resolve paths ----------------------------------------------------------
# Source env to get BASE_CHECKPOINT_PATH (best-effort; may not work on login node)
source "${SCRIPT_DIR}/setup_env_${CLUSTER}.sh" 2>/dev/null || true
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/leonardo_scratch/fast/IscrB_Decentro/mciccone/checkpoints}"

# ---- Resolve run name -------------------------------------------------------
if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="boom_$(date +%Y%m%d_%H%M%S)"
    echo "Auto-generated run name: ${RUN_NAME}"
fi

RUNS_DIR="${BASE_CHECKPOINT_PATH}/runs"
RUN_CKPT_DIR="${RUNS_DIR}/${RUN_NAME}"

# ---- Validate resume/new ----------------------------------------------------
if [[ "${RESUME}" == true ]]; then
    if [[ ! -d "${RUN_CKPT_DIR}" ]]; then
        echo "ERROR: --resume requested but checkpoint dir not found: ${RUN_CKPT_DIR}"
        echo "Available runs:"
        ls "${RUNS_DIR}" 2>/dev/null || echo "  (none)"
        exit 1
    fi
    echo "Resuming run: ${RUN_NAME}"
    for (( i=0; i<NUM_REPLICAS; i++ )); do
        latest="${RUN_CKPT_DIR}/replica_${i}/step-latest"
        if [[ -d "${latest}" ]]; then
            echo "  replica_${i}: found checkpoint at $(readlink -f "${latest}")"
        fi
    done
else
    if [[ -d "${RUN_CKPT_DIR}" ]]; then
        echo "ERROR: Run '${RUN_NAME}' already exists at ${RUN_CKPT_DIR}"
        echo "Use --resume ${RUN_NAME} to continue, or choose a different name."
        exit 1
    fi
    echo "New run: ${RUN_NAME}"
    for (( i=0; i<NUM_REPLICAS; i++ )); do
        mkdir -p "${RUN_CKPT_DIR}/replica_${i}"
    done
fi

# ---- Summary ----------------------------------------------------------------
echo "================================================"
echo "boomtitan launch_replicas"
echo "  cluster:   ${CLUSTER}"
echo "  run:       ${RUN_NAME}"
[[ -n "${CONFIG_FILE}" ]] && \
echo "  config:    ${CONFIG_FILE}"
echo ""
echo "  topology:  ${NUM_REPLICAS} replicas x (${NUM_NODES} nodes x ${NUM_GPUS} GPUs x ${GRAD_ACC_STEPS} grad_acc)"
echo "  batch:     global=${GLOBAL_BATCH_SIZE} = ${NUM_REPLICAS} × replica=${REPLICA_BATCH_SIZE}"
echo "             = ${NUM_REPLICAS} × ${LOCAL_BATCH_SIZE} per_gpu_bs x (${NUM_NODES} nodes x ${NUM_GPUS} GPUs x ${GRAD_ACC_STEPS} grad_acc)"
echo "  sync:      ${SYNC_METHOD}  (every ${SYNC_STEPS} steps)"
echo "================================================"

SUBMIT="${SCRIPT_DIR}/submit.sh"
JOB_IDS=()

# ---- Lighthouse -------------------------------------------------------------
LIGHTHOUSE_ADDR_FILE="${BASE_CHECKPOINT_PATH}/lighthouse_addr.txt"

if [[ "${START_LIGHTHOUSE}" == "1" ]]; then
    echo "Submitting lighthouse..."
    # shellcheck disable=SC2086
    lighthouse_out=$(CLUSTER="${CLUSTER}" "${SUBMIT}" ${EXTRA_SBATCH_ARGS} \
        "${SCRIPT_DIR}/lighthouse.sh")
    echo "  ${lighthouse_out}"
    lighthouse_job_id=$(echo "${lighthouse_out}" | grep -oP '(?<=Submitted batch job )\d+' || true)
    [[ -n "${lighthouse_job_id}" ]] && JOB_IDS+=("lighthouse:${lighthouse_job_id}")
else
    # Verify lighthouse is reachable
    if [[ ! -f "${LIGHTHOUSE_ADDR_FILE}" ]]; then
        echo "ERROR: Lighthouse address file not found: ${LIGHTHOUSE_ADDR_FILE}"
        echo "Run: sbatch ${SCRIPT_DIR}/lighthouse.sh  or set START_LIGHTHOUSE=1"
        exit 1
    fi
    LIGHTHOUSE_ADDR=$(cat "${LIGHTHOUSE_ADDR_FILE}")
    LIGHTHOUSE_HOST=$(echo "${LIGHTHOUSE_ADDR}" | sed 's|http://||' | cut -d: -f1)
    LIGHTHOUSE_PORT_NUM=$(echo "${LIGHTHOUSE_ADDR}" | sed 's|http://||' | cut -d: -f2)
    REACHABLE=false
    for i in 1 2 3; do
        if timeout 5 bash -c "cat < /dev/null > /dev/tcp/${LIGHTHOUSE_HOST}/${LIGHTHOUSE_PORT_NUM}" 2>/dev/null; then
            REACHABLE=true; break
        fi
        [[ $i -lt 3 ]] && echo "  Lighthouse not reachable yet, retrying (${i}/3)..." && sleep 5
    done
    if [[ "${REACHABLE}" == false ]]; then
        echo "ERROR: Lighthouse at ${LIGHTHOUSE_ADDR} is not reachable."
        echo "  The address file may be stale. Start a new one with:"
        echo "  sbatch ${SCRIPT_DIR}/lighthouse.sh"
        exit 1
    fi
    echo "Lighthouse: ${LIGHTHOUSE_ADDR} (reachable)"
fi

# ---- Replicas ---------------------------------------------------------------
for (( i=0; i<NUM_REPLICAS; i++ )); do
    echo "Submitting replica ${i}..."
    submit_env=(
        CLUSTER="${CLUSTER}"
        REPLICA_ID="${i}"
        GROUP_SIZE="${NUM_REPLICAS}"
        LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE}"
        GRAD_ACC_STEPS="${GRAD_ACC_STEPS}"
        NUM_NODES="${NUM_NODES}"
        NUM_GPUS="${NUM_GPUS}"
        SYNC_METHOD="${SYNC_METHOD}"
        SYNC_STEPS="${SYNC_STEPS}"
        CHECKPOINT_OVERRIDE="${RUN_CKPT_DIR}/replica_${i}"
        WANDB_RUN_NAME="${RUN_NAME}_replica_${i}"
    )
    [[ -n "${CONFIG_FILE}" ]] && submit_env+=(CONFIG_FILE="${CONFIG_FILE}")

    # shellcheck disable=SC2086
    replica_out=$(env "${submit_env[@]}" "${SUBMIT}" \
        --nodes="${NUM_NODES}" --gres="gpu:${NUM_GPUS}" \
        ${EXTRA_SBATCH_ARGS} \
        "${SCRIPT_DIR}/replica.sh" \
        ${EXTRA_TRAIN_ARGS})
    echo "  ${replica_out}"
    replica_job_id=$(echo "${replica_out}" | grep -oP '(?<=Submitted batch job )\d+' || true)
    [[ -n "${replica_job_id}" ]] && JOB_IDS+=("replica${i}:${replica_job_id}")
done

# ---- Done -------------------------------------------------------------------
echo ""
echo "================================================"
echo "Run:         ${RUN_NAME}"
echo "Jobs:"
for entry in "${JOB_IDS[@]}"; do
    echo "  ${entry}"
done
echo "Checkpoints: ${RUN_CKPT_DIR}/replica_{0..$(( NUM_REPLICAS - 1 ))}/"
echo "Resume with:"
echo "  GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} NUM_REPLICAS=${NUM_REPLICAS} \\"
echo "  NUM_NODES=${NUM_NODES} NUM_GPUS=${NUM_GPUS} SYNC_METHOD=${SYNC_METHOD} \\"
[[ -n "${CONFIG_FILE}" ]] && echo "  CONFIG_FILE=${CONFIG_FILE} \\"
echo "  bash scripts/slurm/launch_replicas.sh --resume ${RUN_NAME}"
echo "================================================"
