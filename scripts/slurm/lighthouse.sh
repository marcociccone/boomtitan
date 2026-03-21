#!/bin/bash
#SBATCH --job-name=boomtitan-lighthouse
#SBATCH -A IscrB_Decentro
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# =============================================================================
# TorchFT Lighthouse coordinator — start this BEFORE any replica jobs.
#
# Submit:
#   sbatch scripts/slurm/lighthouse.sh
#
# Override account/partition/cluster at submit time:
#   CLUSTER=cscs sbatch --account=infra01 --partition=normal scripts/slurm/lighthouse.sh
#
# After this job starts the lighthouse address is written to:
#   ${BASE_CHECKPOINT_PATH}/lighthouse_addr.txt
# replica.sh will auto-discover it from there.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/slurm"
source "${SCRIPT_DIR}/setup_env_${CLUSTER:-leonardo}.sh"

LIGHTHOUSE_PORT="${LIGHTHOUSE_PORT:-29510}"
MIN_REPLICAS="${MIN_REPLICAS:-1}"   # can proceed with 1 replica (safe default)

echo "================================================"
echo "Lighthouse starting on $(hostname):${LIGHTHOUSE_PORT}"
echo "Min replicas: ${MIN_REPLICAS}"
echo "Cluster: ${CLUSTER:-leonardo}"
echo "================================================"

# Write address to shared file so replica jobs can discover it automatically
LIGHTHOUSE_ADDR_FILE="${BASE_CHECKPOINT_PATH}/lighthouse_addr.txt"
mkdir -p "${BASE_CHECKPOINT_PATH}"
echo "http://$(hostname):${LIGHTHOUSE_PORT}" > "${LIGHTHOUSE_ADDR_FILE}"
echo "Address written to: ${LIGHTHOUSE_ADDR_FILE}"

torchft_lighthouse \
    --bind "[::]:${LIGHTHOUSE_PORT}" \
    --min_replicas "${MIN_REPLICAS}" \
    --quorum_tick_ms 500 \
    --join_timeout_ms 30000
