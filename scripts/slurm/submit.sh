#!/bin/bash
# Submit a boomtitan SLURM job with cluster-specific defaults.
#
# Usage:
#   CLUSTER=leonardo scripts/slurm/submit.sh [sbatch-flags] <script> [script-args]
#   CLUSTER=cscs     scripts/slurm/submit.sh [sbatch-flags] <script> [script-args]
#
# Examples:
#   scripts/slurm/submit.sh scripts/slurm/lighthouse.sh
#   REPLICA_ID=0 GROUP_SIZE=2 scripts/slurm/submit.sh scripts/slurm/replica.sh
#   scripts/slurm/submit.sh --nodes=4 --gres=gpu:4 scripts/slurm/train.sh
#   CLUSTER=cscs scripts/slurm/submit.sh --nodes=4 scripts/slurm/train.sh

CLUSTER="${CLUSTER:-leonardo}"

case "${CLUSTER}" in
  leonardo)
    sbatch \
      --account=IscrB_Decentro \
      --partition=boost_usr_prod \
      "$@"
    ;;
  cscs)
    sbatch \
      --account=infra01 \
      --partition=normal \
      --environment=/iopsstor/scratch/cscs/mciccone/cuda128/cuda1281.toml \
      "$@"
    ;;
  *)
    echo "Unknown CLUSTER '${CLUSTER}'. Supported: leonardo, cscs"
    exit 1
    ;;
esac
