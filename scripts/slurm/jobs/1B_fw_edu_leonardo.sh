#!/bin/bash
# 1B model test run on Leonardo — FineWeb-Edu local data, single replica
#
# Usage:
#   scripts/slurm/jobs/1B_fw_edu_leonardo.sh
#   NUM_NODES=4 TIME=01:00:00 scripts/slurm/jobs/1B_fw_edu_leonardo.sh
#   scripts/slurm/jobs/1B_fw_edu_leonardo.sh --training.steps 200

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_NODES="${NUM_NODES:-2}"
NUM_GPUS="${NUM_GPUS:-4}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-4}"
TIME="${TIME:-24:00:00}"

CONFIG_FILE=./torchtitan/models/boom/train_configs/fw-edu-local/local_fw_edu_1B_leonardo.toml

CONFIG_FILE="${CONFIG_FILE}" \
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE}" \
  "${SCRIPT_DIR}/../submit.sh" \
    --nodes="${NUM_NODES}" \
    --gres="gpu:${NUM_GPUS}" \
    --time="${TIME}" \
    "${SCRIPT_DIR}/../train.sh" \
    "$@"
