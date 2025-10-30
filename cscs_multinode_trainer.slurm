#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# --- This script is optimized for AWS with EFA
# --- adjust NCCL_BUFFSIZE if you encounter memory
# --- constraint issues or to tune for improved performance.
# --- LOCAL_BATCH_SIZE=4 GRAD_ACC_STEPS=2 NUM_GPUS=4 NUM_NODES=18 ./new_multinode_trainer.slurm

NUM_NODES=${NUM_NODES:-2}
NUM_GPUS=${NUM_GPUS:-4}
REPLICATE_DEGREE=${REPLICATE_DEGREE:-1}

cat << EOF | sbatch
#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --job-name=boom_3B
#SBATCH --ntasks=${NUM_NODES}
#SBATCH --nodes=${NUM_NODES}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --cpus-per-task=72
#SBATCH --environment=/iopsstor/scratch/cscs/mciccone/cuda128/cuda1281.toml
#SBATCH --signal=SIGUSR2@600	# Send SIGUSR2 600 seconds before hitting the time limit
#SBATCH --output=logs/%x_%j/out/%N_%j_%t.out
#SBATCH --error=logs/%x_%j/err/%N_%j_%t.err
#SBATCH --export=ALL

#SBATCH --open-mode=append
#SBATCH --signal=USR2@120
#SBATCH --distribution=block

# Make output folder
mkdir -p logs/\${SLURM_JOB_NAME}_\${SLURM_JOB_ID}
mkdir -p logs/\${SLURM_JOB_NAME}_\${SLURM_JOB_ID}/out
mkdir -p logs/\${SLURM_JOB_NAME}_\${SLURM_JOB_ID}/err

DUMP_FOLDER="./boom_outputs/slurm_jobs/\${SLURM_JOB_NAME}_\${SLURM_JOB_ID}/"
# DUMP_FOLDER="./boom_outputs/slurm_jobs/torchtitan_multi_node_25676635"
# DUMP_FOLDER="./boom_outputs/slurm_jobs/lr_ablation_25879997/"
# DUMP_FOLDER="./boom_outputs/slurm_jobs/boom_warmup_13B/"
# DUMP_FOLDER="./boom_outputs/slurm_jobs/boom_warmup_13B_cooldown"

# Activate python environment
echo "Activate python env"
source /opt/titan/bin/activate
cd /iopsstor/scratch/cscs/mciccone/exp/boomtitan

# Get node information
nodes=( \$( scontrol show hostnames \$SLURM_JOB_NODELIST ) )
nodes_array=(\$nodes)
head_node=\${nodes_array[0]}
head_node_ip=\$(srun --nodes=1 --ntasks=1 -w "\$head_node" hostname --ip-address)

echo "Head node: \$head_node"
echo "Head node IP: \$head_node_ip"

# EFA settings
export NCCL_P2P_LEVEL=SYS
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# # CUDA settings
export CUDA_LAUNCH_BLOCKING=0
export PYTHONFAULTHANDLER=1
export TORCH_DIST_INIT_BARRIER=1
# export NCCL_BUFFSIZE=268435456 # THIS IS MAKING EVERYTHING SLOWER ON CSCS!!

# CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/boom/train_configs/llama3_8b.toml"}
# CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/boom/train_configs/llama3_base/debug/llama3_smollm13B.toml"}
# CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/boom/train_configs/fw-edu-local/local_fw_edu_smollm3B.toml"}
# CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/boom/train_configs/sl3-datamix/smollm3B_test_elie.toml"}
# CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/boom/train_configs/sl3-datamix/smollm13B.toml"}
# CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/boom/train_configs/sl3-datamix/smollm13B_cooldown.toml"}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/boom/train_configs/sl3-datamix/smollm3B.toml"}

# WANDB
export WANDB_PROJECT="boomtitan"
export WANDB_ENTITY="mciccone"
export WANDB_ENTITY="mciccone"
# export WANDB_MODE="offline"

# HUGGINGFACE
export HF_HOME=$SCRATCH/huggingface/

LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-2}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
NUM_NODES=${NUM_NODES}
NUM_GPUS=${NUM_GPUS}
REPLICATE_DEGREE=${REPLICATE_DEGREE}
GLOBAL_BATCH_SIZE=\$((LOCAL_BATCH_SIZE * NUM_GPUS * NUM_NODES * GRAD_ACC_STEPS))

echo "Configuration: \${NUM_NODES} nodes x \${NUM_GPUS} GPUs x \${LOCAL_BATCH_SIZE} local_batch_size x \${GRAD_ACC_STEPS} grad_acc_steps = \${GLOBAL_BATCH_SIZE} global_batch_size"
echo \${REPLICATE_DEGREE} replicas

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH="/iopsstor/scratch/cscs/mciccone/exp/boomtitan:$PYTHONPATH"
srun --output="logs/%x_%j/out/%N_%j_%t.out" --error="logs/%x_%j/err/%N_%j_%t.err" \
torchrun --nnodes \${NUM_NODES} --nproc_per_node \${NUM_GPUS} --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "\$head_node_ip:29500" --local-ranks-filter 0 --role rank --tee 3 ./torchtitan/train.py \
--job.config_file \${CONFIG_FILE} \
--training.local_batch_size \${LOCAL_BATCH_SIZE} \
--training.global_batch_size \${GLOBAL_BATCH_SIZE} \
--parallelism.data_parallel_replicate_degree \${REPLICATE_DEGREE} \
--job.dump_folder \${DUMP_FOLDER} \
\$@
EOF
