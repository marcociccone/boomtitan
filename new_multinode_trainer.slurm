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
#SBATCH --account=ehpc277
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=72:00:00
#SBATCH --job-name=warmup_boom_13B
#SBATCH --ntasks=${NUM_NODES}
#SBATCH --nodes=${NUM_NODES}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --cpus-per-task=80
#SBATCH --output=logs/%x_%j/out/%N_%j_%t.out
#SBATCH --error=logs/%x_%j/err/%N_%j_%t.err
#SBATCH --export=ALL
#SBATCH --exclude=as04r4b10,as04r4b22,as05r4b01,as05r1b06,as04r1b12,as03r2b29,as07r5b19,as06r4b20,as05r3b19

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
DUMP_FOLDER="./boom_outputs/slurm_jobs/boom_warmup_13B_cooldown"

module load mkl
module load intel
module load python/3.12.1
unset PYTHONPATH

echo "Activating environment: \$WORK/mciccone/envs/titan/bin/activate"
source \$WORK/mciccone/envs/titan/bin/activate

# Get node information
nodes=( \$( scontrol show hostnames \$SLURM_JOB_NODELIST ) )
nodes_array=(\$nodes)
head_node=\${nodes_array[0]}
head_node_ip=\$(srun --nodes=1 --ntasks=1 -w "\$head_node" hostname --ip-address)

echo "Head node: \$head_node"
echo "Head node IP: \$head_node_ip"

# InfiniBand interface detection - let NCCL auto-detect first
export NCCL_IB_DISABLE=0  # Ensure InfiniBand is enabled
export NCCL_NET_GDR_LEVEL=2  # Enable GPUDirect RDMA if supported
export NCCL_SOCKET_IFNAME="ib0,ib1,ib2,ib3"

# NDR200 optimizations
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5  # Mellanox ConnectX adapters (common on NDR systems)
export NCCL_IB_TC=160    # Traffic class for NDR (high priority)
export NCCL_IB_SL=0      # Service level
export NCCL_IB_TIMEOUT=22 # Timeout for IB operations

export NCCL_P2P_LEVEL=SYS
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

# CUDA settings
export CUDA_LAUNCH_BLOCKING=0
export PYTHONFAULTHANDLER=1
export TORCH_DIST_INIT_BARRIER=1
export NCCL_BUFFSIZE=268435456

# CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/boom/train_configs/llama3_8b.toml"}
# CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/boom/train_configs/llama3_base/debug/llama3_smollm13B.toml"}
# CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/boom/train_configs/fw-edu-local/local_fw_edu_smollm3B.toml"}
# CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/boom/train_configs/sl3-datamix/smollm3B.toml"}
# CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/boom/train_configs/sl3-datamix/smollm3B_test_elie.toml"}
# CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/boom/train_configs/sl3-datamix/smollm13B.toml"}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/boom/train_configs/sl3-datamix/smollm13B_cooldown.toml"}

# WANDB
export WANDB_PROJECT="boomtitan"
export WANDB_ENTITY="mciccone"
export WANDB_MODE="offline"

LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-2}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
NUM_NODES=${NUM_NODES}
NUM_GPUS=${NUM_GPUS}
REPLICATE_DEGREE=${REPLICATE_DEGREE}
GLOBAL_BATCH_SIZE=\$((LOCAL_BATCH_SIZE * NUM_GPUS * NUM_NODES * GRAD_ACC_STEPS))

echo "Configuration: \${NUM_NODES} nodes x \${NUM_GPUS} GPUs x \${LOCAL_BATCH_SIZE} local_batch_size x \${GRAD_ACC_STEPS} grad_acc_steps = \${GLOBAL_BATCH_SIZE} global_batch_size"
echo \${REPLICATE_DEGREE} replicas

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun --output="logs/%x_%j/out/%N_%j_%t.out" --error="logs/%x_%j/err/%N_%j_%t.err" \
torchrun --nnodes \${NUM_NODES} --nproc_per_node \${NUM_GPUS} --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "\$head_node_ip:29500" --local-ranks-filter 0 --role rank --tee 3 ./torchtitan/train.py \
--job.config_file \${CONFIG_FILE} \
--training.local_batch_size \${LOCAL_BATCH_SIZE} \
--training.global_batch_size \${GLOBAL_BATCH_SIZE} \
--parallelism.data_parallel_replicate_degree \${REPLICATE_DEGREE} \
--job.dump_folder \${DUMP_FOLDER} \
\$@
EOF
