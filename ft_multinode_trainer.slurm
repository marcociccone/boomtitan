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
#SBATCH --time=1:00:00
#SBATCH --job-name=ft_boom_13B
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
# export WANDB_MODE="offline"

# HUGGINGFACE
export HF_HOME=$SCRATCH/huggingface/

LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-2}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
NUM_NODES=${NUM_NODES}
NUM_GPUS=${NUM_GPUS}
REPLICATE_DEGREE=${REPLICATE_DEGREE}

# ----
# semi-sync FT parameters
GROUP_SIZE=${GROUP_SIZE}
REPLICA_ID=${REPLICA_ID}
SYNC_STEPS=${SYNC_STEPS:-"30"}
SYNC_METHOD=${SYNC_METHOD:-"local_sgd"}
# ----

export TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/iopsstor/scratch/cscs/mciccone/exp/boomtitan:$PYTHONPATH"

ISLAND_BATCH_SIZE=\$((LOCAL_BATCH_SIZE * NUM_GPUS * NUM_NODES * GRAD_ACC_STEPS))
GLOBAL_BATCH_SIZE=\$((ISLAND_BATCH_SIZE * GROUP_SIZE))
DP_PARALLEL_SHARD_DEGREE=\$((NUM_GPUS * NUM_NODES))

echo "Configuration: \${NUM_NODES} nodes x \${NUM_GPUS} GPUs x \${LOCAL_BATCH_SIZE} local_batch_size x \${GRAD_ACC_STEPS} grad_acc_steps = \${ISLAND_BATCH_SIZE} island_batch_size"
echo "Configuration: \${ISLAND_BATCH_SIZE} island_batch_size x \${GROUP_SIZE} num island replicas = \${GLOBAL_BATCH_SIZE} global_batch_size"
echo "FT config: \${GROUP_SIZE} num island replicas"
echo "FT config: \${REPLICA_ID} island id"

export TORCHFT_QUORUM_TIMEOUT_SEC=900
export TORCHFT_TIMEOUT_SEC=600
export TORCHFT_WATCHDOG_TIMEOUT_SEC=300

# # Enable verbose logging
# export TORCH_DISTRIBUTED_DEBUG=DETAIL  # or INFO
# 
# # GLOO-specific verbose output
# export GLOO_LOG_LEVEL=TRACE
# 
# # General PyTorch logging
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_SHOW_CPP_STACKTRACES=1
# 
# export TORCH_NCCL_DUMP_ON_TIMEOUT=true
# export TORCH_NCCL_ENABLE_TIMING=true
# export TORCH_NCCL_ENABLE_TIMING=true
# 
# export TORCHFT_TRIGGER_FR_ON_ABORT=true
# export TORCH_FR_DUMP_DYNAMIC_FILE_NAME=1
# export CUDA_ENABLE_CORE_DUMP_ON_EXCEPTION=1
# 
# # EFA settings
# export NCCL_P2P_LEVEL=SYS
# export NCCL_DEBUG=WARN
# export NCCL_DEBUG_SUBSYS=ALL


# When TorchFT is enabled, HSDP will be used.
# --parallelism.data_parallel_replicate_degree should be 1
# --fault_tolerance.group_size will be used to control the maximum replicate group size as the replicate group size is dynamic.

if [[ "\$SYNC_METHOD" == "hsdp" ]]; then
    srun --output="logs/%x_%j/out/%N_%j_%t.out" --error="logs/%x_%j/err/%N_%j_%t.err" \
    torchrun --nnodes \${NUM_NODES} --nproc_per_node \${NUM_GPUS} --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "\$head_node_ip:29500" --role rank --tee 3 ./torchtitan/train.py \
    --job.config_file \${CONFIG_FILE} \
    --training.local_batch_size \${LOCAL_BATCH_SIZE} \
    --training.global_batch_size \${GLOBAL_BATCH_SIZE} \
    --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.data_parallel_shard_degree \${DP_PARALLEL_SHARD_DEGREE} \
    --fault_tolerance.enable \
    --fault_tolerance.group_size \${GROUP_SIZE} \
    --fault_tolerance.replica_id \${REPLICA_ID} \
    --fault_tolerance.process_group_timeout_ms=300000 \
    --comm.train_timeout_seconds=300 \
    --job.dump_folder \${DUMP_FOLDER} \
    \$@
elif [[ "\$SYNC_METHOD" == "local_sgd" || "\$SYNC_METHOD" == "diloco" ]]; then

    echo "FT config: \${SYNC_STEPS} SYNC_STEPS"
    echo "FT config: \${SYNC_METHOD} SYNC_METHOD"

    srun --output="logs/%x_%j/out/%N_%j_%t.out" --error="logs/%x_%j/err/%N_%j_%t.err" \
    torchrun --nnodes \${NUM_NODES} --nproc_per_node \${NUM_GPUS} --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "\$head_node_ip:29500" --role rank --tee 3 ./torchtitan/train.py \
    --job.config_file \${CONFIG_FILE} \
    --training.local_batch_size \${LOCAL_BATCH_SIZE} \
    --training.global_batch_size \${GLOBAL_BATCH_SIZE} \
    --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.data_parallel_shard_degree \${DP_PARALLEL_SHARD_DEGREE} \
    --fault_tolerance.enable \
    --fault_tolerance.group_size \${GROUP_SIZE} \
    --fault_tolerance.replica_id \${REPLICA_ID} \
    --fault_tolerance.sync_steps \${SYNC_STEPS} \
    --fault_tolerance.semi_sync_method \${SYNC_METHOD} \
    --fault_tolerance.process_group_timeout_ms=300000 \
    --comm.train_timeout_seconds=300 \
    --job.dump_folder \${DUMP_FOLDER} \
    \$@
else
    echo "Unsupported SYNC_METHOD: \${SYNC_METHOD}"
    exit 1
fi
EOF
