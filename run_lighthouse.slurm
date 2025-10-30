#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --job-name=torchft_lighthouse
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --environment=/iopsstor/scratch/cscs/mciccone/cuda128/cuda1281.toml
#SBATCH --signal=SIGUSR2@600    # Send SIGUSR2 600 seconds before hitting the time limit
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

# Activate python environment
echo "Activate python env"
source /opt/titan/bin/activate
cd /iopsstor/scratch/cscs/mciccone/exp/boomtitan

RUST_BACKTRACE=1 torchft_lighthouse --min_replicas 2 --quorum_tick_ms 100 --join_timeout_ms 10000

