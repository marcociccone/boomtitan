#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --job-name=download_boom
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --environment=/iopsstor/scratch/cscs/mciccone/cuda128/cuda1281.toml
#SBATCH --signal=SIGUSR2@600    # Send SIGUSR2 600 seconds before hitting the time limit
#SBATCH --output=logs/%x_%j/out/%N_%j_%t.out
#SBATCH --error=logs/%x_%j/err/%N_%j_%t.err
#SBATCH --export=ALL
#SBATCH --array=0-40

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
# cd /iopsstor/scratch/cscs/mciccone/exp/boomtitan

export AWS_ACCESS_KEY_ID=''
export AWS_SECRET_ACCESS_KEY=''


SCRIPTS=(
  "download_dclm.sh"
  "download_finemath.sh"
  "download_fineweb-edu.sh"
  "download_fw2-cmn.sh"
  "download_fw2-deu.sh"
  "download_fw2-ell.sh"
  "download_fw2-fas.sh"
  "download_fw2-fra.sh"
  "download_fw2-hin.sh"
  "download_fw2-ita.sh"
  "download_fw2-jpn.sh"
  "download_fw2-kor.sh"
  "download_fw2-por.sh"
  "download_fw2-rus.sh"
  "download_fw2-spa.sh"
  "download_fw2-tha.sh"
  "download_fw2-vie.sh"
  "download_github-issues.sh"
  "download_infiwebmath.sh"
  "download_jupyter-scripts.sh"
  "download_kaggle.sh"
  "download_pes2o.sh"
  "download_pull-requests.sh"
  "download_stack-edu-Cpp.sh"
  "download_stack-edu-C.sh"
  "download_stack-edu-C-Sharp.sh"
  "download_stack-edu-Go.sh"
  "download_stack-edu-HTML.sh"
  "download_stack-edu-JavaScript.sh"
  "download_stack-edu-Java.sh"
  "download_stack-edu-Markdown.sh"
  "download_stack-edu-PHP.sh"
  "download_stack-edu-Python.sh"
  "download_stack-edu-Ruby.sh"
  "download_stack-edu-Rust.sh"
  "download_stack-edu-Shell.sh"
  "download_stack-edu-SQL.sh"
  "download_stack-edu-Swift.sh"
  "download_stack-edu-TypeScript.sh"
  "download_stackexchange.sh"
  "download_wiki.sh"
)

SCRIPT_NAME="${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"

cd /capstor/store/cscs/swissai/infra01/boom/
ln -s /capstor/store/cscs/swissai/infra01/boom/s5cmd /usr/bin/s5cmd

# source $SCRIPT_NAME
source download_all.sh

