#!/bin/bash

#SBATCH --job-name=eval_boom
#SBATCH -A IscrC_dynaLORA
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#SBATCH --array=0-1
#SBATCH --output=logs/eval_boom_%A_%a.out
#SBATCH --error=logs/eval_boom_%A_%a.err

set -e

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
module load profile/deeplrn

# Loading eval env
source /leonardo_scratch/fast/IscrB_Decentro/mciccone/envs/eval/bin/activate

# --- Job Array Logic ---
# Define the array of step values
MODELS=()
# MODELS+=("HuggingFaceTB/SmolLM3-3B-Base")
MODELS+=("google/gemma-3-12b-pt")
MODELS+=("google/gemma-3-12b-it")
# MODELS+=("Qwen/Qwen3-14B-Base")
# MODELS+=("Qwen/Qwen2.5-14B-Instruct")
# MODELS+=("meta-llama/Llama-3.1-8B-Instruct")
# MODELS+=("meta-llama/Llama-3.1-8B")
# MODELS+=("meta-llama/Llama-2-13b-hf")
# MODELS+=("HuggingFaceTB/SmolLM3-3B")
# MODELS+=("swiss-ai/Apertus-8B-2509")
# MODELS+=("swiss-ai/Apertus-8B-Instruct-2509")
# MODELS+=("Qwen/Qwen2.5-14B")
# MODELS+=("Qwen/Qwen3-14B")

# Get the step value for this specific array task
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
MODEL_SAFE="${MODEL//\//_}"

echo "Running evaluation for task ${SLURM_ARRAY_TASK_ID} with ${MODEL}"

# --- Execution ---
cd /leonardo_work/IscrB_Decentro/mciccone/exp/boomtitan

# --- Environment Setup ---
# hf download $MODEL

export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BATCH_SIZE=4

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="$MODEL",dtype=bfloat16 \
    --gen_kwargs max_gen_toks=256 \
    --tasks hellaswag,boolq,piqa,triviaqa,winogrande,openbookqa,social_iqa,arc_easy,arc_challenge,commonsense_qa,copa,gsm8k,arc_challenge_chat_cot,arc_multilingual,mmlu_flan_cot_zeroshot,global_mmlu_gen_0shot,drop,cultural_bench,blend,xnli,xcopa \
    --batch_size $BATCH_SIZE \
    --output_path "results/baselines/$MODEL_SAFE.json" --verbosity DEBUG
echo "Evaluation finished for ${MODEL}"
