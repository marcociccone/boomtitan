#!/bin/bash

#SBATCH --job-name=eval_boom
#SBATCH -A IscrC_dynaLORA
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#SBATCH --array=0-4
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
MODEL="allenai/OLMo-2-1124-13B"

# --- Job Array Logic ---
# Define the array of step values
# processed ~100B every 12K steps 

# "main"
# "stage1-step12000-tokens101B"
# "stage1-step24000-tokens202B"
# "stage1-step36000-tokens302B"
# "stage1-step48000-tokens403B"
# "stage1-step60000-tokens504B"
# "stage1-step72000-tokens604B"
# "stage1-step84000-tokens705B"
# "stage1-step96000-tokens806B"
# "stage1-step108000-tokens906B"
# "stage1-step120000-tokens1007B"
# "stage1-step132000-tokens1108B"
# "stage1-step144000-tokens1208B"
# "stage1-step156000-tokens1309B"
# "stage1-step168000-tokens1410B"
# "stage1-step180000-tokens1510B"
# "stage1-step192000-tokens1611B"
# "stage1-step204000-tokens1712B"
# "stage1-step216000-tokens1812B"
# "stage1-step228000-tokens1913B"
# "stage1-step240000-tokens2014B"
# "stage1-step252000-tokens2114B"
# "stage1-step264000-tokens2215B"
# "stage1-step276000-tokens2316B"
# "stage1-step288000-tokens2416B"
# "stage1-step300000-tokens2517B"

BRANCHES=(
  "stage1-step358000-tokens3004B"
  "stage1-step417450-tokens3502B"
  "stage1-step476000-tokens3993B"
  "stage1-step537000-tokens4505B"
  "stage1-step596000-tokens5000B"
)

# for i in "${!BRANCHES[@]}"; do
#   REV_ID="${BRANCHES[$i]}"
#   echo "Downloading branch $((i+1))/${#BRANCHES[@]}: $REV_ID"
#   hf download "$MODEL" --revision "$REV_ID" 
# done

# Get the step value for this specific array task
REV_ID="${BRANCHES[$SLURM_ARRAY_TASK_ID]}"

# Get the step value for this specific array task
MODEL_SAFE="${MODEL//\//_}"

echo "Running evaluation for task ${SLURM_ARRAY_TASK_ID} with ${MODEL} and revision ${REV_ID}"

# --- Execution ---
cd /leonardo_work/IscrB_Decentro/mciccone/exp/boomtitan

# --- Environment Setup ---
# hf download $MODEL

export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BATCH_SIZE=8

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="$MODEL",revision="$REV_ID",dtype=bfloat16 \
    --tasks hellaswag,boolq,piqa,triviaqa,winogrande,openbookqa,social_iqa,arc_easy,arc_challenge,commonsense_qa,copa,gsm8k,arc_challenge_chat_cot,arc_multilingual,mmlu_flan_cot_zeroshot,global_mmlu_gen_0shot,drop,cultural_bench,blend,xnli,xcopa \
    --batch_size $BATCH_SIZE \
    --output_path "results/olmo2/${MODEL_SAFE}_${REV_ID}.json" --verbosity DEBUG
echo "Evaluation finished for ${MODEL}"
