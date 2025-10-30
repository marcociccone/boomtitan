#!/bin/bash

#SBATCH --job-name=eval_boom
#SBATCH -A IscrC_dynaLORA
#SBATCH --partition=boost_usr_prod
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#SBATCH --array=0-7
#SBATCH --output=logs/eval_boom_%A_%a.out
#SBATCH --error=logs/eval_boom_%A_%a.err

# ------------------------------------------------
#
# !! README!!
#
# These are the important dependencies
# For boom model in transformers 
# -- https://github.com/marcociccone/transformers
#
# For the correct version of lm-eval-harness
# -- https://github.com/swiss-ai/lm-evaluation-harness
# -- be sure to use hf datasets 3.6.0
#
# ------------------------------------------------


set -e

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
module load profile/deeplrn

# Load torchtitan env
source /leonardo_scratch/fast/IscrC_dynaLORA/mciccone/envs/titan/bin/activate

# --- Job Array Logic ---
# Define the array of step values
STEPS=()
for i in {8000..64000..8000}; do
  STEPS+=("$i")
done

# Get the step value for this specific array task
STEP="${STEPS[$SLURM_ARRAY_TASK_ID]}"

# --- Dynamic Path Configuration ---
export MODEL_CKP="step-$STEP"
BASE_PATH_CKP="/leonardo_work/IscrB_Decentro/boom_warmup_13B_ckp"
export MODEL_NAME_TT="$BASE_PATH_CKP/$MODEL_CKP/"
export BASE_PATH_HF="/leonardo_work/IscrB_Decentro/boom_warmup_13B_ckp/hf_permute"
export MODEL_NAME_HF="$BASE_PATH_HF/$MODEL_CKP/"
export CONFIG="torchtitan/models/boom/train_configs/sl3-datamix/smollm13B.toml"

echo "Running evaluation for task ${SLURM_ARRAY_TASK_ID} with checkpoint ${MODEL_CKP}"

# --- Execution ---
cd /leonardo_work/IscrB_Decentro/mciccone/exp/boomtitan

# echo "Downloading checkpoint $MODEL_CKP"
# python scripts/download_ckp.py --steps $STEP --base_path $BASE_PATH_CKP 

# --- Environment Setup ---
export TOK="meta-llama/Llama-3.2-1B"
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Convert to HF checkpoint
echo "Converting checkpoint to Hugging Face format... from $MODEL_NAME_TT to $MODEL_NAME_HF"
python scripts/checkpoint_conversion/convert_to_hf.py "$MODEL_NAME_TT" "$MODEL_NAME_HF" --config="$CONFIG"

# Remove pre-consolidated tmp ckp
echo "Cleaning up temporary shard files..."
rm -rf "$MODEL_NAME_HF/sharded"

# Copy model definition config
echo "Copying config to $MODEL_NAME_HF"
cp /leonardo_work/IscrB_Decentro/boom_warmup_13B_ckp/hf_permute/step-184000/config.json $MODEL_NAME_HF

# Run the evaluation
echo "Starting lm-eval..."

# Loading eval env
source /leonardo_scratch/fast/IscrB_Decentro/mciccone/envs/eval/bin/activate

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="$MODEL_NAME_HF",tokenizer="$TOK",dtype=bfloat16 \
    --tasks hellaswag,boolq,piqa,triviaqa,winogrande,openbookqa,social_iqa,arc_easy,arc_challenge,commonsense_qa,copa,gsm8k,arc_challenge_chat_cot,arc_multilingual,mmlu_flan_cot_zeroshot,global_mmlu_gen_0shot,drop,cultural_bench,blend,xnli,xcopa \
    --batch_size 16 \
    --output_path "results/boom/$MODEL_CKP.json" --verbosity DEBUG
echo "Evaluation finished for ${MODEL_CKP}"
