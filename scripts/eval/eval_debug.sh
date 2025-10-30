export MODEL_CKP="step-88000"
# export MODEL_NAME="/leonardo_work/IscrB_Decentro/boom_warmup_13B_cooldown_ckp/hf_permute/$MODEL_CKP/"
export MODEL_NAME="/leonardo_work/IscrB_Decentro/boom_warmup_13B_ckp/hf_permute/$MODEL_CKP/"
export TOK="meta-llama/Llama-3.2-1B"
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# accelerate launch -m 

lm_eval --model hf \
    --model_args pretrained="$MODEL_NAME",tokenizer="$TOK",dtype=bfloat16 \
    --tasks hellaswag,boolq,piqa,triviaqa,winogrande,openbookqa,social_iqa,arc_easy,arc_challenge,commonsense_qa,copa,gsm8k,arc_challenge_chat_cot,arc_multilingual,mmlu_flan_cot_zeroshot,global_mmlu_gen_0shot,drop,cultural_bench,blend,xnli,xcopa \
    --batch_size 16 \
    --output_path "results/$MODEL_CKP.json" --verbosity DEBUG



    # --gen_kwargs temperature=1.0,top_p=0.95,max_gen_toks=8192 \


# accelerate launch -m lm_eval --model hf \
#     --model_args pretrained=Qwen/Qwen3-14B,dtype=bfloat16 \
#     --tasks hellaswag,boolq,piqa,triviaqa,winogrande,openbookqa,arc_easy,arc_challenge,commonsense_qa,copa,gsm8k,mmlu,mmlu_pro \
#     --batch_size 16 \
#     --trust_remote_code \
#     --output_path "results/Qwen3-14B.json"


# lm_eval --model vllm \
#     --model_args pretrained="$MODEL_NAME",tokenizer="$TOK",tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=2 \
#     --tasks mmlu_flan_cot_zeroshot \
#     --batch_size auto \
#     --trust_remote_code \
#     --output_path "results/$MODEL_CKP.json" --verbosity DEBUG
#     # --gen_kwargs temperature=1.0,top_p=0.95,max_gen_toks=8192 \

#
# lm_eval --model hf --model_args meta-llama/Llama-2-13b,dtype=bfloat16  --tasks hellaswag,boolq,piqa,triviaqa,winogrande,openbookqa,social_iqa,arc_easy,arc_challenge,commonsense_qa,copa,gsm8k,mmlu,mmlu_pro \
