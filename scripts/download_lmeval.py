#!/usr/bin/env python3
"""
Script to download all required datasets for lm-evaluation-harness tasks.
Run this on the login node with internet access.
"""

import os
from datasets import load_dataset

# Set to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configure cache directory (change this to your shared filesystem path)
# CACHE_DIR = os.environ.get("HF_DATASETS_CACHE", "/shared/path/hf_cache/datasets")
# os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

# print(f"Downloading datasets to: {CACHE_DIR}\n")

# Dataset configurations based on lm-evaluation-harness task definitions
DATASETS = [
    # Task name, HF dataset path, config name, trust_remote_code
    ("hellaswag", "Rowan/hellaswag", None, False),
    ("boolq", "google/boolq", None, True),
    ("nq_open", "nq_open", None, False),
    ("piqa", "baber/piqa", None, False),
    ("social_iqa", "allenai/social_i_qa", None, True),
    ("triviaqa", "trivia_qa", "rc.nocontext", False),
    ("winogrande", "allenai/winogrande", "winogrande_xl", False),
    ("openbookqa", "allenai/openbookqa", "main", False),
    ("arc_easy", "allenai/ai2_arc", "ARC-Easy", False),
    ("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", False),
    ("race", "ehovy/race", "all", False),
    ("commonsense_qa", "tau/commonsense_qa", None, False),
    ("coqa", "stanfordnlp/coqa", None, False),
    ("copa", "super_glue", "copa", False),
    ("gsm8k", "openai/gsm8k", "main", False),
    ("bbh", "lukaemon/bbh", ['boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa', 'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting'], False),
    ("mmlu", "cais/mmlu", "all", False),
    ("mmlu_pro", "TIGER-Lab/MMLU-Pro", None, False),
]


def download_dataset(task_name, dataset_path, config_name, trust_remote_code):
    """Download a single dataset with error handling."""
    try:
        print(f"Downloading {task_name}...")
        print(f"  Dataset: {dataset_path}")
        if config_name:
            print(f"  Config: {config_name}")
        
        if config_name:
            load_dataset(
                dataset_path, 
                config_name, 
                trust_remote_code=trust_remote_code,
                # cache_dir=CACHE_DIR
            )
        else:
            load_dataset(
                dataset_path, 
                trust_remote_code=trust_remote_code,
                # cache_dir=CACHE_DIR
            )
        
        print(f"✓ {task_name} downloaded successfully\n")
        return True
    
    except Exception as e:
        print(f"✗ Error downloading {task_name}: {str(e)}\n")
        return False

def main():
    print("="*60)
    print("LM Evaluation Harness Dataset Downloader")
    print("="*60 + "\n")
    
    success_count = 0
    failed_tasks = []
    
    for task_name, dataset_path, config_name, trust_remote_code in DATASETS:
        if isinstance(config_name, (list, tuple)):
            for cfg in config_name:
                success = download_dataset(task_name, dataset_path, cfg, trust_remote_code)
                if success:
                    success_count += 1
                else:
                    failed_tasks.append(task_name)
        else:
            success = download_dataset(task_name, dataset_path, config_name, trust_remote_code)
            if success:
                success_count += 1
            else:
                failed_tasks.append(task_name)
    # Summary
    print("="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Total tasks: {len(DATASETS)}")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed: {len(failed_tasks)}")
    
    if failed_tasks:
        print(f"\nFailed tasks: {', '.join(failed_tasks)}")
    else:
        print("\n✓ All datasets downloaded successfully!")
    
    print(f"\nDatasets cached in: {CACHE_DIR}")
    print("\nTo use on compute nodes, set:")
    print(f"  export HF_DATASETS_CACHE={CACHE_DIR}")
    print(f"  export HF_DATASETS_OFFLINE=1")

if __name__ == "__main__":
    main()
