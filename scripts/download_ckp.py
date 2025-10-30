import os
import argparse
from huggingface_hub import snapshot_download

def download_checkpoints(repo_id, steps, base_path, token=None):
    """
    Downloads checkpoints from a Hugging Face repository for specified steps.

    Args:
        repo_id (str): The ID of the repository (e.g., 'the-boom/boom-warmup-13B-dist-ckp').
        steps (list[int]): A list of steps to download.
        token (str, optional): Your Hugging Face Hub token for private repos. Defaults to None.
    """
    for step in steps:
        revision_name = f'step-{step}'
        print(f"Attempting to download checkpoint for {revision_name}...")
        try:
            # The token is now passed directly to the download function
            snapshot_download(
                repo_id=repo_id,
                revision=revision_name,
                local_dir=f'{base_path}/{revision_name}',
                local_dir_use_symlinks=False, # Set to True if your system supports it and you want to save space
                token=token
            )
            print(f"✅ Successfully downloaded {revision_name}")
        except Exception as e:
            print(f"❌ Could not download checkpoint for {revision_name}. It might not exist. Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download model checkpoints from Hugging Face.")
    parser.add_argument('--repo_id', type=str, default='the-boom/boom-warmup-13B-dist-ckp', help='The repository ID on Hugging Face.')
    parser.add_argument('--steps', type=int, nargs='+', required=True, help='A list of steps (as integers) to download, e.g., 8000 16000 24000.')
    parser.add_argument('--token', type=str, help='Your Hugging Face Hub token (optional).')
    parser.add_argument('--base_path', type=str, help='Download path.')

    args = parser.parse_args()

    download_checkpoints(args.repo_id, args.steps, args.base_path, args.token)
