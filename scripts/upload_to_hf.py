import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login

def upload_model(repo_id, model_path, token=None):
    """
    Uploads the model weights to Hugging Face Hub.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    api = HfApi(token=token)
    
    print(f"Preparing to upload {model_path.name} to {repo_id}...")
    
    try:
        # Create repo if it doesn't exist (private by default for safety)
        api.create_repo(repo_id=repo_id, exist_ok=True, private=True)
        print(f"Repository {repo_id} ready.")
        
        # Upload
        print(f"Uploading file...")
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=model_path.name,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"Upload complete: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTip: Make sure you are logged in using `huggingface-cli login` or provide a token.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model weights to Hugging Face Hub")
    parser.add_argument("--repo", type=str, required=True, help="Repository ID (e.g., username/geoguessr-model)")
    parser.add_argument("--file", type=str, default="checkpoints/best_finetune.pt", help="Path to model file")
    parser.add_argument("--token", type=str, help="Hugging Face API token (optional if logged in via CLI)")
    
    args = parser.parse_args()
    
    upload_model(args.repo, args.file, args.token)
