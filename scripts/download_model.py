import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_weights(repo_id, filename="best_finetune.pt", output_dir="checkpoints"):
    """
    Downloads model weights from Hugging Face Hub.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"⬇️  Downloading {filename} from {repo_id}...")
    
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"✅ Successfully downloaded to: {local_path}")
        
    except Exception as e:
        print(f"❌ Error downloading file: {e}")
        print(f"   Please check if the repository {repo_id} exists and is public (or you have access).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model weights from Hugging Face Hub")
    parser.add_argument("--repo", type=str, default="chetangoel01/GeoguessrModel", help="Hugging Face Repository ID")
    parser.add_argument("--file", type=str, default="best_finetune.pt", help="Filename to download")
    parser.add_argument("--out", type=str, default="checkpoints", help="Output directory")
    
    args = parser.parse_args()
    
    download_weights(args.repo, args.file, args.out)
