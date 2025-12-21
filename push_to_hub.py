import os
import argparse
from huggingface_hub import HfApi, create_repo

def push_to_hub():
    parser = argparse.ArgumentParser(description="Upload trained RLAIF model to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="Target Repo ID (e.g., 'username/rlaif-qwen-0.5b')")
    parser.add_argument("--token", type=str, help="Hugging Face Write Token (optional if logged in via CLI)")
    parser.add_argument("--model_dir", type=str, default="trainer_output", help="Directory containing the model artifacts")
    parser.add_argument("--private", action="store_true", help="Make the repository private")
    
    args = parser.parse_args()
    
    print(f"🚀 Preparing to upload '{args.model_dir}' to '{args.repo_id}'...")
    
    if not os.path.exists(args.model_dir):
        print(f"❌ Error: Model directory '{args.model_dir}' not found!")
        return

    api = HfApi(token=args.token)
    
    # 1. Create Repo (if it doesn't exist)
    try:
        url = create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True, token=args.token)
        print(f"✅ Repository ready: {url}")
    except Exception as e:
        print(f"⚠️  Repo creation check failed (might already exist or auth error): {e}")

    # 2. Upload Folder
    print("⏳ Uploading files... (This might take a while for large models)")
    try:
        api.upload_folder(
            folder_path=args.model_dir,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Upload trained RLAIF model adapter",
            ignore_patterns=["checkpoint-*", "*.tmp"] # Skip intermediate checkpoints to save space/time
        )
        print("🎉 Success! Model uploaded safely.")
        print(f"🔗 View it here: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    push_to_hub()
