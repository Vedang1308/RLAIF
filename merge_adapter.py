import argparse
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_and_save(args):
    print(f"🚀 Starting Merge Process...")
    print(f"   Base Model: {args.base_model}")
    print(f"   Adapter:    {args.adapter_path}")
    print(f"   Output Dir: {args.output_dir}")

    # 1. Load Tokenizer
    print("📚 Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # 2. Load Base Model
    print("🏗️ Loading Base Model (CPU/RAM efficient)...")
    # Load in FP16 to save memory, map to CPU first
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 2.5 Smart Adapter Path Detection
    # If the user passed a folder like 'trainer_output' but the config is in 'trainer_output/checkpoint-123'
    if os.path.isdir(args.adapter_path):
        if not os.path.exists(os.path.join(args.adapter_path, "adapter_config.json")):
            print(f"⚠️  No adapter_config.json found in {args.adapter_path}. Scanning for checkpoints...")
            subdirs = [os.path.join(args.adapter_path, d) for d in os.listdir(args.adapter_path) if os.path.isdir(os.path.join(args.adapter_path, d))]
            # Sort by name (checkpoint-100, checkpoint-200...) - naive string sort works for fixed prefix if len is same, but better to sort by number
            # Let's try to parse numbers
            checkpoints = []
            for d in subdirs:
                if "checkpoint-" in d:
                    try:
                        step = int(d.split("-")[-1])
                        checkpoints.append((step, d))
                    except:
                        pass
            
            if checkpoints:
                checkpoints.sort(key=lambda x: x[0])
                latest_step, latest_path = checkpoints[-1]
                print(f"✅ Found latest checkpoint: {latest_path} (Step {latest_step})")
                args.adapter_path = latest_path
            else:
                print("❌ No 'checkpoint-N' folders found. Trying original path anyway...")

    # 3. Load Adapter
    print(f"🔌 Loading LoRA Adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    # 4. Merge
    print("🧬 Merging Weights (This effectively 'bakes in' the training)...")
    model = model.merge_and_unload()

    # 5. Save
    print(f"💾 Saving merged model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("✅ Merge Complete!")
    print(f"You can now load this model directly with AutoModelForCausalLM.from_pretrained('{args.output_dir}')")
    
    # Optional: Push to Hub
    if args.push_to_hub:
        repo_id = args.push_to_hub
        print(f"☁️ Uploading to Hugging Face: {repo_id}...")
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        print("🎉 Upload Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="vedang1308/RLAIF-Qwen")
    parser.add_argument("--output_dir", type=str, default="merged_model")
    parser.add_argument("--push_to_hub", type=str, default=None, help="Repo ID to upload to (e.g. vedang1308/RLAIF-Qwen-Merged)")
    args = parser.parse_args()
    
    merge_and_save(args)
