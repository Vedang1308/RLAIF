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

    # 3. Load Adapter
    print("🔌 Loading LoRA Adapter...")
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
