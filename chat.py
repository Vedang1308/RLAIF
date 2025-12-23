import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

# Config
MERGED_REPO = "vedang1308/RLAIF-Qwen-Merged"
LOCAL_MERGED = "merged_model"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

def load_model():
    print("⏳ Initializing RLAIF Chat (Terminal Mode)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device.upper()}")

    # 1. Try Local
    if os.path.exists(LOCAL_MERGED) and os.path.exists(os.path.join(LOCAL_MERGED, "config.json")):
        print(f"📂 Loading Local Merged Model: {LOCAL_MERGED}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_MERGED)
            model = AutoModelForCausalLM.from_pretrained(LOCAL_MERGED, torch_dtype=torch.float16, device_map=device)
            return tokenizer, model
        except Exception as e:
            print(f"   ⚠️ Local load failed: {e}")

    # 2. Try Cloud
    print(f"☁️ Downloading Remote Merged Model: {MERGED_REPO}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MERGED_REPO)
        model = AutoModelForCausalLM.from_pretrained(MERGED_REPO, torch_dtype=torch.float16, device_map=device)
        return tokenizer, model
    except Exception as e:
        print(f"❌ Critical Error loading model: {e}")
        sys.exit(1)

def main():
    tokenizer, model = load_model()
    
    print("\n" + "="*50)
    print("🤖 RLAIF CLI CHAT (Safe Mode)")
    print("   Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\n👤 You: ")
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Bye!")
                break
            
            if not user_input.strip(): continue

            # Generate
            inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
            
            print("🤖 Model: ", end="", flush=True)
            
            # Streaming-like output using standard generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Strip prompt if model echoes it
            if response.startswith(user_input):
                response = response[len(user_input):].strip()
            
            print(response)

        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error during generation: {e}")

if __name__ == "__main__":
    main()
