import os
import glob
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from trl.experimental.ppo import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
except ImportError:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig
from rewards import verify_reward_func, ai_feedback_reward_func
import sys
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
from tqdm import tqdm

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "checkpoints"
LOG_DIR = "logs"
LEARNING_RATE = 1.41e-5
BATCH_SIZE = 1 # Small batch for demo/small-gpu
MINI_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
TOTAL_STEPS = 100
SAVE_FREQ = 10 

def get_latest_checkpoint(output_dir):
    """Finds the latest checkpoint directory."""
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    # Sort by step number (assuming format checkpoint-N)
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1]

def main():
    # 1. Dataset (GSM8K for math reasoning)
    # Using 'main' split for training
    dataset = load_dataset("gsm8k", "main", split="train[:100]") # Small subset for demo

    def build_dataset(tokenizer, ds):
        """Prepares dataset for PPO."""
        input_min_text_length = 2
        input_max_text_length = 8
        
        def tokenize(sample):
            # Qwen instruct format: try to just prompt with the question
            prompt = f"Question: {sample['question']}\nAnswer:"
            sample["input_ids"] = tokenizer.encode(prompt)
            sample["query"] = tokenizer.decode(sample["input_ids"])
            return sample

        ds = ds.map(tokenize, batched=False)
        ds = ds.filter(lambda x: len(x["input_ids"]) > input_min_text_length)
        ds.set_format(type="torch")
        return ds

    # 2. Model & Tokenizer
    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token # Qwen often needs this

    # Load with LoRA to save memory
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 3. PPO Config
    config = PPOConfig(
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        mini_batch_size=MINI_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )

    # 4. Checkpoint Resumption Strategy
    latest_ckpt = get_latest_checkpoint(OUTPUT_DIR)
    
    # Load model
    # Note: TRL's AutoModelForCausalLMWithValueHead wraps the base model
    # If resuming, we ideally want to load the adapters. 
    # For simplicity in this script, we load base + lora config, then later we would load the specific adapter state if TRL supported it easily.
    # Standard TRL flow is to load the model.
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        MODEL_NAME,
        peft_config=lora_config,
        device_map="auto",
        # load_in_4bit=True, # Optional: enable if GPU is very small
    )

    # Patch: Experimental PPO expects generation_config on the wrapper, but TRL 0.29+ wrapper migth miss it.
    if not hasattr(model, "generation_config") and hasattr(model, "pretrained_model"):
        model.generation_config = model.pretrained_model.generation_config
    
    # Patch: Experimental PPO also needs base_model_prefix for the value model wrapper
    if not hasattr(model, "base_model_prefix") and hasattr(model, "pretrained_model"):
        model.base_model_prefix = model.pretrained_model.base_model_prefix

    # Patch: Experimental PPO tries to access the backbone (e.g. .model) directly on the wrapper
    # based on base_model_prefix. We need to expose it.
    if hasattr(model, "base_model_prefix") and hasattr(model, "pretrained_model"):
        backbone_name = model.base_model_prefix
        if not hasattr(model, backbone_name) and hasattr(model.pretrained_model, backbone_name):
            setattr(model, backbone_name, getattr(model.pretrained_model, backbone_name))

    # Patch: Experimental PPO checks is_gradient_checkpointing on the wrapper
    if not hasattr(model, "is_gradient_checkpointing"):
        if hasattr(model, "pretrained_model") and hasattr(model.pretrained_model, "is_gradient_checkpointing"):
            model.is_gradient_checkpointing = model.pretrained_model.is_gradient_checkpointing
        else:
            model.is_gradient_checkpointing = False # Default to False if not found

    # If we found a checkpoint, we might need to manually load weights or skip steps.
    # TRL's PPO trainer doesn't have a 'resume_from_checkpoint' fully unified like Trainer yet in all versions.
    # WE will implement step skipping.
    
    start_step = 0
    if latest_ckpt:
        print(f"Checkpoints found. Latest: {latest_ckpt}")
        try:
            step_str = latest_ckpt.split("-")[-1]
            start_step = int(step_str)
            print(f"Resuming from step {start_step}...")
        except Exception as e:
            print(f"Error parsing checkpoint: {e}")
    else:
        print("No checkpoints found. Starting fresh training.")

    print("Building dataset...")
    dataset = build_dataset(tokenizer, dataset)

    # Collaborative Data Loader
    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    # 5. Initialize Trainer
    print("Initializing PPOTrainer...")
    ppo_trainer = PPOTrainer(
        args=config,
        model=model,
        reward_model=model, # Required by experimental API; we bypass it by passing explicit rewards
        value_model=model,  # Required by experimental API; shared with policy in this setup
        ref_model=None, # shared ref model with PEFT
        processing_class=tokenizer,
        train_dataset=dataset, # 'dataset' might be 'train_dataset' in Trainer
        data_collator=collator,
    )

    # DEBUG: Inspect the trainer object to find the correct API
    print("DEBUG: PPOTrainer attributes:", dir(ppo_trainer))
    
    # 6. Training Loop
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 64, # Short generation for math/speed
    }

    # If dataset is smaller than total steps, we cycle
    data_iter = iter(ppo_trainer.dataloader)

    print(f"Starting training loop for {TOTAL_STEPS} steps...")
    # Force tqdm to stdout to ensure visibility
    for step in tqdm(range(start_step, TOTAL_STEPS), file=sys.stdout):
        print(f"--- Step {step+1}/{TOTAL_STEPS} ---")

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(ppo_trainer.dataloader)
            batch = next(data_iter)

        query_tensors = batch["input_ids"]

        # Run PPO Step
        #### Get response from Causal LM
        # Experimental PPOTrainer doesn't have .generate(), so we use the model directly.
        # We access 'pretrained_model' to bypass the wrapper and ensure we have .device and .generate
        raw_model = model.pretrained_model if hasattr(model, "pretrained_model") else model
        
        # 1. Prepare inputs (stack list of tensors -> batch tensor)
        # query_tensors is a list of 1D tensors.
        generation_inputs = torch.nn.utils.rnn.pad_sequence(
            query_tensors, batch_first=True, padding_value=tokenizer.pad_token_id
        ).to(raw_model.device)
        
        # 2. Generate
        generated_ids = raw_model.generate(
            input_ids=generation_inputs,
            attention_mask=(generation_inputs != tokenizer.pad_token_id).long(),
            **generation_kwargs
        )

        # 3. Extract purely the response (slice off the prompt)
        response_tensors = []
        for i in range(len(generated_ids)):
            # Warning: this assumes left-padding or strict length matching? 
            # query_tensors[i] tells us the specific input length for this sample
            input_len = len(query_tensors[i]) 
            response_tensors.append(generated_ids[i][input_len:])
            
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        #### Compute Rewards
        # 1. RLVR Reward
        rlvr_rewards = verify_reward_func(batch["response"], answer=batch["answer"])
        
        # 2. RLAIF Reward
        rlaif_rewards = ai_feedback_reward_func(batch["response"])
        
        # Combine Rewards (Weighted Sum)
        # e.g., 0.5 * Verified + 0.5 * AI
        combined_rewards = [
            torch.tensor(0.5 * r_v + 0.5 * r_a, dtype=torch.float32) 
            for r_v, r_a in zip(rlvr_rewards, rlaif_rewards)
        ]

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, combined_rewards)
        ppo_trainer.log_stats(stats, batch, combined_rewards)
        
        # Explicit print for user visibility
        avg_reward = sum([r.item() for r in combined_rewards]) / len(combined_rewards)
        print(f"Step {step+1} Report: Avg Reward = {avg_reward:.4f} | RLVR={rlvr_rewards[0]} | RLAIF={rlaif_rewards[0]}")

        # Save Checkpoint
        if (step + 1) % SAVE_FREQ == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint-{step+1}")
            print(f"Saving checkpoint to {ckpt_path}")
            ppo_trainer.save_pretrained(ckpt_path)

    print("Training complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("CRITICAL ERROR IN MAIN EXECUTION:")
        traceback.print_exc()
        raise e
