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
    tokenizer.padding_side = "left" # Critical for decoder-only generation

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
        return_dict=True, # Explicitly force ModelOutput objects
        # load_in_4bit=True, # Optional: enable if GPU is very small
    )
    # Ensure standard output format
    model.config.return_dict = True
    if hasattr(model, "pretrained_model"):
        model.pretrained_model.config.return_dict = True

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

    # Patch: Experimental PPOTrainer calls .score() on the model wrapper
    # The AutoModelForCausalLMWithValueHead wrapper uses 'v_head' but might not expose 'score'
    if not hasattr(model, "score"):
        def score_func(hidden_states):
            # hidden_states: [batch, seq, dim]
            # v_head expects same.
            return model.v_head(hidden_states)
        
        # Bind the method to the instance
        import types
        model.score = types.MethodType(score_func, model)

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

    # Explicitly load ref_model to ensure return_dict=True works
    # We use the standard AutoModelForCausalLM (no value head wrapper) to guarantee standard outputs
    print("Loading ref_model (Base AutoModelForCausalLM)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        return_dict=True,
    )
    # No patching needed for standard HF model

    print("Building dataset...")
    dataset = build_dataset(tokenizer, dataset)

    # Collaborative Data Loader
    def collator(data):
        # We need to pad input_ids to create a tensor
        # data["input_ids"] are already tensors due to ds.set_format(type="torch")
        input_ids = [d["input_ids"] for d in data]
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        
        batch = {
            "input_ids": padded_input_ids,
        }
        
        # Pass through other keys like 'answer', 'question' as lists
        for key in data[0]:
            if key not in ["input_ids", "attention_mask"]:
                 batch[key] = [d[key] for d in data]
                 
        return batch

    # Shared answer lookup for reward model
    # We populate this from the dataset
    QUESTION_TO_ANSWER = {}
    for item in dataset:
        # Assuming Qwen format: "Question: {q}\nAnswer:"
        # We need to reconstruct the prompt to match what the tokenizer produces?
        # Simpler: map the 'question' text itself.
        QUESTION_TO_ANSWER[item['question']] = item['answer']
    
    class ProgrammaticRewardModel(torch.nn.Module):
        def __init__(self, tokenizer):
            super().__init__()
            self.tokenizer = tokenizer
        
        def forward(self, input_ids, attention_mask=None, **kwargs):
            # 1. Decode inputs
            decoded_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            rewards = []
            
            for text in decoded_texts:
                # 2. Extract Question and Answer
                # Format: "Question: <q>\nAnswer:<response>"
                try:
                    parts = text.split("Answer:")
                    if len(parts) < 2:
                        rewards.append(0.0) # Format failure
                        continue
                    
                    question_part = parts[0].replace("Question:", "").strip()
                    response_part = parts[1].strip()
                    
                    # 3. Lookup Ground Truth
                    # We try to match the question string.
                    # This might be brittle if tokens change spaces, but for exact string it should work?
                    # Let's try flexible matching or just assume 1-1 map if batch is aligned?
                    # Actually, we can just use the parsing logic.
                    
                    ground_truth = QUESTION_TO_ANSWER.get(question_part, None)
                    
                    # If we can't find it directly by string, checking if any key is in the text might work
                    if ground_truth is None:
                        # Fallback: find which question is in this text
                        for q, a in QUESTION_TO_ANSWER.items():
                            if q in parts[0]:
                                ground_truth = a
                                break
                    
                    if ground_truth:
                        # 4. Compute Rewards
                        r_v = verify_reward_func([response_part], answer=[ground_truth])[0]
                        r_a = ai_feedback_reward_func([response_part])[0]
                        rewards.append(0.5 * r_v + 0.5 * r_a)
                    else:
                        rewards.append(0.0) # Could not find ground truth
                        
                except Exception as e:
                    print(f"Reward Error: {e}")
                    rewards.append(0.0)
            
            return torch.tensor(rewards, dtype=torch.float32, device=input_ids.device)

    reward_model = ProgrammaticRewardModel(tokenizer)

    # 5. Initialize Trainer
    print("Initializing PPOTrainer with ProgrammaticRewardModel...")
    ppo_trainer = PPOTrainer(
        args=config,
        model=model,
        reward_model=reward_model, 
        value_model=model,  
        ref_model=ref_model, 
        processing_class=tokenizer,
        train_dataset=dataset, 
        data_collator=collator,
    )

    # DEBUG: Inspect the trainer object to find the correct API
    print("DEBUG: PPOTrainer attributes:", dir(ppo_trainer))
    
    print(f"Starting standard training loop for {TOTAL_STEPS} steps...")
    
    # 6. Run Training
    # We use .train() which likely handles the loop.
    # We assume 'resume_from_checkpoint' argument is handled by Trainer if passed,
    # or we can pass resume_from_checkpoint=latest_ckpt if standard Trainer.
    
    checkpoint = latest_ckpt if latest_ckpt else None
    if checkpoint:
        print(f"Resuming from {checkpoint} (Note: Passing checkpoint to train() is not supported in this API version, relying on internal state if applicable)")
    
    # Experimental PPOTrainer.train() manual override doesn't accept resume_from_checkpoint
    ppo_trainer.train()
    
    print("Training complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("CRITICAL ERROR IN MAIN EXECUTION:")
        traceback.print_exc()
        raise e
