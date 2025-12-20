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
from dataclasses import dataclass
from transformers.modeling_outputs import CausalLMOutputWithPast

# Define Output Class globally so Accelerate can handle it
@dataclass
class CausalLMOutputWithValue(CausalLMOutputWithPast):
    value: torch.FloatTensor = None


import argparse

# Default Configuration
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

def parse_args():
    parser = argparse.ArgumentParser(description="RLAIF/RLVR Training Script")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "research"], help="Training mode: 'demo' (fast, small) or 'research' (full dataset, longer)")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--total_steps", type=int, default=0, help="Total optimization steps (0 = auto based on mode)")
    return parser.parse_args()

args = parse_args()

# Dynamic Configuration based on Mode
MODEL_NAME = DEFAULT_MODEL
OUTPUT_DIR = "checkpoints"
LOG_DIR = "logs"
LEARNING_RATE = 1.41e-5

if args.mode == "research":
    # Research Mode: Real Training
    BATCH_SIZE = args.batch_size if args.batch_size > 1 else 4
    GRADIENT_ACCUMULATION_STEPS = args.accum_steps if args.accum_steps > 1 else 4
    TOTAL_STEPS = args.total_steps if args.total_steps > 0 else 1000
    SAVE_FREQ = 50
    DATASETS_SPLIT = "train" # Full dataset
    MINI_BATCH_SIZE = BATCH_SIZE
else:
    # Demo Mode: Fast Debugging
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 1
    TOTAL_STEPS = 100
    SAVE_FREQ = 10
    DATASETS_SPLIT = "train[:100]" # Tiny subset
    MINI_BATCH_SIZE = 1

print(f"=== Running in {args.mode.upper()} mode ===")
print(f"Batch Size: {BATCH_SIZE} | Accum Steps: {GRADIENT_ACCUMULATION_STEPS} | Total Steps: {TOTAL_STEPS}")
print(f"Dataset Split: {DATASETS_SPLIT}")

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
    dataset = load_dataset("gsm8k", "main", split=DATASETS_SPLIT)


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
    # Load Model with Value Head
    # Explicitly creating the hierarchy: Base -> PEFT -> ValueHeadWrapper
    # This prevents TRL from returning just a PeftModel without the head.
    print("Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        return_dict=True,
    )
    base_model.config.padding_side = "left" # Just in case
    
    print("Applying LoRA...")
    from peft import get_peft_model
    model = get_peft_model(base_model, lora_config)
    
    print("Wrapping with Value Head (Safe Wrapper)...")
    # Custom wrapper to enforce object outputs
    class SafeAutoModelForCausalLMWithValueHead(AutoModelForCausalLMWithValueHead):
        def forward(self, *args, **kwargs):
            # Force return_dict=True
            kwargs["return_dict"] = True
            output = super().forward(*args, **kwargs)
            
            # Failsafe: if output is still a tuple, convert it
            if isinstance(output, tuple):
                logits = output[0]
                value = output[-1]
                
                # Use kwargs to be safe against field ordering (loss vs logits)
                return CausalLMOutputWithValue(logits=logits, value=value)
            
            return output

    model = SafeAutoModelForCausalLMWithValueHead.from_pretrained(model)

    # Patch Hooks: The 'set_device_hook' in TRL/Accelerate crashes on our Object.
    # We must find and replace it with a hook that handles Objects.
    def fix_model_hooks(model):
        for key, hook in model._forward_hooks.items():
            if hasattr(hook, "__name__") and "set_device_hook" in hook.__name__:
                print(f"Patching hook: {hook.__name__}")
                
                # Define robust hook
                def robust_set_device_hook(module, inputs, outputs):
                    # Check for our object
                    if hasattr(outputs, "logits") and hasattr(outputs, "value"):
                        # Move to device if needed
                        # (Trainer usually handles this, but let's emulate hook behavior)
                        if hasattr(module, "device"):
                            dev = module.device
                            outputs.logits = outputs.logits.to(dev)
                            outputs.value = outputs.value.to(dev)
                        return outputs
                    
                    # Fallback to original logic (iterating tuple)
                    # We can't easily call original hook if it's bound?
                    # Replicate basic behavior: tuple(o.to(device) for o in outputs)
                    try:
                        new_outputs = []
                        dev = module.device if hasattr(module, "device") else "cpu"
                        for output in outputs:
                            new_outputs.append(output.to(dev))
                        return tuple(new_outputs)
                    except Exception as e:
                        # If iteration fails, return generic
                        return outputs
                
                model._forward_hooks[key] = robust_set_device_hook
    
    fix_model_hooks(model)
    
    # Ensure config propagation
    model.config.return_dict = True
    if hasattr(model, "pretrained_model"):
         model.pretrained_model.config.return_dict = True

    # Patch: Experimental PPO expects generation_config on the wrapper
    if not hasattr(model, "generation_config") and hasattr(model, "pretrained_model"):
        model.generation_config = model.pretrained_model.generation_config
    
    # Patch: Experimental PPO needs base_model_prefix to find the backbone
    if not hasattr(model, "base_model_prefix") and hasattr(model, "pretrained_model"):
        model.base_model_prefix = model.pretrained_model.base_model_prefix
    
    # Patch: Experimental PPO tries to access certain attributes via base_model_prefix
    if hasattr(model, "base_model_prefix") and hasattr(model, "pretrained_model"):
        backbone_name = model.base_model_prefix
        if not hasattr(model, backbone_name):
             # Ensure the backbone is accessible directly on the wrapper
             # Usually pretrained_model IS the backbone (or Close to it)
             # or we can look it up in pretrained_model
             if hasattr(model.pretrained_model, backbone_name):
                 setattr(model, backbone_name, getattr(model.pretrained_model, backbone_name))
             else:
                 # If the name mismatches, just alias pretrained_model as the backbone
                 setattr(model, backbone_name, model.pretrained_model)
    
    # Patch: Experimental PPO check
    if not hasattr(model, "is_gradient_checkpointing"):
        model.is_gradient_checkpointing = getattr(model.pretrained_model, "is_gradient_checkpointing", False)

    # REMOVED: Monkey-patch score. The real wrapper has .v_head and should work if configured right.
    # If the experimental trainer needs .score, we can map it to .v_head if missing on the wrapper?
    # The standard behavior of AutoModelForCausalLMWithValueHead is to return (logits, _, value)
    # The experimental trainer seems to look for 'score' method specifically on reward models?
    # But for the POLICY model (which is also value model), does it call .score?
    # Usually it calls 'forward' and gets value from output.
    # Let's keep the score patch ONLY if missing, just in case.
    if not hasattr(model, "score") and hasattr(model, "v_head"):
         # Simple bridge
         def score_bridge(self, hidden_states):
             return self.v_head(hidden_states)
         import types
         model.score = types.MethodType(score_bridge, model)

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
            # Patch: TRL's get_reward utility inspects the model for 'base_model_prefix'
            # and then tries to access that attribute.
            self.base_model_prefix = "model"
            self.config = tokenizer # Dummy config just in case
        
        @property
        def model(self):
            # Return self when 'model' attribute is accessed
            # avoiding infinite recursion in nn.Module structure
            return self

        def score(self, hidden_states):
            # hidden_states: [batch, seq_len, dim]
            # TRL expects rewards for EACH token (or at least matching sequence length)
            # causing the index error when we returned [batch, 1] and it tried to mask it.
            
            batch_size, seq_len, _ = hidden_states.shape
            
            # Broadcast our single sequence reward to all tokens
            # or put it on the last token?
            # PPO usually masking handles it, but the shape must likely match [batch, seq_len]
            # Let's expand it.
            if self._current_rewards is None:
                 return torch.zeros((batch_size, seq_len), device=hidden_states.device)
            
            # _current_rewards is [batch, 1]
            return self._current_rewards.expand(batch_size, seq_len)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            # 1. Decode inputs
            decoded_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            rewards = []
            
            for text in decoded_texts:
                # 2. Extract Question and Answer
                try:
                    parts = text.split("Answer:")
                    if len(parts) < 2:
                        rewards.append(0.0) # Format failure
                        continue
                    
                    question_part = parts[0].replace("Question:", "").strip()
                    response_part = parts[1].strip()
                    
                    # 3. Lookup Ground Truth
                    ground_truth = QUESTION_TO_ANSWER.get(question_part, None)
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
                
                # LOGGING SAMPLE FOR STREAMLIT
                try:
                    import json
                    import time
                    log_entry = {
                        "type": "sample",
                        "step": 0, # We don't have step here easily, but timestamp works
                        "timestamp": time.time(),
                        "question": parts[0] if len(parts) > 0 else "Unknown",
                        "response": parts[1] if len(parts) > 1 else "Unknown",
                        "reward": rewards[-1]
                    }
                    with open("logs/metrics.jsonl", "a") as f:
                        f.write(json.dumps(log_entry) + "\n")
                except Exception as e:
                    pass # Don't crash training for logging

            
            # Store rewards for the .score() call
            # Output shape must be [batch, seq_len] or [batch] for score? 
            # Usually score return [batch, seq_len] for token-level rewards?
            # Or [batch_size] for sentence level? 
            # PPO usually expects values per token or per sequence.
            # If we return [batch, 1], it might be broadcasted.
            # Let's assume one reward per sequence.
            current_rewards = torch.tensor(rewards, dtype=torch.float32, device=input_ids.device)
            # Reshape to [batch, 1] to match "score" expectations often?
            self._current_rewards = current_rewards.unsqueeze(1) 
            
            # Return a dummy output that satisfies get_reward(..., output.hidden_states)
            # We need output.hidden_states to exist.
            # We can use a namedtuple or simple Class
            class DummyOutput:
                def __init__(self):
                    self.hidden_states = [torch.zeros_like(input_ids, dtype=torch.float32).unsqueeze(-1)] # Dummy last hidden state
            
            return DummyOutput()

    reward_model = ProgrammaticRewardModel(tokenizer)

    # 5. Initialize Trainer
    
    # Custom Callback for Streamlit Logging
    from transformers import TrainerCallback
    import json
    import time
    
    METRICS_LOG_FILE = os.path.join(LOG_DIR, "metrics.jsonl")
    # Ensure log dir exists
    os.makedirs(LOG_DIR, exist_ok=True)
    
    class StreamlitLogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                entry = {
                    "type": "metrics",
                    "step": state.global_step,
                    "timestamp": time.time(),
                    **logs
                }
                with open(METRICS_LOG_FILE, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                    
    # Patch Reward Model to Log Samples
    # We inject logging directly into the forward pass where we have the text
    original_forward = reward_model.forward
    
    def logging_forward(input_ids, attention_mask=None, **kwargs):
        # Call original to get rewards/output
        # We need to capture the text that was decoded inside.
        # Since we can't easily access locals() of the original function,
        # we will rely on re-decoding here or modifying the class.
        # Modifying the class is cleaner but let's just re-decode for logging purposes (low overhead for demo).
        
        # Actually better: The original forward uses iteration.
        # Let's just modify the class instance method directly?
        # Or simpler: Just rely on the class we defined above? 
        # Wait, the class `ProgrammaticRewardModel` is defined in THIS file (lines 170+).
        # I should edit the Class Definition instead of patching here.
        return original_forward(input_ids, attention_mask, **kwargs)

    # I will edit the class definition directly in a subsequent step for samples.
    # For now, let's attach the callback.

    print("Initializing PPOTrainer with ProgrammaticRewardModel...")
    ppo_trainer = PPOTrainer(
        args=config,
        model=model,
        reward_model=reward_model, 
        value_model=model,  
        ref_model=ref_model, 
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=dataset, 
        data_collator=collator,
        callbacks=[StreamlitLogCallback()]
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
        print(f"RESUMING from checkpoint: {checkpoint}")
        # Standard HF Trainer supports resume_from_checkpoint
        try:
            ppo_trainer.train(resume_from_checkpoint=checkpoint)
        except TypeError as e:
            print(f"Warning: Could not pass resume_from_checkpoint to train(): {e}")
            print("Attempting to continue without explicit resume flag (Trainer might auto-detect)...")
            ppo_trainer.train()
    else:
        print("Starting fresh training...")
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
