import re
from typing import List

def extract_answer(text: str) -> str:
    """
    Extracts the answer from the text.
    It looks for the last occurrence of \boxed{...} which is common in GSM8K.
    """
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    if matches:
        return matches[-1].strip()
    return ""

def verify_reward_func(completions, **kwargs) -> List[float]:
    """
    RLVR Reward Function:
    Checks if the extracted answer matches the ground truth.
    Expects 'solution' or 'answer' in the key-word arguments from the dataset.
    """
    rewards = []
    # Dataset usually provides 'answer' or 'solution' column.
    # TRL passes the batch data in kwargs.
    # We'll assume the dataset has an 'answer' column.
    ground_truths = kwargs.get("answer", [])
    
    # If not present (test time), return 0s
    if not ground_truths:
        return [0.0] * len(completions)

    for completion, truth in zip(completions, ground_truths):
        predicted = extract_answer(completion)
        # Simple string matching; could be robustified (e.g. float parsing)
        # GSM8K clean answers often just numbers or simple expressions.
        if predicted == truth.strip():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards

def ai_feedback_reward_func(completions, **kwargs) -> List[float]:
    """
    RLAIF Reward Function:
    Uses a heuristic or a secondary model call to judge quality.
    For this 'small' demo, we will use a heuristic:
    - Reward length of reasoning (longer chain of thought might be better, up to a point).
    - Reward presence of "Step 1", "Therefore", etc.
    
    NOTE: In a full RLAIF setup, this would call a separate LLM (or the same one) 
    to prompt-engineer a score.
    """
    rewards = []
    for text in completions:
        score = 0.0
        # Heuristic 1: Structure
        if "Step 1" in text or "First" in text:
            score += 0.1
        if "Therefore" in text or "Thus" in text:
            score += 0.1
        
        # Heuristic 2: Presence of a boxed answer (even if wrong)
        if "\\boxed{" in text:
            score += 0.3
            
        rewards.append(score)
    return rewards
