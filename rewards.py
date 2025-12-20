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

def is_numeric_match(pred: str, truth: str) -> bool:
    """
    Checks if two strings are numerically equivalent.
    Handles:
    - Commas: "1,234" == "1234"
    - Currency/Units: "$50" == "50", "50%" == "0.5" (maybe not %, keep simple)
    - Trailing zeros: "5.0" == "5"
    """
    if pred == truth:
        return True
    
    # Normalize: remove $ and , 
    def clean(s):
        s = s.replace(",", "").replace("$", "").replace(" ", "")
        return s
    
    pred_clean = clean(pred)
    truth_clean = clean(truth)
    
    try:
        # Try float comparison with tolerance
        p = float(pred_clean)
        t = float(truth_clean)
        return abs(p - t) < 1e-6
    except ValueError:
        # If text comparison failed and float conversion failed, they are diff
        return False

def verify_reward_func(completions, **kwargs) -> List[float]:
    """
    RLVR Reward Function:
    Checks if the extracted answer matches the ground truth.
    Expects 'solution' or 'answer' in the key-word arguments from the dataset.
    """
    rewards = []
    # Dataset usually provides 'answer' or 'solution' column.
    ground_truths = kwargs.get("answer", [])
    
    # If not present (test time), return 0s
    if not ground_truths:
        return [0.0] * len(completions)

    for completion, truth in zip(completions, ground_truths):
        predicted = extract_answer(completion)
        if not predicted:
             rewards.append(0.0)
             continue
             
        # Ground truth in GSM8K usually looks like "#### 1234"
        # We need to extract the number from the truth if it's formatted that way
        # But 'answer' column in 'main' split usually has the steps then #### number.
        # Let's extract the target from truth as well.
        target_match = re.search(r"####\s*(.+)", truth)
        if target_match:
            target = target_match.group(1).strip()
        else:
            # Fallback if truth is just the number or different format
            target = extract_answer(truth) or truth.strip()

        if is_numeric_match(predicted, target):
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
