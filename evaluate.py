import argparse
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

def extract_answer(text):
    """
    Extracts the last number following '####' which is the standard GSM8K format.
    If not found, tries to find the last number in the text.
    """
    # Look for the gold standard delimiter
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(',', '')
    
    # Fallback: Look for "The answer is X" pattern
    match = re.search(r"[Tt]he answer is\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(',', '')
    
    # Weak fallback: Last number in the text
    matches = re.findall(r"(-?[\d,]+(?:\.\d+)?)", text)
    if matches:
        return matches[-1].replace(',', '')
    
    return None

def evaluate(args):
    print(f"📊 Loading Model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # Load Adapter if provided (RLAIF Trained)
    if args.adapter_path:
        print(f"🔧 Loading Adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model_type = "RLAIF Trained 🧠"
    else:
        model_type = "Base Model 🐣"

    print(f"📚 Loading GSM8K Test Set...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # Select subset for speed if requested
    if args.num_samples > 0:
        dataset = dataset.select(range(min(len(dataset), args.num_samples)))
    
    print(f"🚀 Starting Evaluation: {model_type}")
    print(f"samples: {len(dataset)}")
    
    correct = 0
    total = 0
    
    questions = []
    golds = []
    preds = []
    correct_flags = []

    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        question = example['question']
        gold_answer_str = example['answer']
        gold_answer = extract_answer(gold_answer_str)
        
        questions.append(question)
        golds.append(gold_answer)
        
        # Format the Input using Chat Template if available, or raw text
        # Qwen-Instruct models expect chat format
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the problem step by step and end your response with 'The answer is <number>'."},
            {"role": "user", "content": question}
        ]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0, # Deterministic for Eval
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract Answer
        pred_answer = extract_answer(generated_text)
        preds.append(pred_answer)
        
        # Scoring
        is_correct = False
        if pred_answer and gold_answer:
            try:
                # Compare as floats to handle 1.0 vs 1
                if abs(float(pred_answer) - float(gold_answer)) < 1e-6:
                    is_correct = True
            except:
                pass # Parse error, count as wrong
        
        correct_flags.append(is_correct)
        if is_correct:
            correct += 1
        total += 1
        
        # Live Log for the first few
        if i < 3:
            print("-" * 40)
            print(f"❓ Q: {question[:100]}...")
            print(f"🤖 A: {generated_text[:100]}... [Pred: {pred_answer}]")
            print(f"✅ Gold: {gold_answer} | {'CORRECT' if is_correct else 'WRONG'}")

    accuracy = (correct / total) * 100
    print("=" * 40)
    print(f"🏆 FINAL RESULT: {model_type}")
    print(f"🎯 Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print("=" * 40)

    # Save detailed results if requested
    if args.output_file:
        import json
        print(f"💾 Saving detailed results to {args.output_file}...")
        results = []
        for i in range(len(preds)):
            results.append({
                "question": questions[i],
                "gold": golds[i],
                "pred": preds[i],
                "correct": correct_flags[i]
            })
        
        with open(args.output_file, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print("✅ Log saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model HF ID")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to trained adapter (optional)")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to test (0 for all)")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save output JSONL (e.g. results.jsonl)")
    args = parser.parse_args()
    
    evaluate(args)
