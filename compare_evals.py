import argparse
import json
import pandas as pd

def compare(args):
    print(f"🕵️‍♂️ Comparing Results:\n  A: {args.baseline} (Baseline)\n  B: {args.trained} (Trained)")
    
    # Load Data
    with open(args.baseline, 'r') as f:
        data_a = [json.loads(line) for line in f]
    
    with open(args.trained, 'r') as f:
        data_b = [json.loads(line) for line in f]
    
    # Ensure alignment (Assuming deterministic order from evaluate.py on same dataset)
    if len(data_a) != len(data_b):
        print("⚠️ Warning: File lengths differ! Truncating to shorter length.")
        min_len = min(len(data_a), len(data_b))
        data_a, data_b = data_a[:min_len], data_b[:min_len]

    wins = 0        # A Wrong, B Correct
    regressions = 0 # A Correct, B Wrong
    ties_high = 0   # Both Correct
    ties_low = 0    # Both Wrong

    diff_samples = []

    for i in range(len(data_a)):
        base = data_a[i]
        trained = data_b[i]
        
        # Verify alignment
        if base['question'] != trained['question']:
             print(f"❌ Mismatch at index {i}: Questions differ!")
             break
        
        if not base['correct'] and trained['correct']:
            wins += 1
            diff_samples.append(("Win 🟢", base['question'], base['pred'], trained['pred'], base['gold']))
        
        elif base['correct'] and not trained['correct']:
            regressions += 1
            diff_samples.append(("Regression 🔴", base['question'], base['pred'], trained['pred'], base['gold']))
            
        elif base['correct'] and trained['correct']:
            ties_high += 1
        else:
            ties_low += 1

    total = len(data_a)
    acc_a = (ties_high + regressions) / total * 100
    acc_b = (ties_high + wins) / total * 100
    net_improvement = wins - regressions

    print("\n" + "="*50)
    print(f"📊 COMPARISON REPORT (N={total})")
    print("="*50)
    print(f"Baseline Accuracy : {acc_a:.2f}%")
    print(f"Trained Accuracy  : {acc_b:.2f}%")
    print(f"Net Improvement   : {acc_b - acc_a:+.2f}%")
    print("-" * 50)
    print(f"🟢 Wins (Fixed)      : {wins}")
    print(f"🔴 Regressions (Broke): {regressions}")
    print(f"🤝 Ties (Both Right) : {ties_high}")
    print(f"💀 Ties (Both Wrong) : {ties_low}")
    print("="*50)
    
    # Show samples
    print("\n🔍 DETAILED EXAMPLES (First 5 differences):")
    for type_label, q, a_base, a_trained, gold in diff_samples[:5]:
        print(f"\n{type_label} | Gold: {gold}")
        print(f"Q: {q[:100]}...")
        print(f"   Base: {a_base} | Trained: {a_trained}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", help="Path to baseline.jsonl")
    parser.add_argument("trained", help="Path to trained.jsonl")
    args = parser.parse_args()
    
    compare(args)
