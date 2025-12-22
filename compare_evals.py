import argparse
import json

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
    print("="*50)
    
    # --- VISUALIZATION GEN ---
    try:
        import matplotlib.pyplot as plt
        # Set backend to 'Agg' to avoid "Tcl_AsyncDelete" errors on headless clusters
        plt.switch_backend('Agg') 
        
        labels = ['Both Correct', 'Both Wrong', 'Wins (Fixed)', 'Regressions (Broke)']
        sizes = [ties_high, ties_low, wins, regressions]
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f'] # Green, Red, Blue, Yellow
        
        # Calculate percentages
        sizes_pct = [s/total*100 for s in sizes]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, sizes_pct, color=colors, edgecolor='black', alpha=0.8)
        
        plt.title(f'Evaluation Comparison (N={total})', fontsize=14, pad=20)
        plt.ylabel('Percentage of Test Set (%)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add Count Labels on top of bars
        for bar, count in zip(bars, sizes):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{count}\n({height:.1f}%)',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
            
        plt.tight_layout()
        plt.savefig('comparison_chart.png', dpi=300)
        print("\n📈 Graph generated: 'comparison_chart.png'")
        
        # Create detailed confusion matrix plot
        plt.figure(figsize=(6, 6))
        # Matrix format: [[High(BothRight), Loss(Regression)], [Win(Fixed), Low(BothWrong)]]
        # This is a bit abstract, simply a 2x2 grid
        
        plt.text(0.5, 0.75, f"BOTH RIGHT\n{ties_high} ({ties_high/total*100:.1f}%)", 
                 ha='center', va='center', fontsize=14, color='green', bbox=dict(facecolor='#e8f8f5', alpha=0.5))
        
        plt.text(0.5, 0.25, f"BOTH WRONG\n{ties_low} ({ties_low/total*100:.1f}%)", 
                 ha='center', va='center', fontsize=14, color='red', bbox=dict(facecolor='#fdebd0', alpha=0.5))
        
        plt.text(0.25, 0.5, f"REGRESSIONS\n{regressions} ({regressions/total*100:.1f}%)", 
                 ha='center', va='center', fontsize=12, color='orange')
        
        plt.text(0.75, 0.5, f"WINS (FIXED)\n{wins} ({wins/total*100:.1f}%)", 
                 ha='center', va='center', fontsize=12, color='blue')
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title("Performance Shift Matrix", fontsize=14)
        plt.tight_layout()
        plt.savefig('comparison_matrix.png', dpi=300)
        print("📈 Matrix generated: 'comparison_matrix.png'")
        
    except ImportError:
        print("\n⚠️ Matplotlib not found. Skipping graph generation.")
    except Exception as e:
        print(f"\n⚠️ Graph generation failed: {e}")

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
