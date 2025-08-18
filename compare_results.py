#!/usr/bin/env python3
"""
Compare ChatGPT results from both variants
"""

import json

def load_results(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None

def main():
    print("="*60)
    print("CHATGPT RNA INFERENCE RESULTS COMPARISON")
    print("="*60)
    
    # Load results
    name_only_file = "results/chatgpt_name_only_gpt_4o.json"
    name_seq_file = "results/chatgpt_name_and_sequence_gpt_4o.json"
    
    name_only_data = load_results(name_only_file)
    name_seq_data = load_results(name_seq_file)
    
    if not name_only_data or not name_seq_data:
        print("Could not load results files")
        return
    
    # Extract scores (last item contains averages)
    name_only_scores = name_only_data[-1]
    name_seq_scores = name_seq_data[-1]
    
    print("\nMETRICS COMPARISON:")
    print("-" * 60)
    print(f"{'Metric':<20} {'Name Only':<15} {'Name+Sequence':<15} {'Difference':<15}")
    print("-" * 60)
    
    metrics = ['average_bleu_1', 'average_bleu_2', 'average_bleu_3', 'average_bleu_4', 'average_simcse']
    
    for metric in metrics:
        if metric in name_only_scores and metric in name_seq_scores:
            name_only_val = name_only_scores[metric]
            name_seq_val = name_seq_data[-1][metric]
            diff = name_seq_val - name_only_val
            
            print(f"{metric:<20} {name_only_val:<15.4f} {name_seq_val:<15.4f} {diff:<15.4f}")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    # Determine which variant performed better
    bleu4_diff = name_seq_scores.get('average_bleu_4', 0) - name_only_scores.get('average_bleu_4', 0)
    simcse_diff = name_seq_scores.get('average_simcse', 0) - name_only_scores.get('average_simcse', 0)
    
    print(f"BLEU-4 difference: {bleu4_diff:.4f}")
    print(f"SimCSE difference: {simcse_diff:.4f}")
    
    if bleu4_diff > 0.01 and simcse_diff > 0.01:
        print("✅ Name+Sequence variant performed significantly better")
    elif bleu4_diff < -0.01 and simcse_diff < -0.01:
        print("✅ Name Only variant performed significantly better")
    else:
        print("⚠️  Performance is similar between variants")
    
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    # Show a few sample predictions
    print("\nName Only - Sample 1:")
    print(f"RNA: {name_only_data[0]['query']}")
    print(f"Predicted: {name_only_data[0]['predict_func'][:200]}...")
    
    print("\nName+Sequence - Sample 1:")
    print(f"RNA: {name_seq_data[0]['query'][:100]}...")
    print(f"Predicted: {name_seq_data[0]['predict_func'][:200]}...")

if __name__ == "__main__":
    main()
