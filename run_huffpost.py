import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
from groupdro_ensemble import solve_group_dro_ensemble
from srm_ensemble import solve_srm_ensemble
from erm import solve_erm_cross_entropy
from irm import solve_irm_ensemble
from rse import solve_robust_offline_with_proximal_hf, calculate_batch_moments
from kl_rs import solve_kl_rs_ensemble
import os 
import sys
from helpers import (
    load_huffpost_temporal_split,
    calculate_metrics_with_groups
    
)


if __name__ == "__main__":
    
    
    json_files = [
        'huffpost/wildtime_huffpost_falcon3_7b_it.json',
        'huffpost/wildtime_huffpost_gemma3_12b_it.json',
        'huffpost/wildtime_huffpost_gptoss_20b.json',
        'huffpost/wildtime_huffpost_mistral_7b_it.json',
        'huffpost/wildtime_huffpost_phi_4.json',
        'huffpost/wildtime_huffpost_qwen3_4b.json',
        'huffpost/wildtime_huffpost_mistral_small_24b.json',
        'huffpost/wildtime_huffpost_phi3.5_mini_it.json'
        
    ]
    
    TRAIN_START_YEAR = 2012
    SPLIT_YEAR = 2015
    TEST_END_YEAR = 2018
    BASE_SEED = 42
    
    valid_files = [f for f in json_files if os.path.exists(f)]
    if not valid_files: sys.exit("Error: No valid JSON files found.")

   
    train_preds, train_labels, train_years, test_preds, test_labels, test_years, M, K, class_names = \
        load_huffpost_temporal_split(valid_files, SPLIT_YEAR, 
                                     start_year=TRAIN_START_YEAR, 
                                     test_end_year=TEST_END_YEAR)
        
    
    q_s = np.bincount(train_labels, minlength=K) / len(train_labels)
    A_tr, b_tr = calculate_batch_moments(train_preds, train_labels, M, K)
    
    print(f"\n--- Class Distribution (Train) ---")
    for i, name in enumerate(class_names):
        count = np.sum(train_labels == i)
        print(f"  {name}: {count} ({100*count/len(train_labels):.1f}%)")
    
    
    weights = {}
    
    training_start_time = time.perf_counter()
    # A. ERM 
    weights['ERM'] = solve_erm_cross_entropy(train_preds, train_labels, M)
    training_end_time = time.perf_counter()
    print(f"\nTotal Training Time for ERM: {training_end_time - training_start_time:.2f} seconds.")


    # B. RSE 
    traininig_start_time = time.perf_counter()
    w_erm_offline, w_off, kappa_history, time_robust = solve_robust_offline_with_proximal_hf(
    A_tr, b_tr, q_s, M, K, 
    0.19, T_max_iter=2000, 
    epsilon_convergence=1e-5,
    eta=0.2 
            )
    print(kappa_history)
    weights['RSE'] = w_off    
    training_end_time = time.perf_counter()

    print(f"\nTotal Training Time for RSE: {training_end_time - traininig_start_time:.2f} seconds.")
    
    # C. IRM
    training_start_time = time.perf_counter()
    weights['IRM'], _ = solve_irm_ensemble(train_preds, train_labels, q_s, M, lambda_irm=0.4, lr=0.01)
    training_end_time = time.perf_counter()
    print(f"\nTotal Training Time for IRM: {training_end_time - training_start_time:.2f} seconds.")
    
    
    # E. Group DRO

    print("\nSolving Group DRO...")
    training_start_time = time.perf_counter()
    w_dro, _, _ = solve_group_dro_ensemble(
        train_preds, train_labels,M, K)
    weights['GroupDRO'] = w_dro
    training_end_time = time.perf_counter()
    print(f"\nTotal Training Time for Group DRO: {training_end_time - training_start_time:.2f} seconds.")

    

    # F. SRM
    print("\nSolving SRM...")
    training_start_time = time.perf_counter()
    w_srm, _, _ = solve_srm_ensemble(train_preds, train_labels, M, K)
    weights['SRM'] = w_srm
    training_end_time = time.perf_counter()
    print(f"\nTotal Training Time for SRM: {training_end_time - training_start_time:.2f} seconds.")
    
    # G. KL-RS
    print("\nSolving KL-RS...")
    start_klrs = time.perf_counter()
    
    w_klrs, lambda_klrs = solve_kl_rs_ensemble(train_preds, train_labels, M, tau_factor=1.1, lambda_min=1e-5, lambda_max=1000.0)
    weights['KL-RS'] = w_klrs
    training_end_time = time.perf_counter()
    print(f"\nTotal Training Time for KL-RS: {training_end_time - start_klrs:.2f} seconds.")

    
    individual_risks = []
    for m in range(M):
        ensemble_probs = test_preds[m, :, :]
        predicted_labels = np.argmax(ensemble_probs, axis=1)
        acc = np.mean(predicted_labels == test_labels)
        one_hot = np.zeros((len(test_labels), K))
        one_hot[np.arange(len(test_labels)), test_labels] = 1.0
        risk = np.mean(np.sum((one_hot - ensemble_probs) ** 2, axis=1))
        individual_risks.append(risk)
    best_model_idx = np.argmin(individual_risks)
    best_model_weights = np.zeros(M); best_model_weights[best_model_idx] = 1.0
    print(best_model_weights)
    weights['BestModel'] = best_model_weights
    
    print("\nAdding individual models to evaluation registry...")
    for m in range(M):
        
        raw_name = os.path.basename(json_files[m])
        clean_name = raw_name.replace('wildtime_huffpost_', '').replace('.json', '')
        
        
        if len(clean_name) > 15:
            clean_name = clean_name.replace('mistral_', 'mis_').replace('falcon_', 'fal_')
        
       
        w_ind = np.zeros(M)
        w_ind[m] = 1.0
        
        
        weights[clean_name] = w_ind

    print(f"\nGenerating Kappa Plot...")
    plt.figure(figsize=(8, 5))
    plt.plot(kappa_history, color='#2ecc71', linewidth=2, label=r'RSE $\kappa$ value')
    plt.title(f"RSE: Minimization of Difficulty Parameter ($\\kappa$)\nTrain: {TRAIN_START_YEAR}-{SPLIT_YEAR}", fontsize=12)
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel(r"Kappa ($\kappa$)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("rse_kappa_convergence.png", dpi=300)
    plt.show()
    print("[+] Saved rse_kappa_convergence.png")

    print(f"\n{'='*120}")
    print(f"TEST RESULTS (Train: {TRAIN_START_YEAR}-{SPLIT_YEAR} | Test: {SPLIT_YEAR+1}-{TEST_END_YEAR})")
    print(f"NOTE: 'Year Avg' = Mean of Yearly Accuracies | 'Year Worst' = Min of Yearly Accuracies")
    print(f"{'='*120}")
    
    
    header = f"{'Method':<12} | {'Year Avg':<10} | {'Year Worst':<12} | {'Overall Acc':<12} | {'Risk':<8} | {'Worst Class':<12}"
    print(header)
    print("-" * 120)
    
    all_group_stats = {}
    all_year_stats = {}
    
    for name, w in weights.items():
    
        acc, risk, wga, g_stats, _, y_stats = calculate_metrics_with_groups(
            w, test_preds, test_labels, test_years, K
        )
        all_group_stats[name] = g_stats
        all_year_stats[name] = y_stats
        
    
        yearly_acc_values = [v for k, v in y_stats.items() if k > SPLIT_YEAR]
        paper_avg = np.mean(yearly_acc_values) if yearly_acc_values else 0.0
        
        
        paper_worst = np.min(yearly_acc_values) if yearly_acc_values else 0.0
        
        print(f"{name:<12} | {paper_avg:.4f}     | {paper_worst:.4f}       | {acc:.4f}       | {risk:.4f}   | {wga:.4f}")

    print("-" * 120)

    
    print(f"\nGenerating Accuracy Comparison Bar Chart...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = list(weights.keys())
    overall_accs = []      
    mean_group_accs = []  
    worst_group_accs = [] 
    
    for name in methods:
        w = weights[name]
        ensemble_probs = np.einsum('m,mnk->nk', w, test_preds)
        predicted_labels = np.argmax(ensemble_probs, axis=1)
        overall_acc = np.mean(predicted_labels == test_labels)
        overall_accs.append(overall_acc)
        
       
        g_stats = all_group_stats[name]
        valid_accs = [v for v in g_stats.values() if not np.isnan(v)]
        
        
        mean_group_accs.append(np.mean(valid_accs) if valid_accs else 0.0)
        
       
        worst_group_accs.append(min(valid_accs) if valid_accs else 0.0)
    
    x = np.arange(len(methods))
    width = 0.25  
    
    bars1 = ax.bar(x - width, overall_accs, width, label='Overall Acc (sample-weighted)', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, mean_group_accs, width, label='Mean Group Acc (class-weighted)', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, worst_group_accs, width, label='Worst Group Acc', color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Accuracy Metrics Comparison\n(Train: {TRAIN_START_YEAR}-{SPLIT_YEAR}, Test: {SPLIT_YEAR+1}-{TEST_END_YEAR})', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png", dpi=300)
    plt.show()
    print("[+] Saved accuracy_comparison.png")

    
    print(f"\nGenerating Per-Group Accuracy Heatmap...")
    
    
    group_acc_matrix = np.zeros((len(methods), K))
    for i, name in enumerate(methods):
        for g in range(K):
            group_acc_matrix[i, g] = all_group_stats[name].get(g, np.nan)
    
    
    short_class_names = [name[:15] + '...' if len(name) > 15 else name for name in class_names]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    
    im = ax.imshow(group_acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
   
    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(short_class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(methods, fontsize=10)
    
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Accuracy', rotation=-90, va="bottom", fontsize=10)
    
   
    for i in range(len(methods)):
        for j in range(K):
            val = group_acc_matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=7)
    
    ax.set_title(f'Per-Group (Class) Accuracy Heatmap\n(Test: {SPLIT_YEAR+1}-{TEST_END_YEAR})', fontsize=12)
    plt.tight_layout()
    plt.savefig("per_group_accuracy_heatmap.png", dpi=300)
    plt.show()
    print("[+] Saved per_group_accuracy_heatmap.png")

    
    print(f"\nGenerating Temporal Shift Plot...")
    plt.figure(figsize=(10, 6))
    test_u_years = sorted(np.unique(test_years))
    
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    for idx, (name, stats) in enumerate(all_year_stats.items()):
        accs = [stats.get(y, 0.0) for y in test_u_years]
        print(f"{name} Yearly Accuracies: {accs}")
        plt.plot(test_u_years, accs, marker=markers[idx % len(markers)], 
                 color=colors[idx], label=name, linewidth=2, markersize=8)
        
    plt.title(f"Temporal Shift Robustness (Train: {TRAIN_START_YEAR}-{SPLIT_YEAR})", fontsize=12)
    plt.xlabel("Year", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(test_u_years)
    plt.tight_layout()
    plt.savefig(f"temporal_shift_plot_{SPLIT_YEAR}.png", dpi=300)
    plt.show()
    print(f"[+] Saved temporal_shift_plot_{SPLIT_YEAR}.png")

    
    print(f"\n{'='*80}")
    print("DETAILED PER-GROUP ACCURACY")
    print(f"{'='*80}")
    
    
    test_group_counts = np.bincount(test_labels, minlength=K)
    
    header = f"{'Class':<20} | {'Count':<8} |"
    for name in methods:
        header += f" {name:<10} |"
    print(header)
    print("-" * len(header))
    
    for g in range(K):
        row = f"{class_names[g][:20]:<20} | {test_group_counts[g]:<8} |"
        for name in methods:
            acc = all_group_stats[name].get(g, np.nan)
            row += f" {acc:.4f}     |" if not np.isnan(acc) else f" {'N/A':<10} |"
        print(row)
    
    # =========================================================================
    # === SIGNIFICANCE TESTING (Bootstrap on Average & Worst Year Accuracy) ===
    # =========================================================================
    print("\n" + "="*100)
    print("=== SIGNIFICANCE TESTING (Bootstrap on HuffPost Temporal Results) ===")
    print("="*100)

    B = 1000  
    N_test = len(test_labels)
    hero_method = 'RSE'
    baselines = ['GroupDRO', 'IRM', 'SRM', 'KL-RS', 'ERM'] 

    
    method_correctness = {}
    for name in [hero_method] + baselines:
        w = weights[name]
        w_norm = w / w.sum() if w.sum() > 0 else np.full(M, 1.0/M)
        ensemble_preds = np.tensordot(w_norm, test_preds, axes=([0], [0]))
        pred_labels = np.argmax(ensemble_preds, axis=1)
        method_correctness[name] = (pred_labels == test_labels).astype(int)

    
    boot_results = {m: {'aya': np.zeros(B), 'wya': np.zeros(B)} for m in [hero_method] + baselines}
    test_years_arr = np.array(test_years)
    unique_test_years = np.unique(test_years_arr)

    np.random.seed(BASE_SEED)
    for i in range(B):
        
        indices = np.random.choice(N_test, size=N_test, replace=True)
        sampled_years = test_years_arr[indices]

        for m in [hero_method] + baselines:
            sampled_correct = method_correctness[m][indices]
            
            
            y_accs = []
            for y in unique_test_years:
                mask = (sampled_years == y)
                if np.sum(mask) > 0:
                    y_accs.append(np.mean(sampled_correct[mask]) * 100)
            
            
            if y_accs:
                boot_results[m]['aya'][i] = np.mean(y_accs)
                boot_results[m]['wya'][i] = np.min(y_accs)

    
    metrics = [
        ("Average Yearly Accuracy (AYA)", 'aya'),
        ("Worst-Year Accuracy (WYA)", 'wya')
    ]

    for metric_name, metric_key in metrics:
        print(f"\n--- {metric_name} Significance Table ---")
        print(f"(Scores % ± Std Dev over {B} bootstrap samples, * p<0.05, ** p<0.01 vs RSE)\n")
        
        hero_scores = boot_results[hero_method][metric_key]
        hero_mean = np.mean(hero_scores)
        hero_std = np.std(hero_scores)
        
        header = f"{'Method':<12} | {'Score ± Std':<15} | {'p-value':<10} | {'Sig.'}"
        print(header)
        print("-" * len(header))
        
        print(f"{hero_method+' (Ours)':<12} | {hero_mean:.2f} ± {hero_std:.2f} | {'-':<10} | {'-'}")
        
        for b in baselines:
            b_scores = boot_results[b][metric_key]
            b_mean = np.mean(b_scores)
            b_std = np.std(b_scores)
            
            
            if np.all(hero_scores == b_scores):
                p_val = 1.0
            else:
                p_val = np.mean(b_scores >= hero_scores)
            
            sig_marker = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            
            display_name = "DRO" if b == "GroupDRO" else b
            print(f"{display_name:<12} | {b_mean:.2f} ± {b_std:.2f} | {p_val:.4f}     | {sig_marker}")
            
    print("\n" + "="*100)