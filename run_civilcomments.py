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
from helpers import (
    load_civil_comments_data,
    calculate_metrics_with_groups_civil_comments as calculate_metrics_with_groups
)

if __name__ == "__main__":
    N_EXPERIMENTS = 1
    BASE_SEED = 42
    
    
    label_map = {"non-toxic": 0, "toxic": 1}
    
    train_files = [
        'civil_comments/civilcomments_train_gemma3_12b_it.json',
        'civil_comments/civilcomments_train_gptoss_20b.json',
        'civil_comments/civilcomments_train_mistral_7b.json',
        'civil_comments/civilcomments_train_phi4.json',
        'civil_comments/civilcomments_train_qwen3_4b.json',
        'civil_comments/civilcomments_train_falcon3_7b_it.json',
        'civil_comments/civilcomments_train_mistral_24b.json',
        'civil_comments/civilcomments_train_phi3.5_mini.json'
        
    ]
    test_files = [
        'civil_comments/civilcomments_test_gemma3_12b_it.json',
        'civil_comments/civilcomments_test_gptoss_20b.json',
        'civil_comments/civilcomments_test_mistral_7b.json',
        'civil_comments/civilcomments_test_phi4.json',
        'civil_comments/civilcomments_test_qwen3_4b.json',
        'civil_comments/civilcomments_test_falcon3_7b_it.json',
        'civil_comments/civilcomments_test_mistral_24b.json',
        'civil_comments/civilcomments_test_phi3.5_mini.json'
        
    ]

    
    train_preds_full, train_labels_full, train_groups_full, test_preds, test_labels, test_groups, models, M, K = \
        load_civil_comments_data(train_files, test_files, label_map)

    final_results = defaultdict(list)
    final_group_stats = defaultdict(lambda: defaultdict(list))
    all_kappa_histories = []

    for exp in range(N_EXPERIMENTS):
        print(f"\n{'='*60}")
        print(f"--- Experiment {exp+1}/{N_EXPERIMENTS} ---")
        print(f"{'='*60}")
        current_seed = BASE_SEED + 50 * exp
        np.random.seed(current_seed)
        
      
        
        print(f"Training set size: {len(train_labels_full)}")
        
        print(f"Test set size: {len(test_labels)}")
        

        individual_val_accuracies = []
        for m in range(M):
            
            w_single = np.zeros(M)
            w_single[m] = 1.0
            
            
            val_acc, _, val_wga, _ = calculate_metrics_with_groups(
                w_single, test_preds, test_labels, test_groups    
            )
            individual_val_accuracies.append(val_acc)
            print(f"  Model {m} ({models[m]}): Acc = {val_acc:.4f}")
        
        best_model_idx = np.argmax(individual_val_accuracies)
        print(f"\n  >> Best Individual Model: {models[best_model_idx]} (idx={best_model_idx}, Acc={individual_val_accuracies[best_model_idx]:.4f})")


        w_best_individual = np.zeros(M)
        w_best_individual[best_model_idx] = 1.0
        
    
        q_s = np.bincount(train_labels_full, minlength=K) / len(train_labels_full)
        A_tr, b_tr = calculate_batch_moments(train_preds_full,train_labels_full, M, K)

        # 1. ERM 
        print("\nSolving ERM Ensemble (Cross Entropy)...")
        erm_time_start = time.perf_counter()
        w_erm_ce = solve_erm_cross_entropy(train_preds_full, train_labels_full, M)
        erm_time_end = time.perf_counter()
        print(f" ERM training finished in {erm_time_end - erm_time_start:.2f} seconds.")

        # 2.  (RSE)
        print("\nSolving Robust Ensemble (RSE)...")
        rse_time_start = time.perf_counter()
        w_erm_offline, w_off, kappa_hist, time_robust = solve_robust_offline_with_proximal_hf(
        A_tr, b_tr, q_s, M, K, 
        0.8, T_max_iter=2000, 
        epsilon_convergence=1e-5,
        eta=0.2 
                )
        print(kappa_hist)
        rse_time_end = time.perf_counter()
        print(f" RSE training finished in {rse_time_end - rse_time_start:.2f} seconds.")

        all_kappa_histories.append(kappa_hist)
        
        # 3. IRM
        print("\nSolving IRM Ensemble...")
        irm_time_start = time.perf_counter()
        w_irm, _ = solve_irm_ensemble(train_preds_full, train_labels_full, q_s, M, lambda_irm=0.4, lr=0.01)
        irm_time_end = time.perf_counter()
        print(f" IRM training finished in {irm_time_end - irm_time_start:.2f} seconds.")
        
        # 4. Group DRO
        print("\nSolving Group DRO...")
        dro_time_start = time.perf_counter()
        w_dro, _, _ = solve_group_dro_ensemble(train_preds_full, train_labels_full, M, K)
        dro_time_end = time.perf_counter()
        print(f" Group DRO training finished in {dro_time_end - dro_time_start:.2f} seconds.")

        # 5. SRM
        print("\nSolving SRM Ensemble...")
        srm_time_start = time.perf_counter()
        w_srm, _, _ = solve_srm_ensemble(train_preds_full, train_labels_full, M, K)
        srm_time_end = time.perf_counter()
        print(f" SRM training finished in {srm_time_end - srm_time_start:.2f} seconds.")

    
        # 6. KL-RS
        print("\nSolving KL-RS...")
        kl_rs_start = time.perf_counter()
        w_klrs, lambda_klrs = solve_kl_rs_ensemble(
            train_preds_full, train_labels_full, M, 
            tau_factor=1.1, lambda_min=1e-5, lambda_max=1000.0
        )
        kl_rs_end = time.perf_counter()
        print(f" KL-RS training finished in {kl_rs_end - kl_rs_start:.2f} seconds.")

        

        weights_zoo = {
            'ERM': w_erm_ce,
            'RSE': w_off,
            'DRO': w_dro, 
            'SRM': w_srm, 
            'IRM': w_irm,
            'KL-RS': w_klrs,
            'Best Individual': w_best_individual,
           
        }
        
        print("\nAdding individual models to evaluation registry...")
        for m in range(M):
            
            model_name = models[m] 
            
            w_ind = np.zeros(M)
            w_ind[m] = 1.0
            
            
            weights_zoo[model_name] = w_ind
        print("\n--- Test Set Evaluation ---")
        for name, w in weights_zoo.items():
            acc, risk, wga, g_stats = calculate_metrics_with_groups(w, test_preds, test_labels, test_groups)
            final_results[name].append([acc, wga, risk])
            for g_name, g_acc in g_stats.items():
                final_group_stats[name][g_name].append(g_acc)
            print(f"  {name:<20} | Acc: {acc:.4f} | WGA: {wga:.4f}")


    print("\n" + "="*60 + "\nFINAL RESULTS (Averaged)\n" + "="*60)
    print(f"{'Method':<20} | {'Acc':<8} | {'WGA':<8} | {'Risk':<8}")
    print("-" * 55)
    
    summary_wga = {}
    summary_acc = {}
    
    for name, res in final_results.items():
        arr = np.array(res)
        avg_acc = np.mean(arr[:, 0])
        avg_wga = np.mean(arr[:, 1])
        avg_risk = np.mean(arr[:, 2])
        summary_wga[name] = avg_wga
        summary_acc[name] = avg_acc
        print(f"{name:<20} | {avg_acc:.4f}   | {avg_wga:.4f}   | {avg_risk:.4f}")

   
    print("\n" + "="*100 + "\nDETAILED GROUP STATISTICS (ALL METHODS & INDIVIDUAL MODELS)\n" + "="*100)

    
    all_methods = sorted(list(final_group_stats.keys()))
    all_groups = set()
    for m in all_methods:
        all_groups.update(final_group_stats[m].keys())
    sorted_groups = sorted(list(all_groups))

    
    print("\n--- VIEW 1: GROUP-CENTRIC TABLE ---")
    
    
    col_width = max(len(m) for m in all_methods) + 2
    col_width = max(col_width, 12) # Minimum width
    
    
    header = f"{'Group':<20} |"
    for m in all_methods:
        
        display_name = (m[:10] + '..') if len(m) > 12 else m
        header += f" {display_name:^{col_width}} |"
    
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    
    for g in sorted_groups:
        row = f"{g:<20} |"
        for m in all_methods:
            if g in final_group_stats[m]:
                # Average across experiments
                val = np.mean(final_group_stats[m][g])
                row += f" {val:^{col_width}.4f} |"
            else:
                row += f" {'-':^{col_width}} |"
        print(row)

    
    print("\n\n--- VIEW 2: MODEL-CENTRIC TABLE (Transposed) ---")
    
    
    g_width = max(len(g) for g in sorted_groups) + 2
    g_width = max(g_width, 10)

    
    header_t = f"{'Method/Model':<30} |"
    for g in sorted_groups:
        header_t += f" {g:^{g_width}} |"
    
    print("-" * len(header_t))
    print(header_t)
    print("-" * len(header_t))

    
    for m in all_methods:
        row = f"{m:<30} |"
        for g in sorted_groups:
            if g in final_group_stats[m]:
                val = np.mean(final_group_stats[m][g])
                row += f" {val:^{g_width}.4f} |"
            else:
                row += f" {'-':^{g_width}} |"
        print(row)
    print("-" * len(header_t))

   
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(10, 6))
    for i, hist in enumerate(all_kappa_histories):
        plt.plot(hist, alpha=0.6, label=f'Exp {i+1}')
    plt.title("RSE: Convergence of Kappa (Difficulty Parameter)")
    plt.xlabel("Iteration")
    plt.ylabel("Kappa Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("kappa_convergence.png")
    print("\n[+] Saved kappa_convergence.png")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    methods = list(summary_wga.keys())
    wga_vals = [summary_wga[m] for m in methods]
    acc_vals = [summary_acc[m] for m in methods]
    
   
    sorted_idx = np.argsort(wga_vals)
    methods = [methods[i] for i in sorted_idx]
    wga_vals = [wga_vals[i] for i in sorted_idx]
    acc_vals = [acc_vals[i] for i in sorted_idx]
    
    colors = sns.color_palette("magma", len(methods))

    
    sns.barplot(x=methods, y=wga_vals, hue=methods, palette=colors, ax=axes[0], legend=False)
    axes[0].set_title("Worst-Group Accuracy (Higher is Better)")
    axes[0].set_ylabel("Accuracy")
    axes[0].tick_params(axis='x', rotation=45, labelsize=8) # Reduced label size for readability
    axes[0].set_ylim(0, max(wga_vals)*1.1)

    
    sns.barplot(x=methods, y=acc_vals, hue=methods, palette=colors, ax=axes[1], legend=False)
    axes[1].set_title("Overall Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].tick_params(axis='x', rotation=45, labelsize=8)
    axes[1].set_ylim(0, max(acc_vals)*1.1)

    plt.tight_layout()
    plt.savefig("accuracy_comparison.png")
    print("[+] Saved accuracy_comparison.png")

    # =========================================================================
    # === SIGNIFICANCE TESTING (Bootstrap Resampling on Fixed Test Set) ===
    # =========================================================================
    print("\n" + "="*100)
    print("=== SIGNIFICANCE TESTING (Bootstrap on CivilComments Test Set) ===")
    print("="*100)

    B = 1000  
    N_test = len(test_labels)
    hero_method = 'RSE'
    baselines = ['DRO', 'IRM', 'SRM', 'KL-RS', 'ERM']

   
    method_correctness = {}
    
    for name in [hero_method] + baselines:
        w = weights_zoo[name]
        w_norm = w / w.sum() if w.sum() > 0 else np.full(M, 1.0/M)
        
        
        ensemble_preds = np.tensordot(w_norm, test_preds, axes=([0], [0]))
        pred_labels = np.argmax(ensemble_preds, axis=1)
        
       
        is_correct = (pred_labels == test_labels).astype(int)
        method_correctness[name] = is_correct

    
    boot_results = {m: {'acc': np.zeros(B), 'wga': np.zeros(B)} for m in [hero_method] + baselines}
    test_groups_arr = np.array(test_groups)

    np.random.seed(BASE_SEED) 
    for i in range(B):
        
        indices = np.random.choice(N_test, size=N_test, replace=True)
        sampled_groups = test_groups_arr[indices]
        unique_groups = np.unique(sampled_groups)

        for m in [hero_method] + baselines:
            sampled_correct = method_correctness[m][indices]
            
            # Overall Accuracy
            boot_results[m]['acc'][i] = np.mean(sampled_correct) * 100
            
            # Worst-Group Accuracy (WGA)
            g_accs = [np.mean(sampled_correct[sampled_groups == g]) * 100 for g in unique_groups]
            boot_results[m]['wga'][i] = np.min(g_accs)

   
    for metric_name, metric_key in [("Overall Accuracy", 'acc'), ("Worst-Group Accuracy (WGA)", 'wga')]:
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
            
         
            if p_val < 0.01:
                sig_marker = "**"
            elif p_val < 0.05:
                sig_marker = "*"
            else:
                sig_marker = ""
                
            print(f"{b:<12} | {b_mean:.2f} ± {b_std:.2f} | {p_val:.4f}     | {sig_marker}")
            
    print("\n" + "="*100)