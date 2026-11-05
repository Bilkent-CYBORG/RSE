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
from rse import solve_robust_offline_with_proximal, calculate_batch_moments
from kl_rs import solve_kl_rs_ensemble
from helpers import (
    load_predictions_from_files_trec as load_predictions_from_files, 
    split_train_test_pool, 
    evaluate_on_shifted_data,
    evaluate_majority_vote,
    get_alphas_for_equidistant_kl,
    generate_shift_distributions,
    create_label_shifted_dataset,
    calculate_kl_divergence,
)
import pandas as pd
from scipy.stats import ttest_rel
if __name__ == "__main__":

    
    plt.rcParams.update({
    # 1. Use the system's LaTeX to render text (results in Type 1 fonts)
    'text.usetex': True,
    
    # 2. Use the Times font package
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{times}', 
    
    # 3. Standard figure settings
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'figure.figsize': (3.25, 2.5),
    'savefig.bbox': 'tight',
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})
    sns.set_palette("husl")

    # --- Top-Level Configuration ---
    N_EXPERIMENTS = 20
    BASE_SEED = 42
    all_experiment_results = []
    all_experiment_summaries = []
    all_experiment_training_times = []
    first_exp_logs_offline = None
    first_exp_logs_online = None

    TREC_LABEL_ORDER = ["description and abstract concepts", "entities", "abbreviation", 
                        "human beings", "numeric values", "locations"]
    
    TREC_GT_KEY = "ground_truth_label"

    training_files = [
        "trec/trec_results_train_phi4_15b.json",
        'trec/trec_results_train_mistral_7b_it.json',
        'trec/trec_results_train_gemma3_12b_it.json',
        'trec/trec_results_train_gpt_oss_20b.json',
        'trec/trec_results_train_falcon3_7b_it.json',
        'trec/trec_results_train_mistral_24b_it.json',
        'trec/trec_results_train_qwen3_4b.json',
        'trec/trec_results_train_phi3.5_mini_it.json',
   
    ]
    plot_styles = {
        'ERM': {'color': "#45B5E9", 'marker': 'o', 'linestyle': '-', 'ms': 12, 'linewidth': 4},
        'RSE': {'color': '#2ecc71', 'marker': 's', 'linestyle': '--', 'ms': 7, 'linewidth': 2},
        'Robust (Online)': {'color': "#0e4c40", 'marker': 'X', 'linestyle': '-.', 'ms': 7, 'linewidth': 2},
        'DRO': {'color': '#e74c3c', 'marker': 'D', 'linestyle': ':', 'ms': 8,   'linewidth': 2},
        'KL-RS': {'color': '#d35400', 'marker': 'h', 'linestyle': '--', 'ms': 8, 'linewidth': 2.5},
        'SRM': {'color': '#9b59b6', 'marker': '^', 'linestyle': '--', 'ms': 7, 'linewidth': 2},
        'IRM': {'color': "#000000", 'marker': '*', 'linestyle': ':', 'ms': 5, 'linewidth': 1.5},
        'AccuracyWeighted': {'color': '#e91e63', 'marker': '+', 'linestyle': '-', 'ms': 7, 'linewidth': 2},
        'BestIndividual': {'color': '#795548', 'marker': 'p', 'linestyle': ':', 'ms': 7, 'linewidth': 2},
        'Uniform': {'color': '#95a5a6', 'marker': '.', 'linestyle': ':', 'ms': 7,   'linewidth': 2},
        'MajorityVote': {'color': '#34495e', 'marker': 'v', 'linestyle': '--', 'ms': 7, 'linewidth': 2},
        'phi4': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'ms': 7, 'linewidth': 2},
        'Mistral 7b': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--', 'ms': 7, 'linewidth': 2},
        'Gemma3 12b': {'color': "#0B0C0B", 'marker': 'D', 'linestyle': '-.', 'ms': 7, 'linewidth': 2},
        'Gpt Oss 20b': {'color': '#d62728', 'marker': '^', 'linestyle': ':', 'ms': 7, 'linewidth': 2},
        'Falcon3 7b': {'color': '#9467bd', 'marker': 'h', 'linestyle': '-', 'ms': 7, 'linewidth': 2},
        'Mistral Small 24b': {'color': '#8c564b', 'marker': 'v', 'linestyle': '--', 'ms': 7, 'linewidth': 2},
        'Qwen3 4b': {'color': '#e377c2', 'marker': 'p', 'linestyle': '-.', 'ms': 7, 'linewidth': 2},
        'phi3.5 mini': {'color': '#7f7f7f', 'marker': '*', 'linestyle': ':', 'ms': 7, 'linewidth': 2},

    }

    
    for exp_num in range(N_EXPERIMENTS):
        print("\n" + "=" * 70 + f"\n=== EXPERIMENT {exp_num + 1} / {N_EXPERIMENTS} ===\n" + "=" * 70)
        current_exp_seed = BASE_SEED + exp_num * 50
        training_times = {} 

        
        TEST_POOL_FRACTION = 0.1
        VALIDATION_FRACTION = 0.1
        tau_target_fraction = 0.62
        shift_types_to_test = ['minority_target']
        n_trials_per_shift = 5
        generated_test_set_size = 500
        
        
        all_predictions, all_labels, M, K, N_total = load_predictions_from_files(training_files, TREC_LABEL_ORDER, gt_key=TREC_GT_KEY)

        train_pool_preds, train_pool_labels, test_pool_preds, test_pool_labels = \
            split_train_test_pool(all_predictions, all_labels, K, TEST_POOL_FRACTION, seed=current_exp_seed)
        
        actual_train_pools_preds, actual_train_pool_labels, actual_val_pools_preds, actual_val_pool_labels = \
            split_train_test_pool(train_pool_preds, train_pool_labels, K, VALIDATION_FRACTION, seed=current_exp_seed)
        
        calibrated_test_pool_preds = test_pool_preds

        class_accuracies = np.zeros((M, K))
        for m in range(M):
            for k in range(K):
                idx = np.where(actual_val_pool_labels == k)[0]
                if len(idx) > 0:
                    class_accuracies[m, k] = np.mean(np.argmax(actual_val_pools_preds[m, idx], axis=1) == actual_val_pool_labels[idx])
        hardest_class_index = np.nanargmin(np.nanmean(class_accuracies, axis=0))
        
        
        train_preds_actual, train_labels_actual = actual_train_pools_preds, actual_train_pool_labels
        
        q_s_offline = np.bincount(train_labels_actual, minlength=K) / len(train_labels_actual)

        print("\n--- Calculating Equidistant KL Alphas ---")
        magnitudes_to_test = get_alphas_for_equidistant_kl(
            q_source=q_s_offline, 
            shift_type='minority_target', 
            K=K, 
            target_class_index=hardest_class_index, 
            num_points=11 
        )

        

        # == Phase 2: Training ==
        A_matrices_offline, b_vectors_offline = calculate_batch_moments(train_preds_actual, train_labels_actual, M, K)

        start_gd = time.perf_counter()
        w_erm_offline, w_robust_offline, logs_offline, time_robust = solve_robust_offline_with_proximal(
        A_matrices_offline, b_vectors_offline, q_s_offline, M, K, 
        tau_target_fraction, T_max_iter=2000, 
        epsilon_convergence=1e-5,
        eta=0.2 
                )
        training_times['RSE'] =time.perf_counter() - start_gd
        print('the offline logs are:',logs_offline)

        
        start_gd = time.perf_counter()
        w_groupdro, _, _ = solve_group_dro_ensemble(train_preds_actual, train_labels_actual, M, K)
        training_times['GroupDRO'] = time.perf_counter() - start_gd

        start_gd = time.perf_counter()
        w_erm_ce = solve_erm_cross_entropy(train_preds_actual, train_labels_actual, M)
        training_times['ERM_CE'] = time.perf_counter() - start_gd

        
        start_srm = time.perf_counter()

        w_srm, _, _ = solve_srm_ensemble(train_preds_actual, train_labels_actual,M,K)

        training_times['SRM'] = time.perf_counter() - start_srm
        

        start_klrs = time.perf_counter()
        
        w_klrs, lambda_klrs = solve_kl_rs_ensemble(
            train_preds_actual, train_labels_actual, M, 
            tau_factor=1.1, lambda_min=1e-5, lambda_max=1000.0
        )
        training_times['KL-RS'] = time.perf_counter() - start_klrs
        
        
    
        w_irm, time_irm = solve_irm_ensemble(train_preds_actual, train_labels_actual, q_s_offline, M, lambda_irm=0.4, lr=0.01)
        training_times['IRM'] = time_irm
        
        
        w_uniform = np.full(M, 1.0 / M)
        if exp_num == 0:
            first_exp_logs_offline = logs_offline
        
        individual_accuracies = []
        for m in range(M):
            if len(test_pool_labels) == 0: acc = 0.0
            else:
                pred_labels_m = np.argmax(test_pool_preds[m], axis=1)
                acc = np.mean(pred_labels_m == test_pool_labels)
            individual_accuracies.append(acc)
        
        individual_accuracies = np.array(individual_accuracies)
        if individual_accuracies.sum() > 0:
            w_acc_weighted = individual_accuracies / individual_accuracies.sum()
        else:
            w_acc_weighted = np.full(M, 1.0 / M)
            
        best_model_idx = np.argmax(individual_accuracies)
        w_best_individual = np.zeros(M)
        w_best_individual[best_model_idx] = 1.0
        print('the name of best model is:', training_files[best_model_idx])

        w_phi4_15b = np.zeros(M)
        w_phi4_15b[0] = 1.0
        w_mistral_7b_it = np.zeros(M)
        w_mistral_7b_it[1] = 1.0
        w_gemma3_12b_it = np.zeros(M)
        w_gemma3_12b_it[2] = 1.0
        w_gpt_oss_20b = np.zeros(M)
        w_gpt_oss_20b[3] = 1.0
        w_falcon3_7b_it = np.zeros(M)
        w_falcon3_7b_it[4] = 1.0
        w_mistral_small_24b_it = np.zeros(M)
        w_mistral_small_24b_it[5] = 1.0
        w_qwen3_4b = np.zeros(M)
        w_qwen3_4b[6] = 1.0
        w_phi3_5_mini_it = np.zeros(M)
        w_phi3_5_mini_it[7] = 1.0



        weights_to_test = {
            'RSE': w_robust_offline,
            'phi4': w_phi4_15b,
            'Mistral 7b': w_mistral_7b_it,
            'Gemma3 12b': w_gemma3_12b_it,
            'Gpt Oss 20b': w_gpt_oss_20b,
            'Falcon3 7b': w_falcon3_7b_it,
           'Mistral Small 24b': w_mistral_small_24b_it,
            'Qwen3 4b': w_qwen3_4b,
            'phi3.5 mini': w_phi3_5_mini_it,
            'DRO': w_groupdro, 'IRM': w_irm, 'SRM': w_srm, 'KL-RS': w_klrs, 'ERM': w_erm_ce,
            'Uniform': w_uniform, 'MajorityVote': None,'AccuracyWeighted': w_acc_weighted 
            
        }

        
        for m in range(M):
            w_single = np.zeros(M)
            w_single[m] = 1.0
            model_short_name = training_files[m].split('/')[-1].replace('sst5_results_train_', '').replace('.json', '')
            weights_to_test[f'Individual_{model_short_name}'] = w_single
        
       
        if exp_num == 0: mean_vectors_by_mag = defaultdict(list)
        results = defaultdict(lambda: defaultdict(list))
        q_test_pool = np.bincount(test_pool_labels, minlength=K) / len(test_pool_labels)
        test_count = 0

        for shift_type in shift_types_to_test:
            for magnitude in magnitudes_to_test:
                for trial in range(n_trials_per_shift):
                    test_count += 1
                    target_dist = generate_shift_distributions(q_test_pool, shift_type, magnitude, K, hardest_class_index)
                    shifted_preds, shifted_labels = create_label_shifted_dataset(
                        calibrated_test_pool_preds, test_pool_labels, target_dist, generated_test_set_size, seed=current_exp_seed+test_count)
                    
                    if shifted_preds is None: continue
                    
                    if exp_num == 0: mean_vectors_by_mag[magnitude].append(np.mean(shifted_preds, axis=1).flatten())
                    
                    actual_dist = np.bincount(shifted_labels, minlength=K) / len(shifted_labels)
                    kl_div = calculate_kl_divergence(actual_dist, q_s_offline)

                    for name, w_vec in weights_to_test.items():
                        if name == 'MajorityVote':
                            eval_result = evaluate_majority_vote(shifted_preds, shifted_labels)
                        else:
                            w = w_vec / w_vec.sum() if w_vec.sum() > 0 else np.full(M, 1.0/M)
                            eval_result = evaluate_on_shifted_data(w, shifted_preds, shifted_labels)
                        
                        eval_result['kl_divergence'] = kl_div
                        results[name][(shift_type, magnitude)].append(eval_result)
        
        if exp_num == 0:
            print(f"\n=== Distribution Shift Visualization ===")
            
            
            q_test_by_magnitude = {0.0: q_s_offline.copy()}  
            for magnitude in magnitudes_to_test:
                if magnitude == 0.0:
                    continue
                target_dist = generate_shift_distributions(q_s_offline, 'minority_target', 
                                                            magnitude, K, hardest_class_index)
                q_test_by_magnitude[magnitude] = target_dist
            
    
                
        
        all_experiment_results.append(results)
        all_experiment_training_times.append(training_times)
        current_exp_summary = {}
        for name in weights_to_test.keys():
            all_accs = [t['accuracy'] for k in results.get(name, {}) for t in results[name].get(k, [])]
            current_exp_summary[name] = {'avg_acc': np.mean(all_accs) if all_accs else np.nan, 
                                         'min_acc': np.min(all_accs) if all_accs else np.nan,
                                         }
        all_experiment_summaries.append(current_exp_summary)

    
    print("\n" + "=" * 70 + f"\n=== FINAL AVERAGED RESULTS ({N_EXPERIMENTS} Experiments) ===\n" + "=" * 70)

    avg_times = defaultdict(float)
    for t in all_experiment_training_times:
        for n, v in t.items(): avg_times[n] += v
    print("Avg Training Times:")
    for n, v in avg_times.items(): print(f"  {n}: {v/len(all_experiment_training_times):.4f}s")

    avg_summary_stats = defaultdict(lambda: defaultdict(list))
    for s in all_experiment_summaries:
        for n, stats in s.items():
            for k, v in stats.items(): avg_summary_stats[n][k].append(v)
    
    final_summary_json = {}
    for n, stats in avg_summary_stats.items():
        final_summary_json[n] = {
            'avg_acc': np.mean(stats['avg_acc']), 'std_acc': np.std(stats['avg_acc']),
            
        }
    with open('AVERAGED_final_trec_OVERALL_summary.json', 'w') as f: json.dump(final_summary_json, f, indent=4)

    
    plot_data_aggregator = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ex_strat = list(weights_to_test.keys())[0]
    all_keys = list(all_experiment_results[0].get(ex_strat, {}).keys())
    
    for key in all_keys:
        for res in all_experiment_results:
            kl_vals = [t['kl_divergence'] for t in res.get(ex_strat, {}).get(key, [])]
            avg_kl = np.mean(kl_vals) if kl_vals else 0.0
            for name in weights_to_test.keys():
                trials = res.get(name, {}).get(key, [])
                if trials:
                    plot_data_aggregator[name][key]['mean_accs'].append(np.mean([t['accuracy'] for t in trials]))
                    
                    plot_data_aggregator[name][key]['kls'].append(avg_kl)

    final_plot_data = defaultdict(lambda: defaultdict(list))
    for name, k_data in plot_data_aggregator.items():
        for key, d_lists in k_data.items():
            if not d_lists['mean_accs']: continue
            final_plot_data[name]['key'].append(key)
            final_plot_data[name]['kl_divergences'].append(np.mean(d_lists['kls']))
            final_plot_data[name]['accuracies'].append(np.mean(d_lists['mean_accs']))
            final_plot_data[name]['accuracy_stds'].append(np.std(d_lists['mean_accs']))
            
    
    per_magnitude_results = defaultdict(lambda: defaultdict(dict))
    for name, data in final_plot_data.items():
        for idx, (s_type, mag) in enumerate(data['key']):
            
            per_magnitude_results[name][s_type][f"{mag:.2f}"] = {
                'avg_accuracy': float(data['accuracies'][idx]),
                'std_accuracy': float(data['accuracy_stds'][idx]),
                
                'avg_kl': float(data['kl_divergences'][idx])
            }

    with open('AVERAGED_per_magnitude_detailed_results_trec.json', 'w') as f:
        json.dump(per_magnitude_results, f, indent=4)
    print("✓ Saved detailed per-magnitude averages to 'AVERAGED_per_magnitude_detailed_results_trec.json'")
   
   


    
    available_baselines = [k for k in ['ERM', 'DRO', 'IRM', 'KL-RS', 'SRM'] 
                        if k in weights_to_test.keys()]

    

    
    print("\n" + "="*100)
    print(f"=== INDIVIDUAL MODEL PERFORMANCE (Averaged over {N_EXPERIMENTS} experiments & all shifts) ===")
    print("="*100)

    individual_model_summary = {}

    for name in weights_to_test.keys():
        if name.startswith('Individual_'):
            model_name = name.replace('Individual_', '')
            
            
            all_accs = []
            
            
            for exp_results in all_experiment_results:
                if name in exp_results:
                    for shift_key, trials in exp_results[name].items():
                        for trial in trials:
                            all_accs.append(trial['accuracy'])
                            
            
            individual_model_summary[model_name] = {
                'avg_acc': np.mean(all_accs) if all_accs else np.nan,
                'std_acc': np.std(all_accs) if all_accs else np.nan,
                'min_acc': np.min(all_accs) if all_accs else np.nan,
            }

    
    print(f"\n{'Model':<50} | {'Avg Acc':<10} | {'Std Acc':<10} | {'Min Acc':<10}")
    print("-" * 100)

    for model_name in sorted(individual_model_summary.keys(), 
                            key=lambda x: individual_model_summary[x]['avg_acc'], 
                            reverse=True):
        stats = individual_model_summary[model_name]
        print(f"{model_name:<50} | {stats['avg_acc']:.4f}     | {stats['std_acc']:.4f}     | {stats['min_acc']:.4f}")

    
    with open('AVERAGED_individual_model_summary.json', 'w') as f:
        json.dump(individual_model_summary, f, indent=4)

    print("\n✓ Saved individual model summary to 'AVERAGED_individual_model_summary.json'")

    import matplotlib.ticker as ticker

    
    num_shift_types = len(shift_types_to_test)
   
    fig_width = 3.5 if num_shift_types == 1 else 7.0 
    fig_height = 2.8 

    fig_acc, axes_acc = plt.subplots(1, num_shift_types, figsize=(fig_width, fig_height), squeeze=False)
    
    
    for col, shift_type in enumerate(shift_types_to_test):
        ax = axes_acc[0, col]
        
        
        methods_to_plot = sorted(final_plot_data.keys())
        
        for name in methods_to_plot:
            
            if name.startswith('Individual_'):
                continue
            
            d = final_plot_data[name]
            if not d['kl_divergences']: continue

            
            pts = sorted([(d['kl_divergences'][i], d['accuracies'][i], d['accuracy_stds'][i]) 
                          for i, k in enumerate(d['key']) if k[0]==shift_type], key=lambda x: x[0])
            
            if not pts: continue
            kls, accs, stds = zip(*pts)
            
            
            style = plot_styles.get(name, {'color': 'gray', 'marker': 'o', 'linestyle': '-', 'linewidth': 1})
            
            
            is_hero = (name == 'RSE') 
            
            
            lw = 2.5 if is_hero else 1.5
            alpha = 1.0 if is_hero else 0.7
            z_order = 10 if is_hero else 2  
            marker_interval = max(1, len(kls) // 8) 
            
            
            ax.plot(kls, accs, 
                    label=name, 
                    color=style['color'], 
                    marker=style['marker'], 
                    linestyle=style['linestyle'], 
                    linewidth=lw, 
                    markersize=5,
                    markevery=marker_interval, 
                    alpha=alpha,
                    zorder=z_order) 
            
            
            
        
        ax.set_xlabel(r'KL Divergence ($P_{test} || P_{train}$)', fontsize=11)
        if col == 0: 
            ax.set_ylabel('Avg. Accuracy', fontsize=11)
            
        
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
        sns.despine(ax=ax) 
        
        
        ax.set_xlim(left=0)

    
    lines, labels = axes_acc[0, 0].get_legend_handles_labels()
    
    
    if 'RSE' in labels:
        idx = labels.index('RSE')
        lines.insert(0, lines.pop(idx))
        labels.insert(0, labels.pop(idx))

    axes_acc[0, 0].legend(lines, labels, 
                          loc='lower left', 
                          fontsize=5, 
                          frameon=True, 
                          framealpha=0.6, 
                          edgecolor='white',
                          ncol=2) 

    plt.tight_layout()
    fig_acc.savefig('AVERAGED_final_trec_shift_accuracy_plot.pdf', format='pdf', dpi=300)
    print("✓ Saved accuracy plot")
    plt.show()

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

# ------------------------------------------------------------------
# GLOBAL STYLE
# ------------------------------------------------------------------
plt.rcParams.update({
    'font.family':          'serif',
    'text.usetex':          True,
    'font.size':            11,
    'axes.labelsize':       12,
    'axes.titlesize':       12,
    'xtick.labelsize':      10,
    'ytick.labelsize':      10,
    'axes.linewidth':       0.8,
    'xtick.major.width':    0.8,
    'ytick.major.width':    0.8,
    'xtick.major.size':     3.5,
    'ytick.major.size':     3.5,
    'legend.fontsize':      9.5,
    'legend.framealpha':    0.92,
    'legend.edgecolor':     '#cccccc',
    'lines.solid_capstyle': 'round',
    'pdf.fonttype':         42,
    'ps.fonttype':          42,
})

print("\nGenerating 3-Panel Concept Figure")

# ------------------------------------------------------------------
# DATA EXTRACTION
# ------------------------------------------------------------------
shift_type = shift_types_to_test[0]
model_keys = [k for k in final_plot_data.keys() if k.startswith('Individual_')]

model_decay_rates  = {}
model_initial_accs = {}
for mk in model_keys:
    pts = sorted(
        [(final_plot_data[mk]['key'][i][1], final_plot_data[mk]['accuracies'][i])
         for i, k in enumerate(final_plot_data[mk]['key']) if k[0] == shift_type],
        key=lambda x: x[0])
    _, accs = zip(*pts)
    model_initial_accs[mk] = accs[0]
    model_decay_rates[mk]  = abs(accs[0] - accs[-1])

w_dro_val = np.array(w_groupdro)
w_rse_val = np.array(w_robust_offline)

active_models = []
for mk in model_keys:
    orig_idx = -1
    for i, tf in enumerate(training_files):
        short_name = (tf.split('/')[-1]
                        .replace('sst5_results_train_', '')
                        .replace('trec_results_train_', '')
                        .replace('.json', ''))
        if mk == f'Individual_{short_name}' or mk.endswith(short_name):
            orig_idx = i
            break
    if orig_idx != -1:
        if w_dro_val[orig_idx] > 0.01 or w_rse_val[orig_idx] > 0.01:
            active_models.append((mk, orig_idx))

active_drops    = {mk: model_decay_rates[mk]  for mk, _ in active_models}
active_initials = {mk: model_initial_accs[mk] for mk, _ in active_models}

stable_model_key  = min(active_drops,    key=active_drops.get)
fragile_model_key = max(active_initials, key=active_initials.get)

# ------------------------------------------------------------------
# COLOR PALETTE
# ------------------------------------------------------------------
RSE_COLOR     = '#1f78b4'
STABLE_COLOR  = '#33a02c'
FRAGILE_COLOR = '#e31a1c'
DRO_COLOR     = '#6a3d9a'
OTHER_COLORS  = ['#ff7f00', '#01bfc4', '#f781bf', '#999999']
OTHER_MARKERS = ['s', '^', 'D', 'v']

model_colors  = {}
model_markers = {}
other_idx = 0
for mk, _ in active_models:
    if mk == stable_model_key:
        model_colors[mk]  = STABLE_COLOR
        model_markers[mk] = 's'
    elif mk == fragile_model_key:
        model_colors[mk]  = FRAGILE_COLOR
        model_markers[mk] = '^'
    else:
        model_colors[mk]  = OTHER_COLORS[other_idx % len(OTHER_COLORS)]
        model_markers[mk] = OTHER_MARKERS[other_idx % len(OTHER_MARKERS)]
        other_idx += 1

alphas = np.linspace(0, 1, 11)

pts_rse = sorted(
    [(final_plot_data['RSE']['key'][i][1], final_plot_data['RSE']['accuracies'][i])
     for i, k in enumerate(final_plot_data['RSE']['key']) if k[0] == shift_type],
    key=lambda x: x[0])
_, rse_accs = zip(*pts_rse)
rse_accs = np.array(rse_accs)

pts_dro = sorted(
    [(final_plot_data['DRO']['key'][i][1], final_plot_data['DRO']['accuracies'][i])
     for i, k in enumerate(final_plot_data['DRO']['key']) if k[0] == shift_type],
    key=lambda x: x[0])
_, dro_accs = zip(*pts_dro)
dro_accs = np.array(dro_accs)

# ------------------------------------------------------------------
# FIGURE LAYOUT — taller for bottom labels
# ------------------------------------------------------------------
fig = plt.figure(figsize=(11, 3.7))
gs  = gridspec.GridSpec(1, 3, figure=fig,
                         width_ratios=[1.5, 0.8, 1.5],
                         wspace=0.38)

gs.update(bottom=0.18)   

ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])
ax_c = fig.add_subplot(gs[2])

# ══════════════════════════════════════════════════════════════════
# PANEL A — Individual Classifier Profiles + RSE overlay
# ══════════════════════════════════════════════════════════════════

# Layer 1: background models
for mk, _ in active_models:
    if mk in (stable_model_key, fragile_model_key):
        continue
    pts = sorted(
        [(final_plot_data[mk]['key'][i][1], final_plot_data[mk]['accuracies'][i])
         for i, k in enumerate(final_plot_data[mk]['key']) if k[0] == shift_type],
        key=lambda x: x[0])
    _, accs = zip(*pts)
    ax_a.plot(alphas, accs,
              color=model_colors[mk],
              lw=1.3, alpha=0.55, linestyle='-',
              marker=model_markers[mk], markersize=3.5, markevery=2,
              markerfacecolor='white', markeredgewidth=1.2,
              zorder=2)

# Layer 2: named models
for mk in [stable_model_key, fragile_model_key]:
    pts = sorted(
        [(final_plot_data[mk]['key'][i][1], final_plot_data[mk]['accuracies'][i])
         for i, k in enumerate(final_plot_data[mk]['key']) if k[0] == shift_type],
        key=lambda x: x[0])
    _, accs = zip(*pts)
    ax_a.plot(alphas, accs,
              color=model_colors[mk],
              lw=2.1, alpha=0.92, linestyle='--', dashes=(6, 2.5),
              marker=model_markers[mk], markersize=4.5, markevery=2,
              markerfacecolor='white', markeredgewidth=1.5,
              zorder=6)

# Layer 3: RSE
ax_a.plot(alphas, rse_accs,
          color=RSE_COLOR, lw=2.8, alpha=1.0, linestyle='-',
          marker='o', markersize=5.5, markevery=2,
          markerfacecolor='white', markeredgewidth=1.8,
          zorder=20)

ax_a.set_xlabel(r'shift severity $\alpha$', labelpad=4)
ax_a.set_ylabel(r'accuracy', labelpad=4)
ax_a.set_xlim(-0.02, 1.02)

rse_handle = mlines.Line2D([], [], color=RSE_COLOR, lw=2.8,
                            marker='o', markersize=5.5,
                            markerfacecolor='white', markeredgewidth=1.8,
                            label=r'\textbf{RSE} (ours)')
stable_handle = mlines.Line2D([], [], color=STABLE_COLOR, lw=2.1,
                               linestyle='--', dashes=(6, 2.5),
                               marker='s', markersize=4.5,
                               markerfacecolor='white', markeredgewidth=1.5,
                               label=r'$f_{\mathrm{stable}}$: low avg., stable')
fragile_handle = mlines.Line2D([], [], color=FRAGILE_COLOR, lw=2.1,
                                linestyle='--', dashes=(6, 2.5),
                                marker='^', markersize=4.5,
                                markerfacecolor='white', markeredgewidth=1.5,
                                label=r'$f_{\mathrm{peak}}$: high init., decays')

ax_a.legend(handles=[rse_handle, stable_handle, fragile_handle],
            loc='lower left', frameon=True, fontsize=9,
            handlelength=2.4, handletextpad=0.5,
            borderpad=0.5, labelspacing=0.35)

sns.despine(ax=ax_a)
ax_a.text(0.5, -0.32, r'\textbf{(a)} Individual profiles',
          transform=ax_a.transAxes, fontsize=11, ha='center', va='top')

# ══════════════════════════════════════════════════════════════════
# PANEL B — Weight Assignment
# ══════════════════════════════════════════════════════════════════

sorted_active = sorted(active_models, key=lambda x: active_drops[x[0]])
bottoms = np.zeros(2)
bar_w   = 0.52
x_pos   = [0, 1]

for mk, orig_idx in sorted_active:
    wd = w_dro_val[orig_idx]
    wr = w_rse_val[orig_idx]
    c  = model_colors[mk]
    ax_b.bar(x_pos[0], wd, bar_w, bottom=bottoms[0],
             color=c, edgecolor='white', linewidth=0.7)
    ax_b.bar(x_pos[1], wr, bar_w, bottom=bottoms[1],
             color=c, edgecolor='white', linewidth=0.7)
    bottoms[0] += wd
    bottoms[1] += wr

ax_b.set_xticks(x_pos)
ax_b.set_xticklabels(['DRO', r'\textbf{RSE}'], fontsize=11)
ax_b.set_ylabel('weight', color='#444444', labelpad=4)
ax_b.set_xlim(-0.55, 1.55)
ax_b.set_ylim(0, 1.08)
ax_b.yaxis.set_label_position('left')
ax_b.yaxis.tick_left()
sns.despine(ax=ax_b, left=False, right=True, bottom=False)
ax_b.text(0.5, -0.32, r'\textbf{(b)} Weight assignment',
          transform=ax_b.transAxes, fontsize=11, ha='center', va='top')

# ══════════════════════════════════════════════════════════════════
# PANEL C — Ensemble Performance
# ══════════════════════════════════════════════════════════════════

# DRO — dashed + triangle markers
ax_c.plot(alphas, dro_accs,
          color=DRO_COLOR, lw=2.0,
          linestyle='--', dashes=(6, 2.5),
          alpha=0.82, zorder=5,
          marker='^', markersize=4.5, markevery=2,
          markerfacecolor='white', markeredgewidth=1.5)

# RSE — solid + circle markers
ax_c.plot(alphas, rse_accs,
          color=RSE_COLOR, lw=2.8, zorder=20,
          marker='o', markersize=5.5, markevery=2,
          markerfacecolor='white', markeredgewidth=1.8)

ax_c.fill_between(alphas, dro_accs, rse_accs,
                  where=(rse_accs > dro_accs),
                  color=RSE_COLOR, alpha=0.13, zorder=3)

delta_idx = int(0.75 * (len(alphas) - 1))
ax_c.annotate('',
    xy    =(alphas[delta_idx], rse_accs[delta_idx]),
    xytext=(alphas[delta_idx], dro_accs[delta_idx]),
    arrowprops=dict(arrowstyle='<->', color=RSE_COLOR,
                    lw=2.0, mutation_scale=10))
mid_y = (rse_accs[delta_idx] + dro_accs[delta_idx]) / 2
ax_c.text(alphas[delta_idx] + 0.035, mid_y,
          r'$\Delta_{\mathrm{gain}}$',
          color=RSE_COLOR, fontsize=11, va='center', fontweight='bold')

ax_c.set_xlabel(r'shift severity $\alpha$', labelpad=4)
ax_c.set_ylabel(r'ensemble accuracy', labelpad=4)
ax_c.set_xlim(-0.02, 1.02)
ax_c.set_ylim(0.68, 0.80)

dro_handle = mlines.Line2D([], [], color=DRO_COLOR, lw=2.0,
                            linestyle='--', dashes=(6, 2.5),
                            marker='^', markersize=4.5,
                            markerfacecolor='white', markeredgewidth=1.5,
                            label='DRO ensemble')
rse_c_handle = mlines.Line2D([], [], color=RSE_COLOR, lw=2.8,
                              marker='o', markersize=5.5,
                              markerfacecolor='white', markeredgewidth=1.8,
                              label=r'\textbf{RSE} (ours)')

leg = ax_c.legend(handles=[dro_handle, rse_c_handle],
                  loc='lower left', frameon=True, fontsize=9,
                  handlelength=2.4, handletextpad=0.5,
                  borderpad=0.5, labelspacing=0.35)
leg.get_frame().set_linewidth(0.6)

sns.despine(ax=ax_c)
ax_c.text(0.5, -0.32, r'\textbf{(c)} Ensemble performance',
          transform=ax_c.transAxes, fontsize=11, ha='center', va='top')

# ------------------------------------------------------------------
# SAVE
# ------------------------------------------------------------------
fig.savefig('trec_dro_vs_rse_concept_final.pdf',
            format='pdf', dpi=300,
            metadata={'Creator': ''})

print("✓ Saved: trec_dro_vs_rse_concept_final.pdf")
plt.show()



print("\n" + "=" * 100)
print("=== SIGNIFICANCE TESTING (Split-level paired t-test vs RSE) ===")
print("=" * 100)

hero_method = "RSE"
baselines = ["DRO", "IRM", "SRM", "KL-RS", "ERM"]
methods = [hero_method] + baselines

EXPECTED_RUNS = 20
EXPECTED_TRIALS = 5

if len(all_experiment_results) != EXPECTED_RUNS:
    raise ValueError(
        f"Expected {EXPECTED_RUNS} runs, "
        f"found {len(all_experiment_results)}."
    )

shift_type = shift_types_to_test[0]
sig_results = []
pvalue_results = []

for magnitude in magnitudes_to_test:
    key = (shift_type, magnitude)
    split_accs = {method: [] for method in methods}

    for run_idx, exp_res in enumerate(all_experiment_results, start=1):
        for method in methods:
            # .get() avoids creating missing entries in defaultdicts.
            trials = exp_res.get(method, {}).get(key)

            if trials is None or len(trials) != EXPECTED_TRIALS:
                count = 0 if trials is None else len(trials)
                raise ValueError(
                    f"Run {run_idx}, method {method}, shift {magnitude}: "
                    f"expected {EXPECTED_TRIALS} trials, found {count}. "
                    "Check that shift keys match across runs."
                )

            values = np.asarray(
                [trial["accuracy"] for trial in trials],
                dtype=float
            )

            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Non-finite accuracy in run {run_idx}, "
                    f"method {method}, shift {magnitude}."
                )

            # One observation per split: average its five test draws.
            split_accs[method].append(values.mean())

    hero_accs = np.asarray(split_accs[hero_method])
    hero_mean = hero_accs.mean() * 100
    hero_std = hero_accs.std(ddof=1) * 100

    row_data = {
        "Shift": f"{magnitude:.2f}",
        f"{hero_method} (Ours)": f"{hero_mean:.1f} ± {hero_std:.1f}"
    }
    pvalue_row = {"Shift": f"{magnitude:.2f}"}

    for baseline in baselines:
        baseline_accs = np.asarray(split_accs[baseline])
        baseline_mean = baseline_accs.mean() * 100
        baseline_std = baseline_accs.std(ddof=1) * 100

        if np.array_equal(hero_accs, baseline_accs):
            p_val = 1.0
        else:
            _, p_val = ttest_rel(
                hero_accs,
                baseline_accs,
                alternative="two-sided"
            )

        if not np.isfinite(p_val):
            raise ValueError(
                f"Undefined p-value for {baseline}, shift {magnitude}."
            )

        if p_val < 0.01:
            sig_marker = "**"
        elif p_val < 0.05:
            sig_marker = "*"
        else:
            sig_marker = ""

        row_data[baseline] = (
            f"{baseline_mean:.1f} ± {baseline_std:.1f}{sig_marker}"
        )
        pvalue_row[baseline] = float(p_val)

    sig_results.append(row_data)
    pvalue_results.append(pvalue_row)

cols = ["Shift", f"{hero_method} (Ours)"] + baselines
df_sig = pd.DataFrame(sig_results)[cols]
df_pvalues = pd.DataFrame(pvalue_results)[["Shift"] + baselines]

print(
    "\nAccuracy (%) ± sample SD across 20 split averages "
    "(five test draws per split)."
)
print(
    "Two-sided paired t-tests; unadjusted p-values. "
    "* p<0.05, ** p<0.01 vs RSE."
)
print(df_sig.to_string(index=False))

print("\nUnadjusted p-values:")
print(df_pvalues.to_string(index=False, float_format=lambda p: f"{p:.6g}"))
print("=" * 100 + "\n")
