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
    load_predictions_from_files_sst5 as load_predictions_from_files, 
    split_train_test_pool, 
    evaluate_on_shifted_data,
    evaluate_majority_vote,
    get_alphas_for_equidistant_kl,
    generate_shift_distributions,
    create_label_shifted_dataset,
    calculate_kl_divergence,
)
import seaborn as sns

N_EXPERIMENTS = 20
BASE_SEED = 42
TEST_POOL_FRACTION = 0.1
VALIDATION_FRACTION = 0.1
GENERATED_TEST_SET_SIZE = 1000
N_TRIALS_PER_SHIFT = 5
SST5_LABEL_ORDER = ["very negative", "negative", "neutral", "positive", "very positive"]
SST5_GT_KEY = "ground_truth_label"
TRAINING_FILES = [
    'sst5/sst5_results_train_qwen3_4b.json',
    'sst5/sst5_results_train_phi_4_15b.json',
    'sst5/sst5_results_train_phi3.5_mini_it.json',
    'sst5/sst5_results_train_gpt_oss_20b.json',
    'sst5/sst5_results_train_gemma3_12b.json',
    'sst5/sst5_results_train_mistral_7b_it.json',
    'sst5/sst5_results_train_falcon3_7b_it.json',
    'sst5/sst5_results_train_mistral_small_24b.json',
]

SHIFT_TYPES = ['minority_target']

# RSE (Robust Satisficing Ensemble)
TAU_TARGET_FRACTION = 0.70
RSE_MAX_ITER = 2000
RSE_EPSILON = 1e-5
RSE_ETA = 0.2 

# KL-RS
KLRS_TAU_FACTOR = 1.1
KLRS_LAMBDA_MIN = 1e-5
KLRS_LAMBDA_MAX = 1000.0

# IRM
IRM_LAMBDA = 0.4
IRM_LR = 0.01

KL_GRID_POINTS = 11


RC_PARAMS ={
    # 1. Use the system's LaTeX to render text (results in Type 1 fonts)
    'text.usetex': True,
    
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
}

PLOT_STYLES = {
    'ERM': {'color': "#45B5E9", 'marker': 'o', 'linestyle': '-', 'ms': 12, 'linewidth': 4},
    'RSE': {'color': '#2ecc71', 'marker': 's', 'linestyle': '--', 'ms': 7, 'linewidth': 2},
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

def setup_plotting():
    """Helper to apply styles globally"""
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC_PARAMS)
    sns.set_palette("husl")



if __name__ == "__main__":

    setup_plotting()


    all_experiment_results = []
    all_experiment_summaries = []
    all_experiment_training_times = []
    first_exp_logs_offline = None

   
    for exp_num in range(N_EXPERIMENTS):
        print("\n" + "=" * 70 + f"\n=== EXPERIMENT {exp_num + 1} / {N_EXPERIMENTS} ===\n" + "=" * 70)
        current_exp_seed = BASE_SEED + exp_num *50
        training_times = {} 


        all_predictions, all_labels, M, K, N_total = load_predictions_from_files(TRAINING_FILES, SST5_LABEL_ORDER, gt_key=SST5_GT_KEY)

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
        
        print(f"Selected Alphas: {[f'{a:.3f}' for a in magnitudes_to_test]}")

        start_gd = time.perf_counter()
        w_erm_ce = solve_erm_cross_entropy(train_preds_actual, train_labels_actual, M)
        training_times['ERM_CE'] = time.perf_counter() - start_gd
        
        start_gd = time.perf_counter()
        # == Phase 1: Training ==
        A_matrices_offline, b_vectors_offline = calculate_batch_moments(train_preds_actual, train_labels_actual, M, K)
        finished_training_time = time.perf_counter() - start_gd
        print(f"Finished training in {finished_training_time:.4f}s")
        
        start_gd = time.perf_counter()
        w_erm_offline, w_robust_offline, logs_offline, time_robust = solve_robust_offline_with_proximal(
        A_matrices_offline, b_vectors_offline, q_s_offline, M, K, 
        TAU_TARGET_FRACTION, T_max_iter=2000, 
        epsilon_convergence=1e-5,
        eta=0.2
                )
        training_times['RSE'] =time.perf_counter() - start_gd
        print(logs_offline)
        
        
        start_gd = time.perf_counter()
        w_groupdro, _, _ = solve_group_dro_ensemble(train_preds_actual, train_labels_actual, M, K)
        training_times['GroupDRO'] = time.perf_counter() - start_gd

        
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
        print('the name of best model is:', TRAINING_FILES[best_model_idx])
        """ 
        w_phi4_15b = np.zeros(M)
        w_phi4_15b[1] = 1.0
        w_mistral_7b_it = np.zeros(M)
        w_mistral_7b_it[5] = 1.0
        w_gemma3_12b_it = np.zeros(M)
        w_gemma3_12b_it[4] = 1.0
        w_gpt_oss_20b = np.zeros(M)
        w_gpt_oss_20b[3] = 1.0
        w_falcon3_7b_it = np.zeros(M)
        w_falcon3_7b_it[6] = 1.0
        w_mistral_small_24b_it = np.zeros(M)
        w_mistral_small_24b_it[7] = 1.0
        w_qwen3_4b = np.zeros(M)
        w_qwen3_4b[0] = 1.0
        w_phi3_5_mini_it = np.zeros(M)
        w_phi3_5_mini_it[2] = 1.0
        """

        weights_to_test = {
            'RSE': w_robust_offline,
            #'phi4': w_phi4_15b,
            #'Mistral 7b': w_mistral_7b_it,
            #'Gemma3 12b': w_gemma3_12b_it,
            #'Gpt Oss 20b': w_gpt_oss_20b,
            #'Falcon3 7b': w_falcon3_7b_it,
            #'Mistral Small 24b': w_mistral_small_24b_it,
            #'Qwen3 4b': w_qwen3_4b,
            #'phi3.5 mini': w_phi3_5_mini_it,
            'DRO': w_groupdro, 'IRM': w_irm, 'SRM': w_srm, 'KL-RS': w_klrs, 'ERM': w_erm_ce,
            'Uniform': w_uniform, 'MajorityVote': None,'AccuracyWeighted': w_acc_weighted 
            
        }

       
        for m in range(M):
            w_single = np.zeros(M)
            w_single[m] = 1.0
            model_short_name = TRAINING_FILES[m].split('/')[-1].replace('sst5_results_train_', '').replace('.json', '')
            weights_to_test[f'Individual_{model_short_name}'] = w_single

        
        if exp_num == 0: mean_vectors_by_mag = defaultdict(list)
        results = defaultdict(lambda: defaultdict(list))
        q_test_pool = np.bincount(test_pool_labels, minlength=K) / len(test_pool_labels)
        test_count = 0

        for shift_type in SHIFT_TYPES:
            for magnitude in magnitudes_to_test:
                for trial in range(N_TRIALS_PER_SHIFT):
                    test_count += 1
                    target_dist = generate_shift_distributions(q_test_pool, shift_type, magnitude, K, hardest_class_index)
                    shifted_preds, shifted_labels = create_label_shifted_dataset(
                        calibrated_test_pool_preds, test_pool_labels, target_dist, GENERATED_TEST_SET_SIZE, seed=current_exp_seed+test_count)
                    
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
            
            
            q_test_by_magnitude = {0.0: q_s_offline.copy()}  # α=0 = train distribution
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
    with open('AVERAGED_final_sst5_OVERALL_summary.json', 'w') as f: json.dump(final_summary_json, f, indent=4)

    
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

    with open('AVERAGED_per_magnitude_detailed_results_sst5.json', 'w') as f:
        json.dump(per_magnitude_results, f, indent=4)
    print("✓ Saved detailed per-magnitude averages to 'AVERAGED_per_magnitude_detailed_results_sst5.json'")
    
    
    if first_exp_logs_offline:
        
        fig_k_off, ax_k_off = plt.subplots(figsize=(3.25, 2.5), layout='constrained')
        kl = first_exp_logs_offline['log_kappa'][1:]
        if kl: ax_k_off.plot(range(len(kl)), kl, lw=1.5, color='#2980b9')
        ax_k_off.set_xlabel('Iterations'); ax_k_off.set_ylabel(r'$\kappa$')
        fig_k_off.savefig('sst5_kappa_convergence_plot_offline_exp1.pdf', format='pdf')
        plt.show()
    
   


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

    
    num_shift_types = len(SHIFT_TYPES)
    
    fig_width = 3.5 if num_shift_types == 1 else 7.0 
    fig_height = 2.8 

    fig_acc, axes_acc = plt.subplots(1, num_shift_types, figsize=(fig_width, fig_height), squeeze=False)
    
    
    for col, shift_type in enumerate(SHIFT_TYPES):
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
            
            
            style = PLOT_STYLES.get(name, {'color': 'gray', 'marker': 'o', 'linestyle': '-', 'linewidth': 1})
            
        
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
                          fontsize=6, 
                          frameon=True, 
                          framealpha=0.7, 
                          edgecolor='white',
                          ncol=2) 

    plt.tight_layout()
    fig_acc.savefig('AVERAGED_final_sst5_shift_accuracy_plot.pdf', format='pdf', dpi=300)
    print("✓ Saved ICML-style accuracy plot")
    plt.show()

    import pandas as pd
    from scipy.stats import ttest_rel

    print("\n" + "="*100)
    print("=== SIGNIFICANCE TESTING (Paired t-test vs RSE) ===")
    print("="*100)

   
    hero_method = 'RSE'
    
    baselines = [m for m in weights_to_test.keys() if m != hero_method and not m.startswith('Individual_')]

    shift_type = SHIFT_TYPES[0] 
    sig_results = []

    for magnitude in magnitudes_to_test:
       
        hero_accs = []
        baseline_accs_dict = {b: [] for b in baselines}

        for exp_res in all_experiment_results:
           
            hero_trials = exp_res[hero_method][(shift_type, magnitude)]
            hero_accs.extend([t['accuracy'] for t in hero_trials])

            
            for b in baselines:
                b_trials = exp_res[b][(shift_type, magnitude)]
                baseline_accs_dict[b].extend([t['accuracy'] for t in b_trials])

        hero_accs = np.array(hero_accs)
        hero_mean = np.mean(hero_accs) * 100
        hero_std = np.std(hero_accs) * 100

        
        row_data = {
            'Shift': f"{magnitude:.2f}",
            f'{hero_method} (Ours)': f"{hero_mean:.1f} ± {hero_std:.1f}"
        }

        for b in baselines:
            b_accs = np.array(baseline_accs_dict[b])
            b_mean = np.mean(b_accs) * 100
            b_std = np.std(b_accs) * 100

            
            if np.all(hero_accs == b_accs):
                p_val = 1.0 
            else:
                _, p_val = ttest_rel(hero_accs, b_accs)

            
            if p_val < 0.01:
                sig_marker = "**" 
            elif p_val < 0.05:
                sig_marker = "*"  
            else:
                sig_marker = ""    

            row_data[b] = f"{b_mean:.1f} ± {b_std:.1f}{sig_marker}"

        sig_results.append(row_data)

    
    df_sig = pd.DataFrame(sig_results)
    
   
    cols = ['Shift', f'{hero_method} (Ours)'] + baselines
    df_sig = df_sig[cols]

    print("\nSignificance Table (Accuracies % ± Std Dev, * p<0.05, ** p<0.01 vs RSE):")
    print(df_sig.to_string(index=False))
    print("="*100 + "\n")
