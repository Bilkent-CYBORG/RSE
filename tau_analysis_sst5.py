import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from groupdro_ensemble import solve_group_dro_ensemble
from erm import solve_erm_cross_entropy
from irm import solve_irm_ensemble
from rse import solve_robust_offline_with_proximal, calculate_batch_moments

from helpers import (
    load_predictions_from_files_sst5 as load_predictions_from_files, 
    split_train_test_pool, 
    evaluate_on_shifted_data,
    generate_shift_distributions,
    create_label_shifted_dataset,
)


if __name__ == "__main__":

    plt.rcParams.update({
    
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

    
    N_EXPERIMENTS = 20
    BASE_SEED = 42
    TAU_FRACTIONS_TO_TEST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.60, 0.65, 0.7, 0.75]
    
    
    TEST_POOL_FRACTION = 0.1
    VALIDATION_FRACTION = 0.1

    SST5_LABEL_ORDER = ["very negative", "negative", "neutral", "positive", "very positive"]
    SST5_GT_KEY = "ground_truth_label"

    
    training_files = [
        'sst5/sst5_results_train_qwen3_4b.json',
        'sst5/sst5_results_train_phi_4_15b.json',
        'sst5/sst5_results_train_gpt_oss_20b.json',
        'sst5/sst5_results_train_gemma3_12b.json',
        'sst5/sst5_results_train_mistral_7b_it.json',
        'sst5/sst5_results_train_falcon3_7b_it.json',
        'sst5/sst5_results_train_mistral_small_24b.json',
        'sst5/sst5_results_train_phi3.5_mini_it.json'
        
        
        
    ]

   
    plot_styles = {
        'ERM': {'color': '#45B5E9', 'linestyle': '-'},
        'DRO': {'color': '#e74c3c', 'linestyle': ':'},
        'IRM': {'color': '#f39c12', 'linestyle': '-.'},
        'RSE': {'color': '#0e4c40', 'marker': 'X', 'linestyle': '--'},
    }

    print("\n" + "=" * 70)
    print(f"=== STARTING TAU SENSITIVITY ANALYSIS ===")
    print(f"N_EXPERIMENTS: {N_EXPERIMENTS}, BASE_SEED: {BASE_SEED}")
    print("=" * 70)

   
    all_predictions, all_labels, M, K, N_total = load_predictions_from_files(training_files, SST5_LABEL_ORDER, gt_key=SST5_GT_KEY)
    
    aggregated_min_accuracies = defaultdict(list)
    tau_fraction_map = defaultdict(list)

   
    for exp_num in range(N_EXPERIMENTS):
        print(f"\n>>> EXPERIMENT {exp_num + 1} / {N_EXPERIMENTS} <<<")
        current_exp_seed = BASE_SEED + exp_num * 50
        np.random.seed(current_exp_seed)

        
        train_pool_preds, train_pool_labels, test_pool_preds, test_pool_labels = \
            split_train_test_pool(all_predictions, all_labels, K, TEST_POOL_FRACTION, seed=current_exp_seed)
        
        actual_train_preds, actual_train_labels, val_preds, val_labels = \
            split_train_test_pool(train_pool_preds, train_pool_labels, K, VALIDATION_FRACTION, seed=current_exp_seed)

        
        train_preds_actual = train_pool_preds
        train_labels_actual = train_pool_labels
        calibrated_test_pool_preds = test_pool_preds

        
        A_matrices, b_vectors = calculate_batch_moments(train_preds_actual, train_labels_actual, M, K)
        q_s_offline = np.bincount(train_labels_actual, minlength=K) / len(train_labels_actual)

        
        class_accuracies = np.zeros((M, K))
        for m in range(M):
            for k in range(K):
                idx = np.where(val_labels == k)[0]
                if len(idx) > 0:
                    class_accuracies[m, k] = np.mean(np.argmax(val_preds[m, idx], axis=1) == val_labels[idx])
        hardest_class_index = np.nanargmin(np.nanmean(class_accuracies, axis=0))

       
        weights_map = {}
        
        # ERM
        w_erm = solve_erm_cross_entropy(train_preds_actual, train_labels_actual, M)
        weights_map['ERM'] = w_erm
        
        # DRO
        w_dro, _, _ = solve_group_dro_ensemble(train_preds_actual, train_labels_actual, M, K)
        weights_map['DRO'] = w_dro
        
        # IRM
        w_irm, _ = solve_irm_ensemble(train_preds_actual, train_labels_actual, q_s_offline, M, lambda_irm=0.4, lr=0.01)
        weights_map['IRM'] = w_irm

        
        for frac in TAU_FRACTIONS_TO_TEST:
            
            _, w_robust_off, logs_off, _ = solve_robust_offline_with_proximal(
                A_matrices, b_vectors, q_s_offline, M, K, 
                frac, 2000, 1e-5, 0.2
            )
            
            
            actual_tau = logs_off['final_tau']
            tau_fraction_map[frac].append(actual_tau)

            weights_map[f"RSE(lambda={frac:.1f})"] = w_robust_off
           

       
        shift_magnitudes = [1.0]
        n_trials_per_shift = 5
        generated_test_set_size = 1000
        
        seed_shift_accuracies = defaultdict(list)

        for mag in shift_magnitudes:
            for trial in range(n_trials_per_shift):
                target_dist = generate_shift_distributions(q_s_offline, 'minority_target', mag, K, hardest_class_index)
                shifted_preds, shifted_labels = create_label_shifted_dataset(
                    calibrated_test_pool_preds, test_pool_labels, target_dist, 
                    generated_test_set_size, seed=current_exp_seed + int(mag*100) + trial
                )
                
                if shifted_preds is None: continue

                for name, w in weights_map.items():
                    res = evaluate_on_shifted_data(w, shifted_preds, shifted_labels)
                    seed_shift_accuracies[name].append(res['accuracy'])

        
        for name, acc_list in seed_shift_accuracies.items():
            if acc_list:
                mean_acc = np.mean(acc_list)
                aggregated_min_accuracies[name].append(mean_acc)

    
    print("\n=== Generating Tau Sensitivity Plot ===")
    
    avg_tau_values = []
    for frac in TAU_FRACTIONS_TO_TEST:
        avg_tau_values.append(np.mean(tau_fraction_map[frac]))

    plt.figure(figsize=(3.25, 2.5))

    
    rse_off_means = []
    for frac in TAU_FRACTIONS_TO_TEST:
        name = f"RSE(lambda={frac:.1f})"
        if name in aggregated_min_accuracies:
            rse_off_means.append(np.mean(aggregated_min_accuracies[name]))
        else:
            rse_off_means.append(None)

    plt.plot(avg_tau_values, rse_off_means, label='RSE', 
             color=plot_styles['RSE']['color'], marker=plot_styles['RSE']['marker'], 
             linestyle=plot_styles['RSE']['linestyle'], linewidth=1.5, markersize=4)

   
    
    baselines = ['ERM', 'DRO', 'IRM']
    for base in baselines:
        if base in aggregated_min_accuracies:
            mean_val = np.mean(aggregated_min_accuracies[base])
            style = plot_styles.get(base, {'color': 'black', 'linestyle': ':'})
            plt.axhline(mean_val, label=base, color=style['color'], 
                       linestyle=style['linestyle'], linewidth=1.5, alpha=0.8)

    plt.xlabel(r"$\tau$")
    plt.ylabel("Avg. Accuracy")
    
    plt.legend(
        loc='upper left', 
        fontsize=6,
        labelspacing=0.25,
        handlelength=1.2,
        handletextpad=0.4,
        borderaxespad=0.4,
        frameon=True, 
        framealpha=0.8
    )
    
    out_path = 'tau1.pdf'
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"✓ Saved plot to {out_path}")
    plt.show()

    
    results_json = {
        'tau_fractions': TAU_FRACTIONS_TO_TEST,
        'avg_tau_values': avg_tau_values,
        'rse_offline_means': rse_off_means,
        'erm_mean': float(np.mean(aggregated_min_accuracies.get('ERM', [0]))),
        'dro_mean': float(np.mean(aggregated_min_accuracies.get('DRO', [0]))),
        'irm_mean': float(np.mean(aggregated_min_accuracies.get('IRM', [0]))),
    }
    with open('sst5_tau_sensitivity_results.json', 'w') as f:
        json.dump(results_json, f, indent=4)
    print(f"✓ Saved results to 'sst5_tau_sensitivity_results.json'")