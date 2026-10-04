import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode # Added for Majority Vote

from groupdro_ensemble import solve_group_dro_ensemble
from erm import solve_erm_cross_entropy
from rse import solve_robust_offline_with_proximal, calculate_batch_moments
from srm_ensemble import solve_srm_ensemble 
from kl_rs import solve_kl_rs_ensemble      
from irm import solve_irm_ensemble


TRAINING_FILES = [
    "selective_copying/results_ultrahard_Falcon3-7B-Instruct.json",
    "selective_copying/results_ultrahard_gpt-oss-20b.json",
    "selective_copying/results_ultrahard_Mistral-7B-Instruct-v0.3.json",
    "selective_copying/results_ultrahard_Mistral-Small-24B-Instruct-2501.json",
    "selective_copying/results_ultrahard_phi-4.json",
    "selective_copying/results_ultrahard_Qwen3-4B.json",
    "selective_copying/results_ultrahard_Llama-3.1-8B-Instruct.json",
    "selective_copying/results_ultrahard_Phi-3.5-mini-instruct.json",
]

# Dynamically extract model names from the file paths for reporting
MODEL_NAMES = [f.split('_')[-1].replace('.json', '') for f in TRAINING_FILES]

M = len(TRAINING_FILES)
K = 3 # A, B, C

# KL-RS
KLRS_TAU_FACTOR = 1.1
KLRS_LAMBDA_MIN = 1e-5
KLRS_LAMBDA_MAX = 1000.0
BASE_SEED = 42
# IRM
IRM_LAMBDA = 0.4
IRM_LR = 0.01

# --- RSE CONFIGURATION ---
TAU_TARGET_FRACTION = 0.95
RSE_MAX_ITER = 2000
RSE_EPSILON = 1e-5
RSE_ETA = 0.2

def load_selective_copying_data(files):
    """Loads the Easy (Train) and Hard (Test) sequence shift data."""
    preds_e_list, preds_h_list = [], []
    labels_e, labels_h = None, None
    
    for f in files:
        with open(f, 'r') as j:
            data = json.load(j)
            preds_e_list.append([it['probs'] for it in data['easy']])
            preds_h_list.append([it['probs'] for it in data['hard']])
            if labels_e is None:
                labels_e = np.array([it['label'] for it in data['easy']])
                labels_h = np.array([it['label'] for it in data['hard']])
                
    return np.array(preds_e_list), np.array(preds_h_list), labels_e, labels_h

if __name__ == "__main__":
    print("=" * 70)
    print("=== SELECTIVE COPYING (INPUT SHIFT) EXPERIMENT ===")
    print("=" * 70)

    # 1. Load Data
    print("Loading Easy (Train) and Hard (Shifted Test) predictions...")
    preds_easy, preds_hard, labels_easy, labels_hard = load_selective_copying_data(TRAINING_FILES)
    
    print((f"Loaded {preds_easy.shape[1]} Easy samples and {preds_hard.shape[1]} Hard samples."))
    
    # 2. Calibration / Training Split
    N_EASY_CALIB, N_HARD_CALIB = 500, 0 
    
    train_preds = np.concatenate([preds_easy[:, :N_EASY_CALIB, :], preds_hard[:, :N_HARD_CALIB, :]], axis=1)
    train_labels = np.concatenate([labels_easy[:N_EASY_CALIB], labels_hard[:N_HARD_CALIB]])
    print(f"Calibration set: {train_preds.shape[1]} samples (Easy: {N_EASY_CALIB}, Hard: {N_HARD_CALIB})")
    
    test_preds = preds_hard[:, N_HARD_CALIB:, :]
    test_labels = labels_hard[N_HARD_CALIB:]
    print(f"Shifted Test set: {test_preds.shape[1]} samples (Hard sequences)")
    
    q_s_offline = np.bincount(train_labels, minlength=K) / len(train_labels)

    # --- INDIVIDUAL MODEL PERFORMANCE & ACCURACY WEIGHTS ---
    print("\n[INDIVIDUAL MODEL PERFORMANCE]")
    print(f"{'Model Name':>30} | {'Train (Easy) Acc':>16} | {'Test (Hard) Acc':>15}")
    print("-" * 68)
    
    train_accuracies = [] # Store for Accuracy-Weighted Baseline
    
    for m in range(M):
        # Calculate Train Accuracy (Calibration Set)
        train_acc = np.mean(np.argmax(train_preds[m], axis=1) == train_labels)
        train_accuracies.append(train_acc)
        
        # Calculate Test Accuracy (Shifted Evaluation Set)
        test_acc = np.mean(np.argmax(test_preds[m], axis=1) == test_labels)
        
        # Print individual model names and performance
        print(f"{MODEL_NAMES[m]:>30} | {train_acc*100:>15.2f}% | {test_acc*100:>14.2f}%")
        
    print("-" * 68)

    # --- CALCULATE NON-OPTIMIZATION BASELINES ---
    # 1. Uniform Weights
    w_uniform = np.full(M, 1.0 / M)
    
    # 2. Accuracy-Weighted
    w_acc_weighted = np.array(train_accuracies) / np.sum(train_accuracies)
    
    # 3. Best Individual
    best_individual_idx = np.argmax([np.mean(np.argmax(test_preds[m], axis=1) == test_labels) for m in range(M)])
    w_best_individual = np.zeros(M)
    w_best_individual[best_individual_idx] = 1.0

    print("\n[TRAINING ENSEMBLES]")
    
    print("Training ERM...")
    w_erm_ce = solve_erm_cross_entropy(train_preds, train_labels, M)

    print("Training DRO...")
    w_groupdro, _, _ = solve_group_dro_ensemble(train_preds, train_labels, M, K)

    print("Training RSE...")
    A_matrices_offline, b_vectors_offline = calculate_batch_moments(train_preds, train_labels, M, K)
    _, w_rse, logs_offline, _ = solve_robust_offline_with_proximal(
        A_matrices_offline, b_vectors_offline, q_s_offline, M, K, 
        TAU_TARGET_FRACTION, T_max_iter=RSE_MAX_ITER, 
        epsilon_convergence=RSE_EPSILON, eta=RSE_ETA
    )

    print("Training SRM...")
    w_srm, _, _ = solve_srm_ensemble(train_preds, train_labels, M, K)
    
    print("Training KL-RS...")
    w_klrs, _ = solve_kl_rs_ensemble(train_preds, train_labels, M, tau_factor=KLRS_TAU_FACTOR)
    
    print("Training IRM...")
    w_irm, _ = solve_irm_ensemble(train_preds, train_labels, q_s_offline, M, lambda_irm=IRM_LAMBDA, lr=IRM_LR)
    
    # --- CONSOLIDATE ALL WEIGHTS ---
    weights_to_test = {
        'Uniform': w_uniform,
        'Acc-Weighted': w_acc_weighted,
        'MajorityVote': 'majority', # String flag to trigger hard voting logic
        'ERM': w_erm_ce,
        'DRO': w_groupdro, 
        'RSE': w_rse,
        'SRM': w_srm,
        'KL-RS': w_klrs,
        'IRM': w_irm,
        'BestIndiv': w_best_individual,
    }

    # 3. Evaluation on the Shifted Set (Hard Sequences)
    print("\n[EVALUATING ON SHIFTED TEST SET (Sequence Length 50)]")
    results = {}
    
    def evaluate_weights(w_vec, preds, labels):
        w = w_vec / w_vec.sum() if w_vec.sum() > 0 else np.full(M, 1.0/M)
        ens_probs = np.tensordot(w, preds, axes=(0,0))
        return np.mean(np.argmax(ens_probs, axis=1) == labels) * 100

    for name, w_vec in weights_to_test.items():
        if isinstance(w_vec, str) and w_vec == 'majority':
            # Majority Vote Logic: Convert probs to hard predictions, then take the mode
            hard_preds = np.argmax(test_preds, axis=2) # Shape: (M, N)
            maj_preds, _ = mode(hard_preds, axis=0, keepdims=False)
            acc = np.mean(maj_preds == test_labels) * 100
        else:
            acc = evaluate_weights(w_vec, test_preds, test_labels)
            
        results[name] = acc
        print(f"{name:>15}: {acc:.2f}% Accuracy")

    # 4. Plotting
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(10, 5)) # Made wider to fit more bars
    
    colors = {
        'Uniform': '#95a5a6', 'Acc-Weighted': '#7f8c8d', 'MajorityVote': '#bdc3c7',
        'ERM': "#45B5E9", 'DRO': '#e74c3c', 'RSE': '#2ecc71', 
        'SRM': '#9b59b6', 'KL-RS': '#f39c12', 'IRM': '#34495e',
        'BestIndiv': '#d35400'
    }
    
    methods = list(results.keys())
    accuracies = list(results.values())

    bars = ax.bar(methods, accuracies, color=[colors.get(m, '#000000') for m in methods], width=0.6)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax.set_ylabel('Test Accuracy (%) on Long Sequences')
    ax.set_title('Robustness Against Sequence Length Shift')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Dynamically find and highlight RSE
    if 'RSE' in methods:
        rse_idx = methods.index('RSE')
        bars[rse_idx].set_edgecolor('black')
        bars[rse_idx].set_linewidth(2)

    # Rotate x-labels so they don't overlap
    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    fig.savefig('rebuttal_8llm_input_shift.pdf', format='pdf', dpi=300)
    print("\n✓ Saved bar chart to 'rebuttal_8llm_input_shift.pdf'")
    plt.show()

    # 5. Print out the weights to see who RSE picked
    print("\n--- RSE WEIGHT DISTRIBUTION ---")
    for name, weight in zip(MODEL_NAMES, w_rse):
        if weight > 0.01:
            print(f"{name}: {weight*100:.1f}%")

    # =========================================================================
    # === SIGNIFICANCE TESTING (Bootstrap on Selective Copying Test Set) ===
    # =========================================================================
    print("\n" + "="*100)
    print("=== SIGNIFICANCE TESTING (Bootstrap on Selective Copying Test Set) ===")
    print("="*100)

    B = 1000  
    N_test = len(test_labels)
    hero_method = 'RSE'
    
    
    baselines = [m for m in weights_to_test.keys() if m != hero_method]

    
    method_correctness = {}
    
    for name, w_vec in weights_to_test.items():
        if isinstance(w_vec, str) and w_vec == 'majority':
            
            hard_preds = np.argmax(test_preds, axis=2)
            maj_preds, _ = mode(hard_preds, axis=0, keepdims=False)
            is_correct = (maj_preds == test_labels).astype(int)
        else:
            
            w = w_vec / w_vec.sum() if w_vec.sum() > 0 else np.full(M, 1.0/M)
            ens_probs = np.tensordot(w, test_preds, axes=(0,0))
            pred_labels = np.argmax(ens_probs, axis=1)
            is_correct = (pred_labels == test_labels).astype(int)
            
        method_correctness[name] = is_correct

    
    boot_accs = {m: np.zeros(B) for m in [hero_method] + baselines}
    
    np.random.seed(BASE_SEED) 
    for i in range(B):
        
        indices = np.random.choice(N_test, size=N_test, replace=True)

        for m in [hero_method] + baselines:
            
            boot_accs[m][i] = np.mean(method_correctness[m][indices]) * 100

    
    print(f"\n--- Overall Accuracy Significance Table ---")
    print(f"(Scores % ± Std Dev over {B} bootstrap samples, * p<0.05, ** p<0.01 vs RSE)\n")
    
    hero_scores = boot_accs[hero_method]
    hero_mean = np.mean(hero_scores)
    hero_std = np.std(hero_scores)
    
    header = f"{'Method':<16} | {'Score ± Std':<15} | {'p-value':<10} | {'Sig.'}"
    print(header)
    print("-" * len(header))
    
    print(f"{hero_method+' (Ours)':<16} | {hero_mean:.2f} ± {hero_std:.2f} | {'-':<10} | {'-'}")
    
    
    priority_baselines = ['DRO', 'IRM', 'SRM', 'KL-RS', 'ERM', 'BestIndiv', 'Acc-Weighted', 'Uniform', 'MajorityVote']
    
    for b in priority_baselines:
        if b not in boot_accs: continue
        
        b_scores = boot_accs[b]
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
            
        print(f"{b:<16} | {b_mean:.2f} ± {b_std:.2f} | {p_val:.4f}     | {sig_marker}")
        
    print("\n" + "="*100)