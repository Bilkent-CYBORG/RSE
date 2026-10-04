import numpy as np
import json
import sys
import matplotlib.pyplot as plt
import os
from scipy.optimize import brentq
import numpy as np

def calculate_kl_divergence(p, q):
    p = np.asarray(p); q = np.asarray(q)
    p = np.where(p == 0, 1e-10, p); q = np.where(q == 0, 1e-10, q)
    return np.sum(p * (np.log(p) - np.log(q)))


def generate_shift_distributions(source_dist, shift_type, magnitude, K, target_class_index=None):
    source_dist = np.array(source_dist)

    if shift_type == 'minority_target':
        idx = K-1 if target_class_index is None else target_class_index
        hardest = np.zeros(K); hardest[idx] = 1.0
        target_dist = (1 - magnitude) * source_dist + magnitude * hardest

    else: target_dist = source_dist
    return target_dist / target_dist.sum()

def create_label_shifted_dataset(pool_predictions, pool_labels, target_distribution, target_set_size, seed=42):
    np.random.seed(seed)
    M, N_pool, K = pool_predictions.shape
    target_distribution = np.array(target_distribution) / np.sum(target_distribution)
    target_counts = np.round(target_distribution * target_set_size).astype(int)
    while target_set_size - target_counts.sum() != 0:
        target_counts[np.argmax(target_counts)] += np.sign(target_set_size - target_counts.sum())
    sampled_indices = []
    for j in range(K):
        class_j = np.where(pool_labels == j)[0]
        if len(class_j) == 0 or target_counts[j] <= 0: continue
        sampled_indices.extend(np.random.choice(class_j, size=target_counts[j], replace=(target_counts[j] > len(class_j))))
    if not sampled_indices: return None, None
    sampled_indices = np.array(sampled_indices); np.random.shuffle(sampled_indices)
    return pool_predictions[:, sampled_indices, :], pool_labels[sampled_indices]

def evaluate_on_shifted_data(w, predictions, labels):
    if len(labels) == 0: return {'accuracy': 0.0, 'risk': np.inf}
    ensemble_probs = np.einsum('m,mnk->nk', w, predictions)
    predicted_labels = np.argmax(ensemble_probs, axis=1)
    accuracy = np.mean(predicted_labels == labels)
    one_hot = np.zeros((len(labels), predictions.shape[2])); one_hot[np.arange(len(labels)), labels] = 1.0
    risk = np.mean(np.sum((one_hot - ensemble_probs) ** 2, axis=1))
    return {'accuracy': accuracy, 'risk': risk}

def split_train_test_pool(predictions, labels, K, test_pool_fraction=0.2, seed=42):
    np.random.seed(seed)
    M, N, _ = predictions.shape
    train_indices = []; test_pool_indices = []
    print(f"\nSplitting data: {1-test_pool_fraction:.0%} train pool, {test_pool_fraction:.0%} test pool...")
    for j in range(K):
        class_j_indices = np.where(labels == j)[0]
        n_j = len(class_j_indices)
        if n_j == 0: continue
        n_pool_j = int(n_j * test_pool_fraction)
        n_pool_j = max(1, n_pool_j) if n_j > 0 and n_pool_j == 0 else n_pool_j
        if n_j - n_pool_j <= 0: train_indices.extend(class_j_indices); continue
        shuffled = np.random.permutation(class_j_indices)
        test_pool_indices.extend(shuffled[:n_pool_j])
        train_indices.extend(shuffled[n_pool_j:])
    return predictions[:, train_indices, :], labels[train_indices], predictions[:, test_pool_indices, :], labels[test_pool_indices]

def evaluate_majority_vote(predictions, labels):
    if len(labels) == 0: return {'accuracy': 0.0, 'risk': np.inf}
    model_votes = np.argmax(predictions, axis=2).T
    predicted_labels = np.array([np.argmax(np.bincount(votes, minlength=predictions.shape[2])) for votes in model_votes])
    accuracy = np.mean(predicted_labels == labels)
    return {'accuracy': accuracy, 'risk': 2.0 * (1.0 - accuracy)}


def get_alphas_for_equidistant_kl(q_source, shift_type, K, target_class_index, num_points=11):
    
    # 1. Get the extreme target distribution (alpha = 1.0)
    
    q_target_extreme = generate_shift_distributions(
        q_source, shift_type, 1.0, K, target_class_index
    )
    
    # 2. Calculate Maximum possible KL (at alpha=1.0)
    max_kl = calculate_kl_divergence(q_target_extreme, q_source)
    print(f"Max possible KL for {shift_type}: {max_kl:.4f}")

    # 3. Define target KLs (Equally spaced on the X-axis)
    # We cap it slightly below max_kl to avoid numerical instability at the simplex boundary
    target_kls = np.linspace(0, max_kl * 0.99, num_points)
    
    found_alphas = [0.0] # Always include 0.0

    # 4. Solve for alpha for each target KL
    for target_kl in target_kls[1:]:
        def objective(a):
            # Mix distributions
            q_mix = (1 - a) * q_source + a * q_target_extreme
            # Calculate KL
            current_kl = calculate_kl_divergence(q_mix, q_source)
            return current_kl - target_kl

        # Find alpha in range [0, 1] that satisfies KL(alpha) = target_kl
        try:
            # brentq is a robust root-finding method
            alpha_sol = brentq(objective, 0.0, 1.0)
            found_alphas.append(alpha_sol)
        except ValueError:
            # Fallback if solver fails (usually if max_kl calculation was slightly off)
            found_alphas.append(1.0)

    # Ensure 1.0 is included if we want the extreme case
    if found_alphas[-1] < 0.99:
        found_alphas.append(1.0)
        
    return sorted(list(set(found_alphas))) 

def project_to_simplex(v, z=1.0):
    v = np.asarray(v)
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    if not np.any(cond): return np.full(n_features, 1.0 / n_features)
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = np.maximum(v - theta, 0)
    return w

def load_predictions_from_files_sst5(file_list, ordered_label_list, gt_key='ground_truth'):
    all_model_predictions = []
    ground_truth_labels = []
    K = len(ordered_label_list)
    label_map = {label: i for i, label in enumerate(ordered_label_list)}
    M = len(file_list)

    print(f"Loading data from {M} model files for {K} classes...")
    if not file_list: sys.exit("FATAL ERROR: No model files provided.")
    try:
        with open(file_list[0], 'r') as f: data = json.load(f)
    except Exception as e: sys.exit(f"FATAL ERROR reading {file_list[0]}: {e}")
    
    N = len(data)
    model_probs = []
    for record in data:
        gt_label_str = record.get(gt_key)
        if gt_label_str not in label_map: continue
        ground_truth_labels.append(label_map[gt_label_str])
        probs_dict = record['probabilities']
        prob_vector = np.array([probs_dict.get(label, 1.0/K) for label in ordered_label_list])
        model_probs.append(prob_vector)
    
    N = len(ground_truth_labels)
    all_model_predictions.append(np.array(model_probs))
    labels = np.array(ground_truth_labels)

    for m in range(1, M):
        try:
            with open(file_list[m], 'r') as f: data = json.load(f)
        except Exception as e: print(f"Skipping {file_list[m]}: {e}"); continue
        model_probs = []
        for i in range(N): 
            if i >= len(data): model_probs.append(np.full(K, 1.0 / K)); continue
            record = data[i]
            gt_label_str = record.get(gt_key)
            if gt_label_str not in label_map or label_map[gt_label_str] != labels[i]:
                 model_probs.append(np.full(K, 1.0 / K)); continue 
            probs_dict = record['probabilities']
            prob_vector = np.array([probs_dict.get(label, 1.0/K) for label in ordered_label_list])
            model_probs.append(prob_vector)
        all_model_predictions.append(np.array(model_probs))

    predictions = np.stack(all_model_predictions, axis=0)
    return predictions, labels, len(all_model_predictions), K, N

def plot_distribution_trajectory(q_train, q_test_by_magnitude, class_names, hardest_class_idx,
                                 save_path='distribution_trajectory.pdf'):
    
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    
    magnitudes = sorted(q_test_by_magnitude.keys())
    K = len(class_names)
    
    # Build matrix: rows = magnitudes, cols = classes
    dist_matrix = np.array([q_test_by_magnitude[m] for m in magnitudes])
    
    # Stacked area plot
    colors = plt.cm.Set2(np.linspace(0, 1, K))
    ax.stackplot(magnitudes, dist_matrix.T, labels=class_names, colors=colors, alpha=0.8)
    
    # Mark the hardest class
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
    
    ax.set_xlabel(r'Shift Magnitude ($\alpha$)')
    ax.set_ylabel('Class Proportion')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    
    ax.legend(loc='upper left', ncol=2, fontsize=6, framealpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"✓ Saved to '{save_path}'")
    plt.show()


def load_predictions_from_files_trec(file_list, ordered_label_list, gt_key='ground_truth'):
    all_model_predictions = []
    ground_truth_labels = []
    K = len(ordered_label_list)
    label_map = {label: i for i, label in enumerate(ordered_label_list)}
    M = len(file_list)

    print(f"Loading data from {M} model files for {K} classes...")
    if not file_list: sys.exit("FATAL ERROR: No model files provided.")
    try:
        with open(file_list[0], 'r') as f: data = json.load(f)
    except Exception as e: sys.exit(f"FATAL ERROR reading {file_list[0]}: {e}")
    
    N = len(data)
    model_probs = []
    for record in data:
        gt_label_str = record.get(gt_key)
        if gt_label_str not in label_map: continue
        ground_truth_labels.append(label_map[gt_label_str])
        probs_dict = record['probabilities']
        prob_vector = np.array([probs_dict.get(label, 1.0/K) for label in ordered_label_list])
        model_probs.append(prob_vector)
    
    N = len(ground_truth_labels)
    all_model_predictions.append(np.array(model_probs))
    labels = np.array(ground_truth_labels)

    for m in range(1, M):
        try:
            with open(file_list[m], 'r') as f: data = json.load(f)
        except Exception as e: print(f"Skipping {file_list[m]}: {e}"); continue
        model_probs = []
        for i in range(N): 
            if i >= len(data): model_probs.append(np.full(K, 1.0 / K)); continue
            record = data[i]
            gt_label_str = record.get(gt_key)
            if gt_label_str not in label_map or label_map[gt_label_str] != labels[i]:
                 model_probs.append(np.full(K, 1.0 / K)); continue 
            probs_dict = record['probabilities']
            prob_vector = np.array([probs_dict.get(label, 1.0/K) for label in ordered_label_list])
            model_probs.append(prob_vector)
        all_model_predictions.append(np.array(model_probs))

    predictions = np.stack(all_model_predictions, axis=0)
    return predictions, labels, len(all_model_predictions), K, N



def load_huffpost_temporal_split(file_list, split_year, start_year=None, test_end_year=None):
    
    if start_year is None: start_year = -1
    if test_end_year is None: test_end_year = 9999

    print(f"\n--- Loading Data & Splitting ---")
    print(f"Train Window: {start_year if start_year > 0 else 'Start'} to {split_year}")
    print(f"Test Window:  {split_year + 1} to {test_end_year if test_end_year < 9999 else 'End'}")
    
    if not file_list: sys.exit("No files provided.")
    
    try:
        with open(file_list[0], 'r') as f: reference_data = json.load(f)
    except Exception as e: sys.exit(f"Error reading {file_list[0]}: {e}")

    # Build Label Map
    unique_labels = sorted(list(set(item['ground_truth_label'] for item in reference_data)))
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    K = len(unique_labels)
    print(f"Detected {K} Classes: {unique_labels}")

    
    train_indices, test_indices = [], []
    ground_truth_labels, years = [], []

    for idx, item in enumerate(reference_data):
        yr = item['year']
        lbl = item['ground_truth_label']
        ground_truth_labels.append(label_map[lbl])
        years.append(yr)
        
        if yr <= split_year and yr >= start_year:
            train_indices.append(idx)
        elif yr > split_year and yr <= test_end_year:
            test_indices.append(idx)

    ignored_count = len(reference_data) - len(train_indices) - len(test_indices)
    print(f"Total Examples: {len(reference_data)}")
    print(f"Train Set: {len(train_indices)} | Test Set: {len(test_indices)} | Ignored: {ignored_count}")

    if len(train_indices) == 0 or len(test_indices) == 0:
        sys.exit("Error: One of the splits is empty. Adjust your Year Window.")

    
    M = len(file_list)
    N = len(reference_data)
    all_preds = np.zeros((M, N, K))

    for m, filepath in enumerate(file_list):
        print(f"Loading model {m+1}/{M}: {os.path.basename(filepath)}")
        with open(filepath, 'r') as f: data = json.load(f)
        
        for i, item in enumerate(data):
            if i >= N: break
            probs = item['probabilities']
            vec = np.zeros(K)
            for l_str, l_idx in label_map.items(): vec[l_idx] = probs.get(l_str, 0.0)
            if vec.sum() > 0: vec /= vec.sum()
            else: vec = np.full(K, 1.0/K)
            all_preds[m, i, :] = vec

    # Slice
    train_preds = all_preds[:, train_indices, :]
    train_labels = np.array(ground_truth_labels)[train_indices]
    train_years = np.array(years)[train_indices]

    test_preds = all_preds[:, test_indices, :]
    test_labels = np.array(ground_truth_labels)[test_indices]
    test_years = np.array(years)[test_indices]

    return train_preds, train_labels, train_years, test_preds, test_labels, test_years, M, K, unique_labels


def calculate_metrics_with_groups(w, predictions, labels, years, K):
    
    ensemble_probs = np.einsum('m,mnk->nk', w, predictions)
    predicted_labels = np.argmax(ensemble_probs, axis=1)
    
    
    acc = np.mean(predicted_labels == labels)
    
    
    one_hot = np.zeros((len(labels), K))
    one_hot[np.arange(len(labels)), labels] = 1.0
    risk = np.mean(np.sum((one_hot - ensemble_probs) ** 2, axis=1))
    
    
    group_accuracies = {}
    for g in range(K):
        idx = np.where(labels == g)[0]
        if len(idx) > 0:
            group_accuracies[g] = np.mean(predicted_labels[idx] == labels[idx])
        else:
            group_accuracies[g] = np.nan
            
    valid_group_accs = [v for v in group_accuracies.values() if not np.isnan(v)]
    worst_group_acc = min(valid_group_accs) if valid_group_accs else 0.0
    
    
    unique_years = sorted(np.unique(years))
    year_accuracies = {}
    for y in unique_years:
        idx = np.where(years == y)[0]
        if len(idx) > 0:
            year_accuracies[y] = np.mean(predicted_labels[idx] == labels[idx])
            
    return acc, risk, worst_group_acc, group_accuracies, None, year_accuracies


def load_civil_comments_data(train_files, test_files, label_map):
    def get_model_name(f):
        base = os.path.basename(f)
        name = os.path.splitext(base)[0]
        for p in ['civilcomments_train_', 'civilcomments_test_', 'civilcomments_val_']:
            name = name.replace(p, '')
        return name

    train_map = {get_model_name(f): f for f in train_files}
    test_map = {get_model_name(f): f for f in test_files}
    common_models = sorted(list(set(train_map.keys()) & set(test_map.keys())))
    M = len(common_models)
    K = len(label_map)
    print(f"Found {M} models: {common_models}")
    
    if M == 0: sys.exit("FATAL ERROR: No matching train/test pairs found.")

    def load_set(file_list):
        try:
            with open(file_list[0], 'r') as f: data_blob = json.load(f)
        except Exception as e: sys.exit(f"Error reading {file_list[0]}: {e}")
        
        items = data_blob['results'] if (isinstance(data_blob, dict) and 'results' in data_blob) else data_blob
        gt_labels = []; groups = []; valid_indices = []
        
        for i, item in enumerate(items):
            lbl = item.get('ground_truth_label')
            if lbl in label_map:
                gt_labels.append(label_map[lbl])
                groups.append(item.get('group_name', 'unknown'))
                valid_indices.append(i)
        
        labels = np.array(gt_labels)
        group_array = np.array(groups)
        
        all_preds = []
        for path in file_list:
            try:
                with open(path, 'r') as f: 
                    d = json.load(f)
                    d_items = d['results'] if isinstance(d, dict) and 'results' in d else d
            except: 
                print(f"Warning: Failed to load {path}"); continue
            
            m_probs = []
            for i in valid_indices:
                if i >= len(d_items): m_probs.append(np.full(K, 1.0/K)); continue
                rec = d_items[i]
                probs = rec.get('probabilities', {})
                vec = np.zeros(K)
                for l_str, l_idx in label_map.items(): vec[l_idx] = probs.get(l_str, 0.0)
                if vec.sum() == 0: vec = np.full(K, 1.0/K)
                else: vec /= vec.sum()
                m_probs.append(vec)
            all_preds.append(np.array(m_probs))
        return np.stack(all_preds, axis=0), labels, group_array

    print("Loading Train...")
    train_preds, train_labels, train_groups = load_set([train_map[m] for m in common_models])
    print("Loading Test...")
    test_preds, test_labels, test_groups = load_set([test_map[m] for m in common_models])
    return train_preds, train_labels, train_groups, test_preds, test_labels, test_groups, common_models, M, K

def calculate_metrics_with_groups_civil_comments(w, predictions, labels, groups):
    ensemble_probs = np.einsum('m,mnk->nk', w, predictions)
    predicted_labels = np.argmax(ensemble_probs, axis=1)
    acc = np.mean(predicted_labels == labels)
    
    
    one_hot = np.zeros((len(labels), predictions.shape[2]))
    one_hot[np.arange(len(labels)), labels] = 1.0
    risk = np.mean(np.sum((one_hot - ensemble_probs) ** 2, axis=1))
    
    unique_groups = np.unique(groups)
    group_accuracies = {}
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        if len(idx) > 0:
            g_acc = np.mean(predicted_labels[idx] == labels[idx])
            group_accuracies[g] = g_acc
    wga = min(group_accuracies.values()) if group_accuracies else 0.0
    return acc, risk, wga, group_accuracies


