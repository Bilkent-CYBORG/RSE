"""
# Third-party implementation of the SRM algorithm.
# Original Paper: "Structure-informed Risk Minimization..." (ICML 2025)
# Note: We adapted this method to use class-based environments as described below.
=============================================================================
KEY DIFFERENCES FROM ORIGINAL PAPER:
=============================================================================

1. ENVIRONMENT DEFINITION:
   - Paper: Uses domains (e.g., different time periods in FMoW) as environments
   - Our adaptation: Uses classes as environments (common in Group DRO literature)
   
2. FEATURE SPACE:
   - Paper: Computes Wasserstein distances on deep features (featurizer output)
   - Our adaptation: Uses prediction probabilities (what we have access to)
   - Alternative: Uses per-model prediction vectors as higher-dim representation

3. GRAPH CONSTRUCTION:
   - Paper: Full Wasserstein distance on feature distributions
   - Our adaptation: Wasserstein distance on prediction distributions
   
These adaptations are necessary when:
- No domain/environment labels are available
- Only model predictions (not features) are available

=============================================================================
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from typing import Tuple, Optional, Dict, Literal
import warnings


def gaussian_wasserstein_distance(mu1: np.ndarray, cov1: np.ndarray,
                                   mu2: np.ndarray, cov2: np.ndarray) -> float:
    """
    Compute 2-Wasserstein distance between two Gaussian distributions.
    
    W2^2(P1, P2) = ||μ1 - μ2||^2 + Tr(Σ1 + Σ2 - 2(Σ1^{1/2} Σ2 Σ1^{1/2})^{1/2})
    """
    mean_diff_sq = np.linalg.norm(mu1 - mu2) ** 2
    
    try:
        # Full Wasserstein formula
        sqrt_cov1 = sqrtm(cov1)
        if np.iscomplexobj(sqrt_cov1):
            sqrt_cov1 = sqrt_cov1.real
        
        product = sqrt_cov1 @ cov2 @ sqrt_cov1
        sqrt_product = sqrtm(product)
        if np.iscomplexobj(sqrt_product):
            sqrt_product = sqrt_product.real
        
        cov_term = np.trace(cov1) + np.trace(cov2) - 2 * np.trace(sqrt_product)
        cov_term = max(0, cov_term)  # Numerical stability
        
    except Exception:
    
        try:
            sqrt_cov1 = sqrtm(cov1).real if np.iscomplexobj(sqrtm(cov1)) else sqrtm(cov1)
            sqrt_cov2 = sqrtm(cov2).real if np.iscomplexobj(sqrtm(cov2)) else sqrtm(cov2)
            cov_term = np.linalg.norm(sqrt_cov1 - sqrt_cov2, 'fro') ** 2
        except:
            cov_term = abs(np.trace(cov1) - np.trace(cov2))
    
    return np.sqrt(max(0, mean_diff_sq + cov_term))


def compute_class_statistics_extended(predictions: np.ndarray, 
                                       labels: np.ndarray,
                                       M: int, K: int,
                                       use_extended_features: bool = True) -> Dict:
    
    class_stats = {}
    
    for k in range(K):
        class_mask = labels == k
        n_k = np.sum(class_mask)
        
        if n_k == 0:
            dim = M * K if use_extended_features else K
            class_stats[k] = {
                'mean': np.zeros(dim),
                'cov': np.eye(dim) * 1e-6,
                'n_samples': 0
            }
            continue
        
        # Get predictions for samples of class k: (M, n_k, K)
        class_preds = predictions[:, class_mask, :]
        
        if use_extended_features:
            # Concatenate all model predictions: (n_k, M*K)
            # This preserves individual model information
            features = class_preds.transpose(1, 0, 2).reshape(n_k, M * K)
        else:
            # Average across models: (n_k, K)
            features = np.mean(class_preds, axis=0)
        
        # Compute mean and covariance
        mean_feat = np.mean(features, axis=0)
        
        if n_k > features.shape[1]:  # Need more samples than dimensions
            cov_feat = np.cov(features.T)
            if cov_feat.ndim == 0:
                cov_feat = np.array([[cov_feat]])
        else:
            # Shrinkage estimator for ill-conditioned case
            cov_feat = np.cov(features.T) if n_k > 1 else np.eye(features.shape[1]) * 1e-6
            if cov_feat.ndim == 0:
                cov_feat = np.array([[cov_feat]])
            # Ledoit-Wolf style shrinkage
            cov_feat = 0.9 * cov_feat + 0.1 * np.eye(cov_feat.shape[0]) * np.trace(cov_feat) / cov_feat.shape[0]
        
        # Regularization for numerical stability
        cov_feat = cov_feat + 1e-5 * np.eye(cov_feat.shape[0])
        
        class_stats[k] = {
            'mean': mean_feat,
            'cov': cov_feat,
            'n_samples': n_k
        }
    
    return class_stats


def build_distribution_graph(class_stats: Dict, K: int) -> np.ndarray:
    """
    Build the distribution graph G.
    
    Nodes = classes (environments/distributions)
    Edge weights = 2-Wasserstein distance between distributions
    
    Returns:
        A: (K, K) distance matrix
    """
    A = np.zeros((K, K))
    
    for i in range(K):
        for j in range(i + 1, K):
            if class_stats[i]['n_samples'] == 0 or class_stats[j]['n_samples'] == 0:
                dist = 1e6
            else:
                dist = gaussian_wasserstein_distance(
                    class_stats[i]['mean'], class_stats[i]['cov'],
                    class_stats[j]['mean'], class_stats[j]['cov']
                )
            A[i, j] = dist
            A[j, i] = dist
    
    return A


def compute_graph_centrality(A: np.ndarray, centrality_type: str = 'closeness') -> np.ndarray:
    """
    Compute centrality for each node in the graph.
    
    Args:
        A: (K, K) distance matrix
        centrality_type: 'closeness', 'degree', or 'eigenvector'
    
    Returns:
        centrality: (K,) centrality scores
    """
    K = A.shape[0]
    
    if centrality_type == 'closeness':
        # Closeness centrality: c(v) = (K-1) / Σ_u d(v,u)
        # Higher = more central
        centrality = np.zeros(K)
        for i in range(K):
            total_dist = np.sum(A[i, :]) + 1e-10
            centrality[i] = (K - 1) / total_dist
            
    elif centrality_type == 'degree':
        # Degree centrality (inverse of average distance)
        centrality = 1.0 / (np.mean(A, axis=1) + 1e-10)
        
    elif centrality_type == 'eigenvector':
        # Eigenvector centrality on similarity matrix
        # Convert distance to similarity
        max_dist = np.max(A) + 1e-10
        similarity = max_dist - A
        np.fill_diagonal(similarity, 0)
        
        # Power iteration
        centrality = np.ones(K) / K
        for _ in range(100):
            centrality = similarity @ centrality
            centrality = centrality / (np.linalg.norm(centrality) + 1e-10)
    else:
        raise ValueError(f"Unknown centrality_type: {centrality_type}")
    
    return centrality


def centrality_to_prior(centrality: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Convert centrality scores to prior distribution.
    
    p_k = softmax(centrality / temperature)
    
    Higher temperature = more uniform prior
    Lower temperature = more peaked at high-centrality classes
    """
    # Normalize centrality to prevent overflow
    centrality_normalized = centrality - np.max(centrality)
    p = np.exp(centrality_normalized / temperature)
    p = p / (p.sum() + 1e-10)
    return p


def project_to_simplex(v: np.ndarray) -> np.ndarray:

    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    
    if not np.any(cond):
        return np.ones(n) / n
    
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = np.maximum(v - theta, 0)
    
    # Ensure exact sum to 1
    w = w / (w.sum() + 1e-10)
    return w


def compute_per_class_risk(predictions: np.ndarray,
                           labels: np.ndarray,
                           w: np.ndarray,
                           K: int,
                           loss_type: str = 'cross_entropy') -> np.ndarray:
    """
    Compute risk R_k(w) for each class k.
    
    R_k(w) = E_{(x,y) ~ P_k} [L(Σ w_m f_m(x), y)]
    """
    M, N, _ = predictions.shape
    
    # Ensemble prediction: (N, K)
    ensemble_pred = np.einsum('m,mnk->nk', w, predictions)
    
    risks = np.zeros(K)
    
    for k in range(K):
        mask = labels == k
        n_k = mask.sum()
        
        if n_k == 0:
            risks[k] = 1.0  # Max risk for empty class
            continue
        
        class_preds = ensemble_pred[mask]  # (n_k, K)
        
        if loss_type == 'cross_entropy':
            # -log(p_k) where k is the true class
            log_probs = np.log(class_preds[:, k] + 1e-10)
            risks[k] = -np.mean(log_probs)
        elif loss_type == 'brier':
            # ||one_hot(k) - p||^2
            one_hot = np.zeros((n_k, K))
            one_hot[:, k] = 1.0
            risks[k] = np.mean(np.sum((one_hot - class_preds) ** 2, axis=1))
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
    
    return risks


def solve_srm_ensemble(predictions: np.ndarray,
                       labels: np.ndarray,
                       M: int,
                       K: int,
                       lambda_reg: float = 1.0,
                       n_iterations: int = 200,
                       lr_w: float = 0.1,
                       lr_q: float = 0.1,
                       loss_type: str = 'cross_entropy',
                       centrality_type: str = 'closeness',
                       use_extended_features: bool = True,
                       verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Structure-informed Risk Minimization (SRM) for Ensemble Learning.
    
    Solves the minimax problem:
        min_w max_q  Σ_k q_k R_k(w) - λ D_KL(q || p)
    
    where:
    - w ∈ Δ^M: ensemble weights
    - q ∈ Δ^K: adversarial class weights  
    - p ∈ Δ^K: centrality-based prior (structure-informed)
    - λ: regularization controlling robustness vs. structure adherence
    
    Key insight from paper:
    - λ = 0: Pure worst-case (Group DRO)
    - λ → ∞: Follow prior (ERM-like if prior = source distribution)
    - Intermediate λ: Balance robustness with structural plausibility
    
    Args:
        predictions: (M, N, K) model probability outputs
        labels: (N,) ground truth labels
        M: number of models
        K: number of classes
        lambda_reg: KL regularization strength
        n_iterations: optimization iterations
        lr_w: learning rate for w (gradient descent)
        lr_q: learning rate for q (gradient ascent)
        loss_type: 'cross_entropy' or 'brier'
        centrality_type: 'closeness', 'degree', or 'eigenvector'
        use_extended_features: use (M*K)-dim features for Wasserstein
        verbose: print progress
    
    Returns:
        w: (M,) optimal ensemble weights
        q: (K,) final adversarial weights
        info: dict with optimization details
    """
    if verbose:
        print("\n" + "=" * 65)
        print("  SRM: Structure-informed Risk Minimization")
        print("=" * 65)
        print(f"  Config: λ={lambda_reg}, loss={loss_type}, centrality={centrality_type}")
        print(f"  Extended features: {use_extended_features}")
    

    if verbose:
        print("\n  [Phase 1] Building distributional graph...")
    
    # Compute class statistics
    class_stats = compute_class_statistics_extended(
        predictions, labels, M, K, 
        use_extended_features=use_extended_features
    )
    
    # Build Wasserstein distance matrix
    dist_matrix = build_distribution_graph(class_stats, K)
    
    # Compute centrality
    centrality = compute_graph_centrality(dist_matrix, centrality_type)
    
    # Convert to prior
    prior_p = centrality_to_prior(centrality, temperature=1.0)
    
    if verbose:
        print(f"  Wasserstein distances:\n{np.round(dist_matrix, 2)}")
        print(f"  Centrality ({centrality_type}): {np.round(centrality, 4)}")
        print(f"  Prior p: {np.round(prior_p, 4)}")
    
  
    if verbose:
        print(f"\n  [Phase 2] Optimizing ({n_iterations} iterations)...")
    
    # Initialize on simplex
    w = np.ones(M) / M
    q = prior_p.copy()  # Start from prior (warm start)
    
    log_objective = []
    log_risks = []
    
    for t in range(n_iterations):
       
        risks = compute_per_class_risk(predictions, labels, w, K, loss_type)
        
       
        # Gradient w.r.t. w: ∂/∂w [Σ_k q_k R_k(w)]
        # We use numerical gradient for simplicity
        grad_w = np.zeros(M)
        eps = 1e-5
        for m in range(M):
            w_plus = w.copy()
            w_plus[m] += eps
            w_plus = w_plus / w_plus.sum()  # Re-normalize
            
            risks_plus = compute_per_class_risk(predictions, labels, w_plus, K, loss_type)
            grad_w[m] = (np.dot(q, risks_plus) - np.dot(q, risks)) / eps
        
        # Gradient descent step
        w = w - lr_w * grad_w
        w = project_to_simplex(w)
        
      
        # Objective for q: Σ_k q_k R_k - λ D_KL(q||p)
        # Gradient: ∂L/∂q_k = R_k - λ(1 + log(q_k) - log(p_k))
        
        grad_q = risks - lambda_reg * (1 + np.log(q + 1e-10) - np.log(prior_p + 1e-10))
        
        # Gradient ascent step
        q = q + lr_q * grad_q
        q = project_to_simplex(q)
        
        
        kl_div = np.sum(q * (np.log(q + 1e-10) - np.log(prior_p + 1e-10)))
        objective = np.dot(q, risks) - lambda_reg * kl_div
        
        log_objective.append(objective)
        log_risks.append(risks.copy())
        
        if verbose and (t + 1) % max(1, n_iterations // 5) == 0:
            print(f"    Iter {t+1:3d}: obj={objective:.4f}, "
                  f"risk={np.dot(q, risks):.4f}, KL={kl_div:.4f}")
    
    
    final_risks = compute_per_class_risk(predictions, labels, w, K, loss_type)
    
    if verbose:
        print(f"\n  [Result]")
        print(f"  Ensemble weights w: {np.round(w, 4)}")
        print(f"  Final q: {np.round(q, 4)}")
        print(f"  Prior p: {np.round(prior_p, 4)}")
        print(f"  Per-class risks: {np.round(final_risks, 4)}")
        print("=" * 65 + "\n")
    
    info = {
        'prior_p': prior_p,
        'centrality': centrality,
        'distance_matrix': dist_matrix,
        'class_stats': class_stats,
        'log_objective': log_objective,
        'log_risks': log_risks,
        'final_q': q,
        'final_risks': final_risks
    }
    
    return w, q, info




def solve_srm_ensemble_compat(predictions: np.ndarray,
                               labels: np.ndarray,
                               M: int,
                               K: int,
                               lambda_reg: float = 1.0,
                               n_iterations: int = 100,
                               lr_w: float = 0.05,
                               lr_q: float = 0.05,
                               loss_type: str = 'cross_entropy',
                               verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
   
    return solve_srm_ensemble(
        predictions=predictions,
        labels=labels,
        M=M,
        K=K,
        lambda_reg=lambda_reg,
        n_iterations=n_iterations,
        lr_w=lr_w,
        lr_q=lr_q,
        loss_type=loss_type,
        centrality_type='closeness',
        use_extended_features=True,
        verbose=verbose
    )


if __name__ == "__main__":
    np.random.seed(42)
    
    print("=" * 70)
    print("  Testing Improved SRM Implementation")
    print("=" * 70)
    
    
    M, N, K = 5, 500, 5
    
   
    predictions = np.zeros((M, N, K))
    labels = np.random.randint(0, K, size=N)
    
    for n in range(N):
        true_class = labels[n]
        for m in range(M):
            
            model_strength = 0.3 + 0.4 * ((m + true_class) % K == 0)
            probs = np.random.dirichlet(np.ones(K) * 0.5)
            probs[true_class] += model_strength
            probs = probs / probs.sum()
            predictions[m, n] = probs
    
    print(f"\nSynthetic data: M={M}, N={N}, K={K}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    
    for lam in [0.0, 0.5, 1.0, 5.0]:
        print(f"\n{'='*70}")
        print(f"  λ = {lam}")
        print(f"{'='*70}")
        
        w, q, info = solve_srm_ensemble(
            predictions, labels, M, K,
            lambda_reg=lam,
            n_iterations=50,
            verbose=True
        )
        
        
        ensemble_pred = np.einsum('m,mnk->nk', w, predictions)
        pred_labels = np.argmax(ensemble_pred, axis=1)
        accuracy = np.mean(pred_labels == labels)
        
        print(f"  Overall accuracy: {accuracy:.4f}")
        
        for k in range(K):
            mask = labels == k
            if mask.sum() > 0:
                class_acc = np.mean(pred_labels[mask] == k)
                print(f"    Class {k}: {class_acc:.4f} (n={mask.sum()}, q={q[k]:.3f})")
    
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print("  λ=0: Group DRO behavior (worst-case focus)")
    print("  λ→∞: Follows centrality prior (structure-informed)")
    print("  Intermediate λ: Balanced robustness")