
import torch
import numpy as np
from typing import Tuple, Dict, List, Optional


def solve_group_dro_ensemble(
    predictions: np.ndarray,
    labels: np.ndarray,
    M: int,
    K: int,
    n_iterations: int = 200,
    step_size_q: float = 0.1,
    lr_w: float = 0.1,
    use_adjustment: bool = False,
    adjustment_coef: float = 0.0,
    loss_type: str = 'cross_entropy',
    verbose: bool = True,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
    """
    Group DRO for ensemble weight optimization.
    
    Solves: min_w max_q∈Δ^K  Σ_k q_k · L_k(w)
    
    where L_k(w) is the average loss on group (class) k.
    
    This is equivalent to:
        min_w  max_k L_k(w)   (as the max emerges from the minimax)
    
    Algorithm:
        1. Compute per-group losses L_k(w)
        2. Update q via exponentiated gradient ascent: q ← q * exp(η * L)
        3. Normalize q
        4. Update w via gradient descent on weighted loss
    
    Args:
        predictions: (M, N, K) - M models, N samples, K classes
        labels: (N,) - ground truth labels
        M: number of ensemble models (NOT USED from signature, computed from predictions)
        K: number of classes/groups (NOT USED from signature, computed from predictions)
        n_iterations: number of optimization iterations
        step_size_q: learning rate for q update (exponentiated gradient)
        lr_w: learning rate for w update
        use_adjustment: whether to use group adjustment C/√n_g
        adjustment_coef: the coefficient C for adjustment
        loss_type: 'cross_entropy' or 'brier'
        verbose: print progress
        device: torch device (auto-detected if None)
    
    Returns:
        w: (M,) optimal ensemble weights
        q_history: list of q vectors over iterations
        info: dict with training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get dimensions from data
    M_actual, N, K_actual = predictions.shape
    
    # Convert to tensors
    preds = torch.tensor(predictions, dtype=torch.float64, device=device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    
    # Group information
    group_counts = torch.bincount(labels_t, minlength=K_actual).float()
    group_indices = [torch.where(labels_t == g)[0] for g in range(K_actual)]
    
    # Compute adjustments if needed
    if use_adjustment and adjustment_coef > 0:
        # C / sqrt(n_g) - upweights small groups
        adjustments = adjustment_coef / torch.sqrt(torch.clamp(group_counts, min=1.0))
        adjustments = adjustments.to(device)
    else:
        adjustments = torch.zeros(K_actual, device=device)
    
    # Initialize
    q = torch.ones(K_actual, dtype=torch.float64, device=device) / K_actual
    w = torch.ones(M_actual, dtype=torch.float64, device=device) / M_actual
    
    # History tracking
    q_history = [q.cpu().numpy().copy()]
    info = {
        'worst_risk': [],
        'avg_risk': [],
        'group_risks': [],
        'w_history': [w.cpu().numpy().copy()]
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("  Group DRO for Ensemble Learning")
        print(f"{'='*60}")
        print(f"  Device: {device}")
        print(f"  Loss: {loss_type}")
        print(f"  η_q: {step_size_q}, η_w: {lr_w}")
        print(f"  Iterations: {n_iterations}")
        if use_adjustment:
            print(f"  Adjustment: C={adjustment_coef}")
    
    for t in range(n_iterations):
        
        group_losses = _compute_all_group_losses(
            preds, labels_t, w, group_indices, K_actual, loss_type
        )
        
        # Apply adjustment (additive, as in paper)
        adjusted_losses = group_losses + adjustments
        
        # Track metrics
        worst_loss = torch.max(group_losses).item()
        avg_loss = (torch.sum(group_counts * group_losses) / N).item()
        
        info['worst_risk'].append(worst_loss)
        info['avg_risk'].append(avg_loss)
        info['group_risks'].append(group_losses.cpu().numpy().copy())
        
        if verbose and (t % 20 == 0 or t == n_iterations - 1):
            print(f"  Iter {t:3d}: worst={worst_loss:.4f}, avg={avg_loss:.4f}, "
                  f"q={np.round(q.cpu().numpy(), 3)}")
        
        
        # q_new = q * exp(η * adjusted_losses)  [maximize weighted loss]
        q_new = q * torch.exp(step_size_q * adjusted_losses)
        q_new = q_new / (q_new.sum() + 1e-10)
        
        # Convergence check
        q_change = torch.norm(q_new - q).item()
        q = q_new
        q_history.append(q.cpu().numpy().copy())
        
        
        w = _update_w_gradient_descent(
            preds, labels_t, w, q, group_indices, K_actual,
            loss_type, lr_w, n_inner_steps=5
        )
        
        info['w_history'].append(w.cpu().numpy().copy())
        
        # Early stopping if converged
        if q_change < 1e-6 and t > 20:
            if verbose:
                print(f"  Converged at iteration {t}")
            break
    
    if verbose:
        final_losses = _compute_all_group_losses(
            preds, labels_t, w, group_indices, K_actual, loss_type
        )
        print(f"\n  Final Results:")
        print(f"    w = {np.round(w.cpu().numpy(), 4)}")
        print(f"    q = {np.round(q.cpu().numpy(), 4)}")
        print(f"    Per-group losses: {np.round(final_losses.cpu().numpy(), 4)}")
        print(f"    Worst-group loss: {torch.max(final_losses).item():.4f}")
        print(f"{'='*60}\n")
    
    return w.cpu().numpy(), q_history, info


def _compute_all_group_losses(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    w: torch.Tensor,
    group_indices: List[torch.Tensor],
    K: int,
    loss_type: str
) -> torch.Tensor:
    """Compute L_g(w) for all groups g."""
    losses = torch.zeros(K, dtype=predictions.dtype, device=predictions.device)
    
    for g in range(K):
        if len(group_indices[g]) == 0:
            losses[g] = 0.0
            continue
        
        idx = group_indices[g]
        
        # Ensemble prediction: (n_g, K)
        ensemble_pred = torch.einsum('m,mnk->nk', w, predictions[:, idx, :])
        
        if loss_type == 'cross_entropy':
            # Get probability of true class
            true_probs = ensemble_pred[torch.arange(len(idx), device=labels.device), 
                                       labels[idx]]
            true_probs = torch.clamp(true_probs, min=1e-10, max=1.0)
            losses[g] = -torch.mean(torch.log(true_probs))
            
        elif loss_type == 'brier':
            # Brier score: ||one_hot(y) - p||^2
            one_hot = torch.zeros_like(ensemble_pred)
            one_hot[torch.arange(len(idx), device=labels.device), labels[idx]] = 1.0
            losses[g] = torch.mean(torch.sum((one_hot - ensemble_pred) ** 2, dim=1))
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
    
    return losses


def _update_w_gradient_descent(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    w: torch.Tensor,
    q: torch.Tensor,
    group_indices: List[torch.Tensor],
    K: int,
    loss_type: str,
    lr: float,
    n_inner_steps: int = 5
) -> torch.Tensor:
    """
    Update ensemble weights w by gradient descent on weighted loss.
    
    Objective: min_w Σ_g q_g · L_g(w)
    """
    M = w.shape[0]
    w_new = w.clone()
    
    for _ in range(n_inner_steps):
        grad = torch.zeros(M, dtype=w.dtype, device=w.device)
        
        for g in range(K):
            if len(group_indices[g]) == 0 or q[g] < 1e-10:
                continue
            
            idx = group_indices[g]
            preds_g = predictions[:, idx, :]  # (M, n_g, K)
            
            # Ensemble prediction
            ensemble_pred = torch.einsum('m,mnk->nk', w_new, preds_g)  # (n_g, K)
            
            if loss_type == 'cross_entropy':
                # Gradient of -log(p_y) w.r.t. w
                # d/dw_m [-log(Σ_j w_j p_j(y))] = -p_m(y) / (Σ_j w_j p_j(y))
                true_probs_ensemble = ensemble_pred[
                    torch.arange(len(idx), device=labels.device), 
                    labels[idx]
                ]  # (n_g,)
                true_probs_ensemble = torch.clamp(true_probs_ensemble, min=1e-10)
                
                # p_m(y) for each model m: (M, n_g)
                labels_g = labels[idx]
                p_m_y = preds_g[:, torch.arange(len(idx)), labels_g]  # (M, n_g)
                
                # Gradient: -mean(p_m(y) / p_ens(y))
                grad_g = -torch.mean(p_m_y / true_probs_ensemble, dim=1)  # (M,)
                
            elif loss_type == 'brier':
                # Gradient of Brier score
                # d/dw_m [||y - p||^2] = -2 * (y - p) · p_m
                one_hot = torch.zeros_like(ensemble_pred)
                one_hot[torch.arange(len(idx), device=labels.device), labels[idx]] = 1.0
                residuals = one_hot - ensemble_pred  # (n_g, K)
                
                # Sum over classes, mean over samples
                grad_g = -2 * torch.mean(
                    torch.einsum('nk,mnk->mn', residuals, preds_g), dim=1
                )  # (M,)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")
            
            grad += q[g] * grad_g
        
        # Gradient descent step
        w_new = w_new - lr * grad
        
        # Project to simplex
        w_new = _project_to_simplex_torch(w_new)
    
    return w_new


def _project_to_simplex_torch(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto probability simplex."""
    n = v.shape[0]
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0
    
    if not torch.any(cond):
        return torch.ones(n, device=v.device, dtype=v.dtype) / n
    
    rho = torch.nonzero(cond, as_tuple=False)[-1].item() + 1
    theta = cssv[rho - 1] / rho
    
    w = torch.clamp(v - theta, min=0)
    return w / (w.sum() + 1e-10)




def solve_group_dro_ensemble_compat(
    predictions: np.ndarray,
    labels: np.ndarray,
    A_matrices: np.ndarray,  # IGNORED - not part of Group DRO
    b_vectors: np.ndarray,   # IGNORED - not part of Group DRO
    M: int,
    K: int,
    n_iterations: int = 200,
    step_size: float = 0.01,
    convergence_tol: float = 1e-6,
    use_adjustment: bool = False,
    adjustment_coef: float = 0.0,
    loss_type: str = 'cross_entropy',
    verbose: bool = True,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
    """
    Backward-compatible wrapper.
    
    NOTE: A_matrices and b_vectors are IGNORED - they are not part of 
    the Group DRO algorithm from Sagawa et al.
    """
    if verbose and (A_matrices is not None or b_vectors is not None):
        print("  [Note] A_matrices/b_vectors ignored - not part of Group DRO")
    
    return solve_group_dro_ensemble(
        predictions=predictions,
        labels=labels,
        M=M,
        K=K,
        n_iterations=n_iterations,
        step_size_q=step_size,
        lr_w=0.1,
        use_adjustment=use_adjustment,
        adjustment_coef=adjustment_coef,
        loss_type=loss_type,
        verbose=verbose,
        device=device
    )



if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=" * 70)
    print("  Testing Corrected Group DRO Implementation")
    print("=" * 70)
    
    
    M, N, K = 5, 1000, 5
    
    
    class_probs = [0.05, 0.15, 0.25, 0.25, 0.30]
    labels = np.random.choice(K, size=N, p=class_probs)
    
   
    predictions = np.zeros((M, N, K))
    for n in range(N):
        true_class = labels[n]
        for m in range(M):
            
            if true_class == 0:
                strength = 0.3  
            else:
                strength = 0.6 + 0.1 * (m % 3)  
            
            probs = np.random.dirichlet(np.ones(K) * 0.3)
            probs[true_class] += strength
            probs = probs / probs.sum()
            predictions[m, n] = probs
    
    print(f"\nData: M={M} models, N={N} samples, K={K} classes")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Test Group DRO
    w, q_history, info = solve_group_dro_ensemble(
        predictions, labels, M, K,
        n_iterations=100,
        step_size_q=0.05,
        lr_w=0.1,
        loss_type='cross_entropy',
        verbose=True
    )
    
    # Evaluate
    ensemble_pred = np.einsum('m,mnk->nk', w, predictions)
    pred_labels = np.argmax(ensemble_pred, axis=1)
    
    print(f"\nEvaluation:")
    print(f"  Overall accuracy: {np.mean(pred_labels == labels):.4f}")
    
    for k in range(K):
        mask = labels == k
        if mask.sum() > 0:
            class_acc = np.mean(pred_labels[mask] == k)
            print(f"  Class {k} accuracy: {class_acc:.4f} (n={mask.sum()})")
    
    # Compare with uniform weights
    w_uniform = np.ones(M) / M
    ensemble_uniform = np.einsum('m,mnk->nk', w_uniform, predictions)
    pred_uniform = np.argmax(ensemble_uniform, axis=1)
    
    print(f"\n  Uniform weights accuracy: {np.mean(pred_uniform == labels):.4f}")
    for k in range(K):
        mask = labels == k
        if mask.sum() > 0:
            class_acc = np.mean(pred_uniform[mask] == k)
            print(f"    Class {k}: {class_acc:.4f}")
    
    print("\n" + "=" * 70)
    print("  Group DRO should improve worst-group (minority class) performance")
    print("=" * 70)