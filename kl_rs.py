import math
import torch
def solve_kl_rs_ensemble(predictions, labels, M, tau_factor=1.1, lambda_min=1e-4, lambda_max=100.0, epsilon=1e-3):
    """
    Solves for ensemble weights using the KL-Divergence-based Robust Satisficing (KL-RS) model.
    Paper: arXiv:2408.09157v1
    
    Args:
        predictions: (M, N, K) numpy array
        labels: (N,) numpy array
        M: Number of models
        tau_factor: Multiplier for ERM loss to set target tau (tau = tau_factor * erm_loss).
                    Paper suggests (1 + epsilon) * E_0.
        lambda_min: Minimum lambda for bisection.
        lambda_max: Maximum lambda for bisection.
        epsilon: Convergence threshold for bisection.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Solving KL-RS Ensemble on {device} (tau_factor={tau_factor}) ---")

    
    preds_tensor = torch.tensor(predictions, dtype=torch.float32).permute(1, 0, 2).to(device) # (N, M, K)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    N = preds_tensor.shape[0]

    
    labels_expanded = labels_tensor.unsqueeze(1).unsqueeze(2).expand(N, M, 1)
    probs_true = torch.gather(preds_tensor, 2, labels_expanded).squeeze(2) # (N, M)

    
    w_logits_erm = torch.zeros(M, requires_grad=True, device=device)
    opt_erm = torch.optim.Adam([w_logits_erm], lr=0.05)
    
    erm_loss_val = 0.0
    for _ in range(200):
        opt_erm.zero_grad()
        w = torch.softmax(w_logits_erm, dim=0)
        p = probs_true @ w
        loss = -torch.mean(torch.log(p + 1e-10))
        loss.backward()
        opt_erm.step()
        erm_loss_val = loss.item()

    tau = erm_loss_val * tau_factor
    print(f"  > ERM Loss: {erm_loss_val:.4f} | Target Tau: {tau:.4f}")

    # 3. Define Feasibility Oracle (Algorithm 1 from paper)
    def check_feasibility(lam_val):
        """
        Checks if a given lambda allows for a solution satisfying the constraint.
        Minimizes E[exp((loss - tau)/lambda)]. If min value <= 1, it's feasible.
        """
        lam = torch.tensor(lam_val, dtype=torch.float32, device=device)
        w_logits = torch.zeros(M, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([w_logits], lr=0.05)
        
        min_obj_val = float('inf')
        best_w_state = w_logits.detach().clone()
        
        # Optimization loop for fixed lambda
        for _ in range(300): # Iterations for inner optimization
            optimizer.zero_grad()
            w = torch.softmax(w_logits, dim=0)
            
            # Per-sample loss: l(theta, z)
            ensemble_probs = probs_true @ w
            per_sample_loss = -torch.log(ensemble_probs + 1e-10)
            
            # Tilted Objective: E[exp((l - tau)/lambda)]
            # We compute exp((l - tau)/lambda) carefully to avoid overflow if lambda is tiny
            scaled_loss = (per_sample_loss - tau) / lam
            objective = torch.mean(torch.exp(scaled_loss))
            
            objective.backward()
            optimizer.step()
            
            if objective.item() < min_obj_val:
                min_obj_val = objective.item()
                best_w_state = w_logits.detach().clone()
            
            # Early exit if feasible
            if min_obj_val <= 1.0:
                return True, best_w_state

        return min_obj_val <= 1.0, best_w_state

    # 4. Bisection Search (Algorithm 2 from paper )
    # The paper notes R(theta, lambda) is non-increasing w.r.t lambda.
    # We want the smallest lambda (most robust) that is feasible.
    
    low = lambda_min
    high = lambda_max
    best_w_logits = w_logits_erm.detach().clone() 
    optimal_lambda = high

    # Initial check on boundaries
    is_feasible_min, _ = check_feasibility(low)
    if is_feasible_min:
        print("  > Minimum lambda is feasible. Returning min robustness.")
        return torch.softmax(best_w_logits, dim=0).detach().cpu().numpy(), low

    for _ in range(int(math.log2((lambda_max - lambda_min)/epsilon)) + 5):
        mid = (low + high) / 2
        is_feasible, w_state = check_feasibility(mid)
        
        if is_feasible:
            # If feasible, try to find a smaller (tighter) lambda (more robustness)
            optimal_lambda = mid
            best_w_logits = w_state
            high = mid
        else:
            # If not feasible, we need a larger lambda (less robustness constraints)
            low = mid
            
        if (high - low) < epsilon:
            break

    print(f"  > Optimal Lambda: {optimal_lambda:.4f}")
    final_w = torch.softmax(best_w_logits, dim=0).detach().cpu().numpy()
    return final_w, optimal_lambda