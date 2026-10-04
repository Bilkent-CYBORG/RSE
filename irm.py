import time
import torch


def solve_irm_ensemble(predictions, labels, q_s, M, lambda_irm=1.0, n_iterations=200, lr=0.01):
    """
    Solves for ensemble weights using Cross Entropy and the EXACT IRMv1 penalty.
    Treats each CLASS as an 'Environment' to enforce invariance across classes.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Training IRM Ensemble (Cross Entropy) on {device} (lambda={lambda_irm}) ---")

    # 1. Prepare Data
    preds_tensor = torch.tensor(predictions, dtype=torch.float32).permute(1, 0, 2).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    env_weights = torch.tensor(q_s, dtype=torch.float32).to(device) 
    
    N = preds_tensor.shape[0]
    K = preds_tensor.shape[2]

    # Pre-calculate masks for each environment (Class = Environment)
    env_indices = [torch.where(labels_tensor == k)[0] for k in range(K)]

    # 2. Optimization Variable (Logits for Softmax Weights)
    w_logits = torch.zeros(M, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([w_logits], lr=lr)

    start_time = time.perf_counter()

    for t in range(n_iterations):
        optimizer.zero_grad()
        
        # Calculate Softmax Weights (w)
        w = torch.softmax(w_logits, dim=0)
        
        total_erm_loss = 0.0
        total_irm_penalty = 0.0
        valid_envs = 0

        # --- Loop over Environments (Classes) ---
        for k in range(K):
            indices = env_indices[k]
            if len(indices) == 0: continue
            valid_envs += 1
            
            # Extract data for this environment
            # env_preds: (N_k, M, K)
            env_preds = preds_tensor[indices]
            # We only care about the probability of the TRUE class (which is k)
            # Shape: (N_k, M)
            probs_true = env_preds[:, :, k] 

            # A. ERM Risk (NLL) for this environment
            # Ensemble Prob = Weighted Sum of Model Probs
            ensemble_probs = probs_true @ w
            nll = -torch.mean(torch.log(ensemble_probs + 1e-10))
            
            # Weighted by class prevalence (q_s)
            total_erm_loss += env_weights[k] * nll

            # B. IRM Penalty (Gradient Norm w.r.t Dummy Scale)
            # 1. Create Dummy Scale
            scale = torch.tensor(1.0).to(device).requires_grad_()
            
            # 2. Apply Scale (w_scaled = w * scale)
            # scaled_probs = probs_true @ (w * scale)
            # Note: In linear ensemble, scaling w scales the output probability directly.
            scaled_probs = probs_true @ (w * scale)
            
            # 3. Calculate Loss with Scale
            scaled_nll = -torch.mean(torch.log(scaled_probs + 1e-10))
            
            # 4. Compute Gradient w.r.t Scale
            grad_scale = torch.autograd.grad(
                scaled_nll, [scale], create_graph=True
            )[0]
            
            # 5. Penalty = Gradient Squared
            total_irm_penalty += grad_scale ** 2

        # Average penalty across valid environments
        if valid_envs > 0:
            total_irm_penalty /= valid_envs

        # --- Final Objective ---
        final_objective = total_erm_loss + lambda_irm * total_irm_penalty
        
        final_objective.backward()
        optimizer.step()
        
    training_time = time.perf_counter() - start_time
    final_w = torch.softmax(w_logits, dim=0).detach().cpu().numpy()
    
    print(f" IRM training finished in {training_time:.2f} seconds.")
    return final_w, training_time