import torch
def solve_erm_cross_entropy(predictions, labels, M):
    """
    Optimizes ensemble weights using Cross Entropy Loss and plots the training curve.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Solving ERM Ensemble via Cross Entropy on {device} ---")
    
    # predictions shape: (M, N, K) -> (N, M, K)
    preds_tensor = torch.tensor(predictions, dtype=torch.float32).permute(1, 0, 2).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    
    N = preds_tensor.shape[0]
    
    # Extract probabilities of the true class
    # Gather indices: (N, 1) -> expanded to (N, M) for gather 
    labels_expanded = labels_tensor.unsqueeze(1).unsqueeze(2).expand(N, M, 1)
    probs_true_class = torch.gather(preds_tensor, 2, labels_expanded).squeeze(2) # Shape (N, M)
    
    # Optimization Variable (logits for weights to ensure sum=1, >0 via Softmax)
    w_logits = torch.zeros(M, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([w_logits], lr=0.01)
    
    
    loss_history = []
    
    for i in range(500): 
        optimizer.zero_grad()
        w = torch.softmax(w_logits, dim=0)
        
        
        ensemble_probs = probs_true_class @ w
        
        
        loss = -torch.mean(torch.log(ensemble_probs + 1e-10))
        
        loss.backward()
        optimizer.step()
        
        
        loss_history.append(loss.item())


        
    final_weights = torch.softmax(w_logits, dim=0).detach().cpu().numpy()
    
    print(f"Final Loss: {loss_history[-1]:.4f}")
    return final_weights
