import numpy as np
import cvxpy as cp
from time import time
import math 

def calculate_risk(w, A_j, b_j):
    return w.T @ A_j @ w + b_j.T @ w + 1.0

def compute_worst_case_q(r_vector, q_s, kappa):
    if kappa == 0.0 or np.isinf(kappa):
        if kappa == 0.0: q_star = np.zeros_like(q_s); q_star[np.argmax(r_vector)] = 1.0; return q_star
        return q_s
    r_max = np.max(r_vector); kappa_safe = max(kappa, 1e-12)
    exp_terms = np.exp((r_vector - r_max) / kappa_safe)
    numerator = q_s * exp_terms; denominator = np.sum(numerator)
    if denominator == 0: return q_s
    return numerator / denominator

def calculate_batch_moments(predictions, labels, M, K):
    A_matrices = np.zeros((K, M, M)); b_vectors = np.zeros((K, M))
    for j in range(K):
        class_indices = np.where(labels == j)[0]
        if len(class_indices) == 0: continue 
        class_preds = predictions[:, class_indices, :]
        p_bar_j = np.mean(class_preds, axis=1)
        b_vectors[j] = -2 * p_bar_j[:, j]
        for a in range(M):
            for b in range(M):
                pa=class_preds[a,:,:]; pb=class_preds[b,:,:]
                A_matrices[j,a,b]=np.mean(np.einsum('nk,nk->n', pa, pb))
    return A_matrices, b_vectors

def solve_k_series(qs, r_values, tau, order=4):
    qs = np.array(qs); r_values = np.array(r_values)
    if qs.sum() <= 0: qs = np.full_like(qs, 1.0 / len(qs))
    R_source = np.sum(qs * r_values); R_worst = np.max(r_values)
    if tau <= R_source + 1e-9: 
        print('kappa solve:', tau, R_source)
        return np.inf
    if tau >= R_worst - 1e-9: return 0.0
    max_moment = order + 1; m = np.zeros(max_moment + 1)
    for p in range(1, max_moment + 1): m[p] = np.sum(qs * (r_values ** p))
    kappa_cumulants = np.zeros(max_moment + 1); kappa_cumulants[1] = m[1]
    for n in range(2, max_moment + 1):
        s = 0.0
        for k in range(1, n):
            try: binom = math.comb(n - 1, k - 1)
            except ValueError: binom = 0
            s += binom * kappa_cumulants[k] * m[n - k]
        kappa_cumulants[n] = m[n] - s
    a = np.zeros(max_moment + 1)
    for mm in range(1, max_moment + 1):
        if mm > 170: a[mm] = 0
        else: a[mm] = kappa_cumulants[mm] / math.factorial(mm)
    if np.isclose(a[2], 0.0): return 0.0 
    delta_tau = tau - a[1]
    if delta_tau <= 1e-9: return np.inf
    c = np.zeros(order + 1); c[1] = 1.0 / a[2]
    S_poly = np.zeros(order + 1); S_poly[1] = c[1]
    for n in range(2, order + 1):
        powers = [None] * (n + 1); powers[0] = np.zeros(order + 1); powers[0][0] = 1.0
        powers[1] = S_poly.copy()
        for mm in range(2, n + 1): powers[mm] = np.convolve(powers[mm-1], S_poly)[:order+1]
        sum_term = 0.0
        for m in range(2, n + 1):
            if m + 1 > max_moment: continue
            if n < len(powers[m]): sum_term += a[m+1] * powers[m][n]
        if np.isclose(a[2], 0.0): return 0.0
        c[n] = -sum_term / a[2]
        if n < len(S_poly): S_poly[n] = c[n]
    S_approx = 0.0
    for n in range(1, order + 1): S_approx += c[n] * (delta_tau ** n)
    if not np.isfinite(S_approx) or S_approx <= 0.0:
        raise FloatingPointError(
            f"Invalid RSE inverse-series approximation S={S_approx!r}, "
            f"tau={tau!r}, R_source={R_source!r}, R_worst={R_worst!r}. "
            "The numerical failure must not be converted to kappa=0."
        )

    kappa = 1.0 / S_approx

    if not np.isfinite(kappa) or kappa <= 0.0:
        raise FloatingPointError(
            f"Invalid reciprocal kappa={kappa!r} from S={S_approx!r}."
        )

    return kappa


def solve_adversarial_weights(A_matrices, b_vectors, q_star, M, K):
    A_adv = np.zeros((M, M)); b_adv = np.zeros(M)
    for j in range(K): A_adv += q_star[j] * A_matrices[j]; b_adv += q_star[j] * b_vectors[j]
    A_adv += 1e-6 * np.eye(M) 
    w_opt = cp.Variable(M)
    try: 
        prob = cp.Problem(cp.Minimize(cp.quad_form(w_opt, A_adv) + b_adv @ w_opt), [cp.sum(w_opt)==1, w_opt>=0])
        prob.solve(solver='OSQP', verbose=False)
    except: 
        try: prob.solve(solver='SCS', verbose=False)
        except: return np.full(M, 1.0/M), np.inf
    if w_opt.value is None: return np.full(M, 1.0/M), np.inf
    final_w = np.maximum(w_opt.value, 0); final_w /= final_w.sum()
    return final_w, prob.value



def solve_adversarial_weights_proximal(A_matrices, b_vectors, q_star, M, K, w_prev=None, eta=0.1):
    """
    Solves for adversarial weights WITH proximal regularization term.
    
    Minimizes: <q_star, r(w)> + (1/2η)||w - w_prev||²₂
    
    Args:
        A_matrices: (K, M, M) array of second moment matrices
        b_vectors: (K, M) array of first moment vectors
        q_star: (K,) adversarial distribution over classes
        M: number of models
        K: number of classes
        w_prev: previous weight vector (if None, defaults to uniform)
        eta: step size parameter (smaller = stronger regularization)
    """
    
    
    
    A_adv = np.zeros((M, M))
    b_adv = np.zeros(M)
    for j in range(K):
        A_adv += q_star[j] * A_matrices[j]
        b_adv += q_star[j] * b_vectors[j]
    
    
    A_adv += 1e-6 * np.eye(M)
    
    
    w_opt = cp.Variable(M)
    
    # Original objective: w^T A w + b^T w
    original_obj = cp.quad_form(w_opt, A_adv) + b_adv @ w_opt
    
    # Proximal term: (1/2η)||w - w_prev||²₂
    if w_prev is None:
        w_prev = np.full(M, 1.0 / M) 
    
    proximal_term = (1.0 / (2 * eta)) * cp.sum_squares(w_opt - w_prev)
    
    # Combined objective
    objective = original_obj + proximal_term
    
    # Constraints: w in simplex
    constraints = [cp.sum(w_opt) == 1, w_opt >= 0]
    
    # Solve
    try:
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver='OSQP', verbose=False)
    except:
        try:
            prob.solve(solver='SCS', verbose=False)
        except:
            return np.full(M, 1.0/M), np.inf
    
    if w_opt.value is None:
        return np.full(M, 1.0/M), np.inf
    
    final_w = np.maximum(w_opt.value, 0)
    final_w /= final_w.sum()
    
    return final_w, prob.value


def solve_robust_offline_with_proximal(A_matrices, b_vectors, q_s, M, K, 
                                        tau_target_fraction, T_max_iter, 
                                        epsilon_convergence,
                                        eta=0.1):  # NEW PARAMETER
    
    print("\n--- Offline Robust Solver (WITH PROXIMAL REGULARIZATION) ---")
    
    # Initial ERM solution (no proximal term yet)
    w_init, _ = solve_adversarial_weights(A_matrices, b_vectors, q_s, M, K) # one-step warmup
    
    # Calculate tau
    erm_risks = np.array([calculate_risk(w_init, A_matrices[j], b_vectors[j]) for j in range(K)])
    R_source = np.sum(q_s * erm_risks)
    R_worst = np.max(erm_risks)
    tau = max(R_source + tau_target_fraction * max(0, R_worst - R_source), R_source + 1e-6)
    
    # Initialize weights
    w_t = w_init.copy()
    
    log_kappa = []
    
    import time
    start_robust = time.perf_counter()
    
    for t in range(T_max_iter):
        # Calculate current risks
        class_risks = np.array([calculate_risk(w_t, A_matrices[j], b_vectors[j]) for j in range(K)])
        
        # Solve for kappa (fragility)
        kappa_t = solve_k_series(q_s, class_risks, tau, order=8)
        log_kappa.append(kappa_t)
        
        if kappa_t is None:
            break
        
        # Compute adversarial distribution
        q_star = compute_worst_case_q(class_risks, q_s, kappa_t)
        
       
        w_next, _ = solve_adversarial_weights_proximal(
            A_matrices, b_vectors, q_star, M, K, 
            w_prev=w_t,  # Pass current weights
            eta=eta       # Step size parameter
        )
        
        
        w_t = w_next

        # Convergence check
        if t > 0 and np.linalg.norm(w_t - w_prev_iter) < epsilon_convergence:
            break
        w_prev_iter = w_t.copy()
    
    robust_time = time.perf_counter() - start_robust
    
    return w_init, w_t, {'log_kappa': log_kappa, 'final_tau': tau}, robust_time

def solve_robust_offline_with_proximal_hf(A_matrices, b_vectors, q_s, M, K, 
                                        tau_target_fraction, T_max_iter, 
                                        epsilon_convergence,
                                        eta=0.1):  
    
    print("\n--- Offline Robust Solver (WITH PROXIMAL REGULARIZATION) ---")
    kappa_history = []
    
    w_init, _ = solve_adversarial_weights(A_matrices, b_vectors, q_s, M, K)
    
    # Calculate tau
    erm_risks = np.array([calculate_risk(w_init, A_matrices[j], b_vectors[j]) for j in range(K)])
    R_source = np.sum(q_s * erm_risks)
    R_worst = np.max(erm_risks)
    tau = max(R_source + tau_target_fraction * max(0, R_worst - R_source), R_source + 1e-6)
    
    # Initialize weights
    w_t = w_init.copy()
    log_kappa = [np.inf]
    
    import time
    start_robust = time.perf_counter()
    
    for t in range(T_max_iter):
        # Calculate current risks
        class_risks = np.array([calculate_risk(w_t, A_matrices[j], b_vectors[j]) for j in range(K)])
        
        # Solve for kappa (fragility)
        kappa_t = solve_k_series(q_s, class_risks, tau, order=8)
        kappa_history.append(kappa_t)
        log_kappa.append(kappa_t)
        
        if kappa_t is None:
            break
        
        # Compute adversarial distribution
        q_star = compute_worst_case_q(class_risks, q_s, kappa_t)
        
        
        w_next, _ = solve_adversarial_weights_proximal(
            A_matrices, b_vectors, q_star, M, K, 
            w_prev=w_t,  # Pass current weights
            eta=eta       # Step size parameter
        )
        
       
        
        w_t = w_next

        # Convergence check
        if t > 0 and np.linalg.norm(w_t - w_prev_iter) < epsilon_convergence:
            break
        w_prev_iter = w_t.copy()
    
    robust_time = time.perf_counter() - start_robust

    return w_init, w_t, kappa_history, robust_time