import numpy as np
import time


def soft_bellman_backup(V, P, R, gamma, tau):
    """One-step soft (max-entropy) Bellman backup.

    Computes the max-entropy Bellman operator with temperature tau:
      - Q[s,a] = R[s,a] + gamma * sum_{s'} P[s,a,s'] * V[s']
      - V_next[s] = tau * log(sum_a exp(Q[s,a] / tau))
      - pi[a|s] = exp(Q[s,a] / tau) / sum_{a'} exp(Q[s,a'] / tau)
    
    Uses logsumexp trick for numerical stability: 
    log(sum_a exp(x_a)) = max_a(x_a) + log(sum_a exp(x_a - max_a(x_a)))
    
    Note: P[s,a,s'] = P(s'|s,a)

    Args:
        V: Array of state values, shape (S,).
        P: Transition probabilities per (s, a, s'), shape (S, A, S).
        R: Immediate rewards per (s, a), shape (S, A).
        gamma: Discount factor in [0, 1).
        tau: Temperature parameter, tau > 0. Lower tau -> more deterministic.

    Returns:
        V_next: Soft-backed-up state values, shape (S,).
        pi_prob: Soft policy probabilities per (s, a), shape (S, A).
        Q: Action-values for each (s, a), shape (S, A).
    """
    # Expected next-state value for each (s,a):
    # EV[s,a] = sum_{s'} P[s,a,s'] * V[s']
    EV = P @ V                                              # shape: (S, A)
    Q = R + gamma * EV                                      # shape: (S, A)
    
    # Apply temperature scaling and logsumexp trick for numerical stability
    Q_scaled = Q / tau                                      # shape: (S, A)
    max_Q = Q_scaled.max(axis=1, keepdims=True)            # shape: (S, 1)
    Q_exp_shifted = np.exp(Q_scaled - max_Q)               # shape: (S, A)
    
    # Soft value function: V[s] = tau * logsumexp(Q[s,:] / tau)
    V_next = tau * (max_Q.squeeze() + np.log(np.sum(Q_exp_shifted, axis=1)))
    
    # Soft policy: pi[a|s] = exp(Q[s,a] / tau) / sum_a' exp(Q[s,a'] / tau)
    pi_prob = Q_exp_shifted / np.sum(Q_exp_shifted, axis=1, keepdims=True)
    
    return V_next, pi_prob, Q
    



def run_soft_vi(mdp, tau: float, tol: float, max_iters: int, logger) -> dict:
    """Soft Value Iteration (SVI) for tabular MDPs with entropy regularization.

    Implements the soft Bellman optimality operator to find the optimal policy
    for the entropy-regularized objective:
        maximize E[sum_t gamma^t (R_t + tau * H(pi(·|s_t)))]
    where H(pi) is the entropy of policy pi.

    Procedure:
      1) Initialize V = 0 and uniform policy pi
      2) Repeat until convergence:
         - Apply soft Bellman backup: V_next = T^soft V
         - Extract soft policy: pi[a|s] ∝ exp(Q[s,a] / tau)
         - Check convergence: ||V_next - V||_inf < tol

    Args:
        mdp: Tabular MDP with P in R^{SxAxS}, R in R^{SxA}, discount gamma in [0,1).
        tau: Temperature parameter, tau > 0. Controls exploration vs exploitation.
             Higher tau -> more exploration (uniform policy limit).
             Lower tau -> more exploitation (deterministic policy limit).
        tol: Convergence threshold for the value function residual.
        max_iters: Maximum iterations before giving up.
        logger: Optional logger with .info(str) for progress logs.

    Returns:
        A dict with:
            - 'V': Final state values, shape (S,).
            - 'Q': Action-values R + gamma * (P @ V) at convergence, shape (S, A).
            - 'pi': Deterministic policy extracted via argmax, shape (S,).
            - 'logs': List of per-iteration metrics each containing
                      'delta' (== bellman_residual), 'policy_l1_change', 
                      'entropy', 'wall_clock_time', and standardized fields.
            - 'run_time': Total wall-clock time in seconds.

    Notes:
        - Returns deterministic 'pi' for API consistency, but the true policy
          is soft (probabilistic) during optimization.
        - 'entropy' measures the entropy of the current soft policy.
        - As tau -> 0, this converges to standard value iteration.
        - As tau -> inf, this converges to uniform random policy.
    """
    # Input validation
    assert tau > 0, "Temperature must be positive"
    assert tol > 0, "Tolerance must be positive"
    assert max_iters > 0, "Max iterations must be positive"
    assert 0 <= mdp.gamma < 1, "Discount must be in [0,1)"
    assert np.allclose(mdp.P.sum(axis=-1), 1.0), "Transitions must be stochastic"
    
    num_states = mdp.P.shape[0]
    num_actions = mdp.P.shape[1]
    V = np.zeros(num_states, dtype=np.float64)
    pi_prob = np.ones((num_states, num_actions)) / num_actions  # Initialize to uniform distribution
    pi_det = np.zeros(num_states, dtype=np.int64)
    logs = []
    
    start_time = time.time()
    
    # TODO(human): Debug convergence issues
    # Current soft VI doesn't converge to regular VI as tau->0
    # Need to investigate: backup equation, initialization, or numerical issues?
    for i in range(max_iters):
        V_next, pi_prob_next, Q = soft_bellman_backup(V, mdp.P, mdp.R, mdp.gamma, tau)
        
        # Extract deterministic policy for consistent API (argmax of soft policy)
        pi_det = pi_prob_next.argmax(axis=1)                   # shape: (S,)
        
        # Monitoring metrics
        delta = np.max(np.abs(V_next - V)).item()               # Bellman residual ||T*V - V||_inf
        policy_l1_change = np.sum(np.abs(pi_prob - pi_prob_next)) # L1 change in policy distributions
        entropy = -np.sum(pi_prob_next * np.log(pi_prob_next + 1e-8))  # Policy entropy H(pi)
        wall_clock_time = time.time() - start_time
        
        logs.append({
            "i": i,
            "delta": delta,
            "bellman_residual": delta,
            "policy_l1_change": policy_l1_change,
            "entropy": entropy,
            "wall_clock_time": wall_clock_time,
            "iter": int(i),
            "algo": "soft_vi",
            "gamma": float(mdp.gamma),
            "max_q": float(np.max(Q)),
        })
        
        if logger:
            logger.info(f"Iter {i}: delta={delta:.2e}, "
                         f"policy_changes={policy_l1_change}")
            
        V = V_next
        pi_prob = pi_prob_next
        
        if delta < tol:
            break
        
    end_time = time.time()
    run_time = end_time - start_time
    
    if logger:
        logger.info(f"Converged after {i+1} iterations")
        
    return {
        "V": V,
        "Q": Q,
        "pi": pi_det,
        "logs": logs,
        "run_time": run_time
    }
        
        
