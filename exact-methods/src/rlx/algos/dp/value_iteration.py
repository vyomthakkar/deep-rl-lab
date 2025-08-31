import numpy as np
import time

def bellman_backup(V, P, R, gamma):
    """One-step Bellman optimality backup.

    Computes:
      - Q[s,a] = R[s,a] + gamma * sum_{s'} P[s,a,s'] * V[s']
      - V_next[s] = max_a Q[s,a]
      - Pi[s] = argmax_a Q[s,a] (greedy policy w.r.t. Q)
    
    Note: P[s,a,s'] = P(s'|s,a)

    Args:
        V: Array of state values, shape (S,).
        P: Transition probabilities per (s, a, s'), shape (S, A, S).
        R: Immediate rewards per (s, a), shape (S, A).
        gamma: Discount factor in [0, 1).

    Returns:
        V_next: Backed-up state values, shape (S,).
        Pi: Greedy policy indices per state (ties broken by first max), shape (S,).
        Q: Action-values for each (s, a), shape (S, A).
    """
    # Expected next-state value for each (s,a):
    # EV[s,a] = sum_{s'} P[s,a,s'] * V[s']
    EV = P @ V                  #shape: (S, A)
    Q = R + gamma * EV          #shape: (S, A)
    V_next = Q.max(axis=1)      #shape: (S)
    pi = Q.argmax(axis=1)       #shape: (S)
    return V_next, pi, Q
    
    

def run_vi(mdp, tol: float, max_iters: int, logger) -> dict:
    """Returns {'V': V, 'Q': Q, 'pi': pi, 'logs': history}"""
    
    #Input validation
    assert tol > 0, "Tolerance must be positive"
    assert max_iters > 0, "Max iterations must be positive"
    assert 0 <= mdp.gamma < 1, "Discount must be in [0,1)"
    assert np.allclose(mdp.P.sum(axis=-1), 1.0), "Transitions must be stochastic"

    num_states = mdp.P.shape[0]
    V = np.zeros(num_states, dtype=np.float64)
    pi = np.zeros(num_states, dtype=np.int64)
    logs = []
    
    start_time = time.time()
    
    for i in range(max_iters):
        V_next, pi_next, Q = bellman_backup(V, mdp.P, mdp.R, mdp.gamma)
        
        #monitoring metrics
        delta = np.max(np.abs(V_next - V)).item()
        policy_l1_change = np.sum(pi_next != pi).item()  # Convert to Python int
        wall_clock_time = time.time() - start_time
        
        logs.append({
            "i": i,
            "delta": delta,
            "bellman_residual": delta,  # For VI: ||T*V - V|| = ||V_next - V||
            "policy_l1_change": policy_l1_change,
            "entropy": 0.0,  # Always 0.0 for deterministic VI policy
            "wall_clock_time": wall_clock_time
        })
        
        if logger:
            logger.info(f"Iter {i}: delta={delta:.2e}, "
                         f"policy_changes={policy_l1_change}")
        
        V = V_next
        pi = pi_next
        
        if delta < tol:
            break
        
    end_time = time.time()
    run_time = end_time - start_time
    logs.append({"total_run_time": run_time})
    
    if logger:
        logger.info(f"Converged after {i+1} iterations")
    
        
    return {
        "V": V,
        "Q": Q,
        "pi": pi,
        "logs": logs
    }
        
        
        
    

