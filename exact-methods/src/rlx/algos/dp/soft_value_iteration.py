import numpy as np
import time


def soft_bellman_backup(V, P, R, gamma, tau):
    """
    max-ent bellman backup
    """
    
    EV = P @ V
    Q = R + gamma * EV
    Q_scaled = Q/tau
    max_Q = Q_scaled.max(axis=1, keepdims=True)
    Q_exp_shifted = np.exp(Q_scaled - max_Q)
    V_next = tau * (max_Q.squeeze() + np.log(np.sum(Q_exp_shifted, axis=1)))
    pi_prob = Q_exp_shifted/np.sum(Q_exp_shifted, axis=1, keepdims=True)
    
    return V_next, pi_prob, Q
    



def run_soft_vi(mdp, tau: float, tol: float, max_iters: int, logger) -> dict:
    """Soft Value Iteration (SVI) for tabular MDPs.
    """
    
    #Input validation
    assert tau > 0, "Temperature must be positive"
    assert tol > 0, "Tolerance must be positive"
    assert max_iters > 0, "Max iterations must be positive"
    assert 0 <= mdp.gamma < 1, "Discount must be in [0,1)"
    assert np.allclose(mdp.P.sum(axis=-1), 1.0), "Transitions must be stochastic"
    
    num_states = mdp.P.shape[0]
    num_actions = mdp.P.shape[1]
    V = np.zeros(num_states, dtype=np.float64)
    pi_prob = np.ones((num_states, num_actions))/num_actions #init to uniform prob distribution
    pi_det = np.zeros(num_states, dtype=np.int64)
    logs = []
    
    start_time = time.time()
    
    for i in range(max_iters):
        V_next, pi_prob_next, Q = soft_bellman_backup(V, mdp.P, mdp.R, mdp.gamma, tau)
        pi_det = pi_prob_next.argmax(axis=1)
        
        #monitoring metrics
        delta = np.max(np.abs(V_next - V)).item()
        policy_l1_change = np.sum(np.abs(pi_prob - pi_prob_next))
        entropy = -np.sum(pi_prob_next * np.log(pi_prob_next + 1e-8))
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
        
        
