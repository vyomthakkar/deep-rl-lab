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
    V = np.zeros(num_states, dtype=np.float64)
    pi = np.zeros(num_states, dtype=np.int64)
    logs = []
    
    start_time = time.time()
    
    # for i in range(max_iters):
        
