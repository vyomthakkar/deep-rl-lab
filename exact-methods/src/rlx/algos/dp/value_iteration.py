import numpy as np


def bellman_backup(V, P, R, gamma):
    """One-step Bellman optimality backup.

    Computes Q[s,a] = R[s,a] + gamma * sum_{s'} P[s,a,s'] * V[s'] and
    V_next[s] = max_a Q[s,a].

    Args:
        V: Array of state values.
        P: Transition probabilities per (s, a, s').
        R: Immediate rewards per (s, a).
        gamma: Discount factor in [0, 1).

    Returns:
        V_next: Backed-up state values.
        Q: Action-values for each (s, a).
    """
    # Expected next-state value for each (s,a):
    # EV[s,a] = sum_{s'} P[s,a,s'] * V[s']
    EV = P @ V                  #shape: (S, A)
    Q = R + gamma * EV          #shape: (S, A)
    V_next = Q.max(axis=1)      #shape: (S)
    return V_next, Q
    
    

def run_vi(mdp, tol: float, max_iters: int, logger) -> dict:
    """Returns {'V': V, 'Q': Q, 'pi': pi, 'logs': history}"""
    num_states = len(mdp.state_names)
    V_prev = np.zeros(num_states)
    V = np.zeros(num_states)
    
    for _ in range(max_iters):
        pass
    

