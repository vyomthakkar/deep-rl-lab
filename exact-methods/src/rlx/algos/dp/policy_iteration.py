import numpy as np
import time


def policy_evaluation_sweep(pi, V, P, R, gamma):
    """One Jacobi-style policy-evaluation sweep for a deterministic policy.

    Computes one application of the policy-evaluation operator T_pi on V:
        V_next[s] = R[s, pi[s]] + gamma * sum_{s'} P[s, pi[s], s'] * V[s'].

    Implemented by forming Q = R + gamma * (P @ V) and selecting the action
    prescribed by pi for each state.

    Args:
        pi: Integer array of shape (S,) with action indices per state.
        V: Float array of shape (S,) with current state-value estimates.
        P: Float array of shape (S, A, S), row-stochastic transitions.
        R: Float array of shape (S, A), expected immediate rewards.
        gamma: Discount factor in [0, 1).

    Returns:
        V_next: Float array of shape (S,) after one T_pi sweep.

    Notes:
        Terminal handling is expected via the MDP construction (e.g.,
        absorbing terminals that self-loop with zero reward). This function
        does not special-case terminals.
    """
    num_states = P.shape[0]
    
    # Expected next-state value for each (s,a): EV[s,a] = sum_{s'} P[s,a,s'] * V[s']
    EV = P @ V                                              # shape: (S, A)
    Q = R + gamma * EV                                      # shape: (S, A)
    
    # Select Q-values according to current policy pi
    V_next = Q[np.arange(num_states), pi]                  # shape: (S,)
    
    return V_next
    
            


def run_pi(mdp, eval_tol: float, max_eval_iters: int, logger) -> dict:
    """Deterministic Policy Iteration (PI) for tabular MDPs.

    Procedure:
      1) Policy evaluation: perform up to `max_eval_iters` Jacobi sweeps on V
         under the current policy pi, stopping early when
         ||T_pi V - V||_inf < eval_tol.
      2) Policy improvement: set
            pi[s] = argmax_a [ R[s,a] + gamma * sum_{s'} P[s,a,s'] * V[s'] ].
         Repeat until the policy stops changing.

    Args:
        mdp: Tabular MDP with P in R^{SxAxS}, R in R^{SxA}, discount gamma in [0,1).
        eval_tol: Convergence threshold for the policy-evaluation residual.
        max_eval_iters: Maximum evaluation sweeps per outer PI iteration.
        logger: Optional logger with .info(str) for progress logs.

    Returns:
        A dict with:
            - 'V': Final state values, shape (S,).
            - 'Q': Action-values R + gamma * (P @ V) at convergence, shape (S, A).
            - 'pi': Greedy policy indices per state, shape (S,).
            - 'logs': List of per-outer-iteration metrics each containing
                      'outer_iter', 'inner_iter', 'delta' (== bellman_residual),
                      'policy_l1_change', 'entropy', 'wall_clock_time'; and a
                      final record {'total_run_time': seconds}.

    Notes:
        - Ties in the greedy improvement are broken by numpy.argmax's
          first-maximum rule (deterministic given Q).
        - 'bellman_residual' here refers to the policy-evaluation residual
          ||T_pi V - V||_inf, not the optimality residual.
    """
    # Input validation
    assert eval_tol > 0, "Tolerance must be positive"
    assert max_eval_iters > 0, "Max iterations must be positive"
    assert 0 <= mdp.gamma < 1, "Discount must be in [0,1)"
    assert np.allclose(mdp.P.sum(axis=-1), 1.0), "Transitions must be stochastic"
    
    num_states = mdp.P.shape[0]
    V = np.zeros(num_states, dtype=np.float64)
    # pi = np.zeros(num_states, dtype=np.int64)  # TODO(human): Consider greedy initialization: pi = mdp.R.argmax(axis=1).astype(np.int64)
    pi = mdp.R.argmax(axis=1).astype(np.int64)
    policy_l1_change = np.inf
    logs = []
    
    start_time = time.time()
    
    outer_iter = 0
    
    while policy_l1_change != 0:
        # Policy evaluation phase
        inner_iter = 0
        for _ in range(max_eval_iters):
            V_next = policy_evaluation_sweep(pi, V, mdp.P, mdp.R, mdp.gamma)
            delta = np.max(np.abs(V_next - V)).item()       # Policy evaluation residual
            V = V_next
            inner_iter += 1
            if delta < eval_tol:
                break
            
        # Policy improvement phase
        EV = mdp.P @ V                                      # shape: (S, A)
        Q = mdp.R + mdp.gamma * EV                          # shape: (S, A)
        # TODO(human): Implement Howard's improvement tie-breaking
        # Current: Q.argmax(axis=1) - always picks first max (can cause oscillation)
        # Should be: When Q[s,a] ≈ Q[s,pi[s]], keep current action pi[s] to avoid unnecessary changes
        # This prevents policy oscillation when multiple actions are equally good
        # Suggested approach: np.where(Q.max(axis=1, keepdims=True) - Q[np.arange(S), pi] < 1e-12, pi, Q.argmax(axis=1))
        # pi_next = Q.argmax(axis=1).astype(np.int64)         # Greedy policy, shape: (S,)
        pi_next = np.where(Q.max(axis=1, keepdims=True) - Q[np.arange(S, pi)] < 1e-12, pi, Q.argmax(axis=1)) #Howard's improvement tie-breaking
        policy_l1_change = np.sum(pi_next != pi).item()     # Number of states with policy change
        pi = pi_next
        # Monitoring metrics
        wall_clock_time = time.time() - start_time
        
        # TODO(human): Add PI-specific backup counting and tighten inner tolerance
        # Current issues:
        # 1. Inner tolerance (eval_tol) might be too loose, causing more outer iterations
        # 2. Not tracking total evaluation sweeps vs. outer policy improvements
        # 3. Should count: inner_iter * num_states for evaluation + num_states * num_actions for improvement
        # 4. Consider: eval_tol = tol / 10 for tighter inner convergence
        # 5. Alternative: Implement direct linear solve V = (I - γP^π)^{-1} r^π
        
        logs.append({
            "outer_iter": outer_iter,
            "inner_iter": inner_iter,
            "delta": delta,                                # Policy evaluation residual
            "bellman_residual": delta,                     # Same as delta for PI
            "policy_l1_change": policy_l1_change,
            "entropy": 0.0,                                # Always 0.0 for deterministic PI policy
            "wall_clock_time": wall_clock_time,
            # TODO(human): Add "bellman_backups": total_eval_sweeps * num_states + outer_iter * num_states * num_actions
            "bellman_backups": (i + 1) * num_states * num_actions,
            # Standardized fields across algorithms
            "iter": int(outer_iter),
            "algo": "pi",
            "gamma": float(mdp.gamma),
            "max_q": float(np.max(Q)),
        })
        
        outer_iter += 1
        
        if logger:
            logger.info(f"Outer iter: {outer_iter}, inner iter: {inner_iter}, delta: {delta:.2e}, policy_l1_change: {policy_l1_change}")
        
        
    end_time = time.time()
    run_time = end_time - start_time
    
    if logger:
        logger.info(f"Converged after {outer_iter} iterations")
    
    return {
        "V": V,
        "Q": Q,
        "pi": pi,
        "logs": logs,
        "run_time": run_time,
        "converged": True
    }
