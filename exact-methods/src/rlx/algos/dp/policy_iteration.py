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
    
            

def policy_evaluation_linear_solve(pi, P, R, gamma):
    """Exact policy evaluation via linear solve.

    Solves (I - gamma * P_pi) V = R_pi for V, where P_pi[s, s'] = P[s, pi[s], s']
    and R_pi[s] = R[s, pi[s]]. Returns V and the policy-evaluation residual
    ||T_pi V - V||_inf computed post-solve.

    Args:
        pi: Integer array of shape (S,) with action indices per state.
        P: Float array of shape (S, A, S), row-stochastic transitions.
        R: Float array of shape (S, A), expected immediate rewards.
        gamma: Discount factor in [0, 1).

    Returns:
        V: Float array of shape (S,) solving the linear system.
        delta: Float, policy-evaluation residual ||T_pi V - V||_inf.
    """
    num_states = P.shape[0]
    rows = np.arange(num_states)

    # Construct policy-specific dynamics and rewards
    P_pi = P[rows, pi, :]                                   # shape: (S, S)
    R_pi = R[rows, pi]                                      # shape: (S,)

    # Solve the linear system (I - gamma P_pi) V = R_pi
    A = np.eye(num_states, dtype=np.float64) - gamma * P_pi
    b = R_pi
    V = np.linalg.solve(A, b)

    # Compute residual ||T_pi V - V||_inf for logging
    EV = P @ V                                              # shape: (S, A)
    Q = R + gamma * EV
    V_tp = Q[rows, pi]
    delta = np.max(np.abs(V_tp - V)).item()

    return V, delta



def run_pi(mdp, eval_tol: float, max_eval_iters: int, logger, use_optimizations: bool = False, opt_config: dict = {}, use_linear_solve_eval: bool = False) -> dict: 
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
        use_linear_solve_eval: If True, evaluate the policy exactly in one step
            by solving (I - gamma P_pi) V = R_pi instead of doing Jacobi sweeps.

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
    num_actions = mdp.P.shape[1]
    V = np.zeros(num_states, dtype=np.float64)
    
    pi = np.zeros(num_states, dtype=np.int64) # Zero initialization (original)
    if use_optimizations and opt_config.get("greedy_init", False):
        pi = mdp.R.argmax(axis=1).astype(np.int64)  # Greedy initialization
           
    
    policy_l1_change = np.inf
    logs = []
    
    start_time = time.time()
    
    outer_iter = 0
    
    while policy_l1_change != 0:
        # Policy evaluation phase
        inner_iter = 0
        if use_linear_solve_eval:
            # Exact evaluation via linear solve
            V, delta = policy_evaluation_linear_solve(pi, mdp.P, mdp.R, mdp.gamma)
            # No Jacobi sweeps performed; inner_iter stays 0
        else:
            for _ in range(max_eval_iters):
                V_next = policy_evaluation_sweep(pi, V, mdp.P, mdp.R, mdp.gamma)
                delta = np.max(np.abs(V_next - V)).item()   # Policy evaluation residual
                V = V_next
                inner_iter += 1
                if delta < eval_tol:
                    break
            
        # Policy improvement phase
        EV = mdp.P @ V                                      # shape: (S, A)
        Q = mdp.R + mdp.gamma * EV                          # shape: (S, A)
        
        
        if use_optimizations and opt_config.get("howards_improvement", False):
            # Howard's improvement: keep current action when tied
            pi_next = np.where(Q.max(axis=1) - Q[np.arange(num_states), pi] < 1e-12, 
                               pi, Q.argmax(axis=1)).astype(np.int64)
        else:
            # Naive greedy: always pick first maximum
            pi_next = Q.argmax(axis=1).astype(np.int64)
        
        policy_l1_change = np.sum(pi_next != pi).item()     # Number of states with policy change
        pi = pi_next
        
        # Monitoring metrics
        wall_clock_time = time.time() - start_time
        
        # Track hardware-independent cost: evaluation + improvement
        total_eval_backups = inner_iter * num_states
        improvement_backups = num_states * mdp.P.shape[1]  # num_actions
        backup_count = total_eval_backups + improvement_backups
        
        #Track more refined return
        terminal_states = mdp.terminal_mask
        non_terminal_states = ~terminal_states
        
        average_return = V[non_terminal_states].mean()
        
        logs.append({
            "outer_iter": outer_iter,
            "inner_iter": inner_iter,
            "delta": delta,                                # Policy evaluation residual
            "bellman_residual": delta,                     # Same as delta for PI
            "policy_l1_change": policy_l1_change,
            "entropy": 0.0,                                # Always 0.0 for deterministic PI policy
            "wall_clock_time": wall_clock_time,
            "bellman_backups": backup_count,
            "average_return": float(average_return),
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
        "total_bellman_backups": sum(log["bellman_backups"] for log in logs),
        "run_time": run_time,
        "converged": True
    }
