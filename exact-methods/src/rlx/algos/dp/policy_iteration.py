import numpy as np
import time


def simplified_bellman_update(pi, V, P, R, gamma):
    num_states = V.shape[0]
    
    EV = P @ V
    Q = R + gamma * EV
    V_next = Q[np.arange(num_states), pi]
    
    return V_next
    
            


def run_pi(mdp, eval_tol: float, max_eval_iters: int, logger) -> dict:
    
    #Input validation
    assert eval_tol > 0, "Tolerance must be positive"
    assert max_eval_iters > 0, "Max iterations must be positive"
    assert 0 <= mdp.gamma < 1, "Discount must be in [0,1)"
    assert np.allclose(mdp.P.sum(axis=-1), 1.0), "Transitions must be stochastic"
    
    num_states = len(mdp.state_names)
    V = np.zeros(num_states, dtype=np.float64)
    pi = np.zeros(num_states, dtype=np.int64)
    policy_l1_change = np.inf
    logs = []
    
    start_time = time.time()
    
    outer_iter = 0
    
    while policy_l1_change != 0:
        
        #policy evalutation
        inner_iter = 0
        for _ in range(max_eval_iters):
            V_next = simplified_bellman_update(pi, V, mdp.P, mdp.R, mdp.gamma)
            delta = np.max(np.abs(V_next - V)).item()
            V = V_next
            if delta < eval_tol:
                break
            inner_iter += 1
            
        #policy improvement
        EV = mdp.P @ V
        Q = mdp.R + mdp.gamma * EV
        pi_next = Q.argmax(axis=1).astype(np.int64)
        policy_l1_change = np.sum(pi_next != pi).item()
        pi = pi_next
        
        #monitoring metrics
        wall_clock_time = time.time() - start_time
        outer_iter += 1
        
        logs.append({
            "outer_iter": outer_iter,
            "inner_iter": inner_iter,
            "delta": delta,
            "bellman_residual": delta,
            "policy_l1_change": policy_l1_change,
            "entropy": 0.0,
            "wall_clock_time": wall_clock_time
        })
        
        if logger:
            logger.info(f"Outer iter: {outer_iter}, inner iter: {inner_iter}, delta: {delta:.2e}, policy_l1_change: {policy_l1_change}")
        
        
    end_time = time.time()
    run_time = end_time - start_time
    logs.append({"total_run_time": run_time})
    
    if logger:
        logger.info(f"Converged after {outer_iter} iterations")
    
    return {
        "V": V,
        "Q": Q,
        "pi": pi,
        "logs": logs,
    }
    

    
        
        
        
        
        
        
        
            
    
        
    
    
    
    
    
    
        
    
    
