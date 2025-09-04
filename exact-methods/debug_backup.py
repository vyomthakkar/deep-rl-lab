#!/usr/bin/env python3

import sys
sys.path.insert(0, 'src')

import numpy as np
from src.rlx.envs.tabular.toy2state import build, closed_form_values
from src.rlx.algos.dp.soft_value_iteration import soft_bellman_backup
from src.rlx.algos.dp.value_iteration import bellman_backup

def debug_backup():
    print("=== Debugging Bellman Backup ===")
    
    # Simple test case
    gamma, p_loop, r_reward = 0.99, 0.5, 1.0
    mdp = build(gamma=gamma, p_loop=p_loop, r_reward=r_reward)
    
    print(f"MDP P shape: {mdp.P.shape}")
    print(f"MDP R: {mdp.R}")
    print(f"MDP gamma: {mdp.gamma}")
    
    # Test convergence at the backup level
    V = np.zeros(2)
    tau = 0.01  # Very low temperature
    
    print(f"\n--- Testing backup convergence (tau={tau}) ---")
    for i in range(20):
        V_next_soft, pi_prob, Q_soft = soft_bellman_backup(V, mdp.P, mdp.R, mdp.gamma, tau)
        V_next_hard, pi_hard, Q_hard = bellman_backup(V, mdp.P, mdp.R, mdp.gamma)
        
        delta_soft = np.max(np.abs(V_next_soft - V))
        delta_hard = np.max(np.abs(V_next_hard - V))
        
        print(f"Iter {i:2d}: V_soft={V_next_soft}, delta_soft={delta_soft:.6f}, delta_hard={delta_hard:.6f}")
        print(f"         pi_prob={pi_prob[0]}")  # Show probabilities for state 0
        
        V = V_next_soft
        
        if delta_soft < 1e-6:
            break
    
    # Compare final values
    print(f"\nFinal soft V: {V}")
    
    # Test with regular VI
    V_hard = np.zeros(2)
    for i in range(50):
        V_next_hard, _, _ = bellman_backup(V_hard, mdp.P, mdp.R, mdp.gamma)
        if np.max(np.abs(V_next_hard - V_hard)) < 1e-8:
            break
        V_hard = V_next_hard
    
    print(f"Final hard V: {V_hard}")
    print(f"Closed form V: {closed_form_values(gamma, p_loop, r_reward)}")
    
    # Check if the backup is correct by manual calculation
    print(f"\n--- Manual verification ---")
    V_test = np.array([2.0, 0.0])  # Test values
    
    # Manual soft backup
    EV = mdp.P @ V_test
    Q = mdp.R + mdp.gamma * EV
    print(f"Q values: {Q}")
    
    Q_scaled = Q / tau
    max_Q = Q_scaled.max(axis=1, keepdims=True)
    Q_exp_shifted = np.exp(Q_scaled - max_Q)
    V_manual = tau * (max_Q.squeeze() + np.log(np.sum(Q_exp_shifted, axis=1)))
    
    print(f"Manual V: {V_manual}")
    
    # Compare with function
    V_func, _, _ = soft_bellman_backup(V_test, mdp.P, mdp.R, mdp.gamma, tau)
    print(f"Function V: {V_func}")
    print(f"Match: {np.allclose(V_manual, V_func)}")

if __name__ == "__main__":
    debug_backup()