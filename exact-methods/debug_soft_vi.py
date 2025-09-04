#!/usr/bin/env python3

import sys
sys.path.insert(0, 'src')

import numpy as np
from src.rlx.envs.tabular.toy2state import build
from src.rlx.algos.dp.soft_value_iteration import run_soft_vi
from src.rlx.algos.dp.value_iteration import run_vi

def debug_soft_vi():
    print("=== Debugging Soft VI ===")
    
    # Simple test case
    gamma, p_loop, r_reward = 0.99, 0.5, 1.0
    mdp = build(gamma=gamma, p_loop=p_loop, r_reward=r_reward)
    
    print(f"MDP shape: {mdp.P.shape}")
    print(f"Gamma: {mdp.gamma}")
    
    # Test with high tolerance first to see if it converges at all
    print("\n--- Testing Soft VI convergence (tau=0.1, tol=1e-4) ---")
    result = run_soft_vi(mdp, tau=0.1, tol=1e-4, max_iters=100, logger=None)
    
    print(f"Converged in {len(result['logs'])} iterations")
    print(f"Final values: {result['V']}")
    print(f"Final policy: {result['pi']}")
    
    # Show convergence trajectory
    print("\nFirst 10 iterations:")
    for i, log in enumerate(result['logs'][:10]):
        print(f"Iter {i}: delta={log['delta']:.6e}, entropy={log['entropy']:.4f}")
    
    if len(result['logs']) > 10:
        print("\nLast 5 iterations:")
        for log in result['logs'][-5:]:
            print(f"Iter {log['i']}: delta={log['delta']:.6e}, entropy={log['entropy']:.4f}")
    
    # Compare with VI
    print("\n--- Regular VI for comparison ---")
    vi_result = run_vi(mdp, tol=1e-8, max_iters=1000, logger=None)
    print(f"VI converged in {len(vi_result['logs'])} iterations")
    print(f"VI values: {vi_result['V']}")
    print(f"VI policy: {vi_result['pi']}")
    
    # Show difference
    value_diff = np.abs(result['V'] - vi_result['V'])
    print(f"\nValue difference: {value_diff}")
    print(f"Max value difference: {np.max(value_diff)}")

if __name__ == "__main__":
    debug_soft_vi()