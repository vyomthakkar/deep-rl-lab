# tests/test_value_iteration.py
import numpy as np
from src.rlx.envs.tabular.toy2state import build, closed_form_values
from src.rlx.algos.dp.value_iteration import run_vi

def test_vi_toy2state_closed_form():
    """Test VI correctness (Toy2State): matches closed form within 1e-8."""
    gamma, p_loop, r_reward = 0.99, 0.5, 1.0
    mdp = build(gamma=gamma, p_loop=p_loop, r_reward=r_reward)
    v_cf = closed_form_values(gamma, p_loop, r_reward)
    
    result = run_vi(mdp, tol=1e-8, max_iters=1000, logger=None)
    
    # Check convergence within tolerance
    assert np.max(np.abs(result["V"] - v_cf)) < 1e-8, f"VI values don't match closed form within 1e-8"
    
def test_vi_convergence_4room():
    """Test that VI converges within reasonable iterations on 4room gridworld."""
    from src.rlx.envs.tabular.gridworld import build_4room
    
    mdp = build_4room(gamma=0.99, slip=0.0)
    result = run_vi(mdp, tol=1e-8, max_iters=1000, logger=None)
    
    # Should converge within 1000 iterations
    assert len(result["logs"]) < 1000, "VI didn't converge within 1000 iterations"
    
    # Final residual should be below tolerance
    final_residual = result["logs"][-1]["delta"]
    assert final_residual < 1e-8, f"Final residual {final_residual} above tolerance 1e-8"
    
def test_bellman_monotonicity():
    """Test monotone Bellman: max_a Q_k(s,a) non-decreasing across VI sweeps."""
    from src.rlx.envs.tabular.gridworld import build_4room
    from src.rlx.algos.dp.value_iteration import bellman_backup
    
    mdp = build_4room(gamma=0.99, slip=0.0)
    V = np.zeros(len(mdp.state_names))
    
    prev_max_q = -np.inf
    for _ in range(10):  # Test first 10 iterations
        V_next, pi_next, Q = bellman_backup(V, mdp.P, mdp.R, mdp.gamma)
        max_q = np.max(Q)
        
        # Allow small numerical noise (1e-8)
        assert max_q >= prev_max_q - 1e-8, f"Bellman backup not monotone: {max_q} < {prev_max_q}"
        
        prev_max_q = max_q
        V = V_next