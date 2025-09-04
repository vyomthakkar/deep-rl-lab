# tests/test_soft_value_iteration.py
import numpy as np
from src.rlx.envs.tabular.toy2state import build, closed_form_values
from src.rlx.algos.dp.soft_value_iteration import run_soft_vi, soft_bellman_backup


def test_soft_vi_toy2state_tau_limit():
    """Soft limit: τ→0 match to VI within 1e-4."""
    from src.rlx.algos.dp.value_iteration import run_vi
    
    gamma, p_loop, r_reward = 0.99, 0.5, 1.0
    mdp = build(gamma=gamma, p_loop=p_loop, r_reward=r_reward)
    
    # Regular VI result
    vi_result = run_vi(mdp, tol=1e-10, max_iters=1000, logger=None)
    
    # Soft VI with very low temperature (should approach VI)
    soft_result = run_soft_vi(mdp, tau=0.01, tol=1e-10, max_iters=1000, logger=None)
    
    # Values should be close to VI as tau -> 0
    assert np.max(np.abs(soft_result["V"] - vi_result["V"])) < 1e-4, \
        f"Soft VI with tau=0.01 doesn't match VI within 1e-4"


def test_soft_vi_entropy_increases_with_tau():
    """Entropy increases with τ."""
    from src.rlx.envs.tabular.gridworld import build_4room
    
    mdp = build_4room(gamma=0.99, slip=0.0)
    
    # Test different temperatures
    tau_low = 0.1
    tau_high = 1.0
    
    result_low = run_soft_vi(mdp, tau=tau_low, tol=1e-8, max_iters=1000, logger=None)
    result_high = run_soft_vi(mdp, tau=tau_high, tol=1e-8, max_iters=1000, logger=None)
    
    # Extract final entropy from logs
    entropy_low = result_low["logs"][-1]["entropy"]
    entropy_high = result_high["logs"][-1]["entropy"]
    
    # Higher temperature should yield higher entropy
    assert entropy_high > entropy_low, \
        f"Entropy doesn't increase with tau: {entropy_high} <= {entropy_low}"


def test_soft_vi_convergence_4room():
    """Test that Soft VI converges within reasonable iterations on 4room gridworld."""
    from src.rlx.envs.tabular.gridworld import build_4room
    
    mdp = build_4room(gamma=0.99, slip=0.0)
    result = run_soft_vi(mdp, tau=0.1, tol=1e-8, max_iters=1000, logger=None)
    
    # Should converge within 1000 iterations
    assert len(result["logs"]) < 1000, "Soft VI didn't converge within 1000 iterations"
    
    # Final residual should be below tolerance
    final_residual = result["logs"][-1]["delta"]
    assert final_residual < 1e-8, f"Final residual {final_residual} above tolerance 1e-8"


def test_soft_vi_entropy_positive():
    """Soft VI should have positive entropy (unlike deterministic VI)."""
    from src.rlx.envs.tabular.gridworld import build_4room
    
    mdp = build_4room(gamma=0.99, slip=0.0)
    result = run_soft_vi(mdp, tau=0.5, tol=1e-8, max_iters=1000, logger=None)
    
    # Entropy should be positive for soft policies
    final_entropy = result["logs"][-1]["entropy"]
    assert final_entropy > 0.0, f"Soft VI entropy should be positive, got {final_entropy}"


def test_soft_bellman_backup_shapes():
    """Test soft Bellman backup returns correct shapes."""
    from src.rlx.envs.tabular.gridworld import build_4room
    
    mdp = build_4room(gamma=0.99, slip=0.0)
    num_states, num_actions = mdp.P.shape[:2]
    
    V = np.zeros(num_states)
    V_next, pi_prob, Q = soft_bellman_backup(V, mdp.P, mdp.R, mdp.gamma, tau=0.1)
    
    # Check shapes
    assert V_next.shape == (num_states,), f"V_next shape {V_next.shape} != ({num_states},)"
    assert pi_prob.shape == (num_states, num_actions), \
        f"pi_prob shape {pi_prob.shape} != ({num_states}, {num_actions})"
    assert Q.shape == (num_states, num_actions), \
        f"Q shape {Q.shape} != ({num_states}, {num_actions})"
    
    # Policy probabilities should sum to 1 for each state
    prob_sums = np.sum(pi_prob, axis=1)
    assert np.allclose(prob_sums, 1.0), "Policy probabilities don't sum to 1 for each state"


def test_soft_vi_vs_vi_agreement_low_tau():
    """Soft VI with very low tau should match VI closely."""
    from src.rlx.algos.dp.value_iteration import run_vi
    from src.rlx.envs.tabular.toy2state import build
    
    gamma, p_loop, r_reward = 0.99, 0.5, 1.0
    mdp = build(gamma=gamma, p_loop=p_loop, r_reward=r_reward)
    
    vi_result = run_vi(mdp, tol=1e-10, max_iters=1000, logger=None)
    soft_result = run_soft_vi(mdp, tau=0.001, tol=1e-10, max_iters=1000, logger=None)
    
    # Values should match closely
    value_diff = np.max(np.abs(soft_result["V"] - vi_result["V"]))
    assert value_diff < 1e-3, f"Soft VI (tau=0.001) values differ from VI by {value_diff}"
    
    # Greedy policies should match (deterministic extraction should be same)
    assert np.all(soft_result["pi"] == vi_result["pi"]), \
        "Greedy policies from Soft VI and VI don't match at low tau"


def test_soft_vi_reproducibility():
    """Same config ⇒ identical outputs (deterministic slip=0 setup)."""
    from src.rlx.envs.tabular.gridworld import build_4room
    
    # Deterministic environment (no slip) with same seed
    mdp1 = build_4room(gamma=0.99, slip=0.0, seed=0)
    mdp2 = build_4room(gamma=0.99, slip=0.0, seed=0)
    
    result1 = run_soft_vi(mdp1, tau=0.1, tol=1e-8, max_iters=1000, logger=None)
    result2 = run_soft_vi(mdp2, tau=0.1, tol=1e-8, max_iters=1000, logger=None)
    
    # Results should be identical
    assert np.allclose(result1["V"], result2["V"]), "Soft VI values not reproducible"
    assert np.all(result1["pi"] == result2["pi"]), "Soft VI policies not reproducible"
    assert len(result1["logs"]) == len(result2["logs"]), "Soft VI convergence not reproducible"


def test_soft_vi_temperature_extremes():
    """Test behavior at temperature extremes."""
    from src.rlx.envs.tabular.toy2state import build
    
    gamma, p_loop, r_reward = 0.99, 0.5, 1.0
    mdp = build(gamma=gamma, p_loop=p_loop, r_reward=r_reward)
    num_actions = mdp.P.shape[1]
    
    # Very high temperature should approach uniform policy (high entropy)
    result_high_tau = run_soft_vi(mdp, tau=10.0, tol=1e-8, max_iters=1000, logger=None)
    final_entropy_high = result_high_tau["logs"][-1]["entropy"]
    
    # Maximum possible entropy for uniform distribution over actions
    max_entropy = -num_actions * (1.0/num_actions) * np.log(1.0/num_actions)
    
    # Should be close to maximum entropy (within some tolerance for convergence)
    assert final_entropy_high > 0.8 * max_entropy, \
        f"High tau doesn't approach uniform policy: entropy {final_entropy_high} vs max {max_entropy}"


def test_soft_vi_policy_l1_change():
    """Test that policy L1 change is computed correctly for probability distributions."""
    from src.rlx.envs.tabular.toy2state import build
    
    gamma, p_loop, r_reward = 0.99, 0.5, 1.0
    mdp = build(gamma=gamma, p_loop=p_loop, r_reward=r_reward)
    
    result = run_soft_vi(mdp, tau=0.5, tol=1e-8, max_iters=50, logger=None)  # Stop early to see changes
    
    # Policy L1 changes should be positive in early iterations (policies are changing)
    early_changes = [log["policy_l1_change"] for log in result["logs"][:5]]
    
    # At least some early iterations should show policy changes
    assert any(change > 0 for change in early_changes), \
        "No policy changes detected in early Soft VI iterations"
    
    # Changes should generally decrease over time (convergence)
    if len(result["logs"]) > 10:
        early_avg = np.mean([log["policy_l1_change"] for log in result["logs"][:5]])
        late_avg = np.mean([log["policy_l1_change"] for log in result["logs"][-5:]])
        assert late_avg <= early_avg, \
            f"Policy changes should decrease over time: {late_avg} > {early_avg}"