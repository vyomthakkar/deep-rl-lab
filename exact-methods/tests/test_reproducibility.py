import numpy as np


def test_vi_pi_reproducibility_4room():
    """Same config ⇒ identical outputs (deterministic slip=0 setup)."""
    from src.rlx.envs.tabular.gridworld import build_4room
    from src.rlx.algos.dp.value_iteration import run_vi
    from src.rlx.algos.dp.policy_iteration import run_pi

    # Deterministic environment (no slip) ⇒ deterministic algorithms given same code path
    mdp1 = build_4room(gamma=0.99, slip=0.0, seed=0)
    mdp2 = build_4room(gamma=0.99, slip=0.0, seed=0)

    vi1 = run_vi(mdp1, tol=1e-8, max_iters=1000, logger=None)
    vi2 = run_vi(mdp2, tol=1e-8, max_iters=1000, logger=None)

    assert np.allclose(vi1["V"], vi2["V"]) and np.all(vi1["pi"] == vi2["pi"])  

    pi1 = run_pi(mdp1, eval_tol=1e-8, max_eval_iters=1000, logger=None)
    pi2 = run_pi(mdp2, eval_tol=1e-8, max_eval_iters=1000, logger=None)

    assert np.allclose(pi1["V"], pi2["V"]) and np.all(pi1["pi"] == pi2["pi"])  


