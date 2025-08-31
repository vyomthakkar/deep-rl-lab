import numpy as np
from src.rlx.envs.tabular.toy2state import build, closed_form_values
from src.rlx.algos.dp.policy_iteration import run_pi


def test_pi_toy2state_closed_form():
    """PI correctness (Toy2State): matches closed form within 1e-8 and greedy policy."""
    gamma, p_loop, r_reward = 0.99, 0.5, 1.0
    mdp = build(gamma=gamma, p_loop=p_loop, r_reward=r_reward)
    v_cf = closed_form_values(gamma, p_loop, r_reward)

    result = run_pi(mdp, eval_tol=1e-10, max_eval_iters=1000, logger=None)

    # Values match closed form
    assert np.max(np.abs(result["V"] - v_cf)) < 1e-8

    # Optimal policy: at s0 take a0; s1 is terminal (any action equivalent, expect 0 by argmax tie)
    assert result["pi"][0] == 0


def test_pi_vi_agreement_4room():
    """PI↔VI agreement on 4Rooms (slip=0): values agree within 1e-6; policies match."""
    from src.rlx.envs.tabular.gridworld import build_4room
    from src.rlx.algos.dp.value_iteration import run_vi

    mdp = build_4room(gamma=0.99, slip=0.0)

    vi = run_vi(mdp, tol=1e-8, max_iters=2000, logger=None)
    pi = run_pi(mdp, eval_tol=1e-8, max_eval_iters=1000, logger=None)

    # Values close
    assert np.max(np.abs(vi["V"] - pi["V"])) < 1e-6

    # Greedy policies match (tie-break by first max is consistent in both)
    assert np.all(vi["pi"] == pi["pi"]) 


