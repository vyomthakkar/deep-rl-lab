# tests/test_toy2state_closed_form.py
import numpy as np
from src.rlx.envs.tabular.toy2state import build, closed_form_values


#run python3 -m pytest tests/test_toy2state_closed_form.py -v to run the test
def test_closed_form_match():
    gamma, p, r = 0.99, 0.5, 1.0
    mdp = build(gamma=gamma, p_loop=p, r_reward=r)
    v_cf = closed_form_values(gamma, p, r)

    # If your VI/PI are already implemented, import and run them here.
    # For now, just check shapes and formula sanity:
    assert np.allclose(v_cf[1], 0.0)
    assert v_cf[0] > r  # discounted geometric sum should exceed single-step reward
