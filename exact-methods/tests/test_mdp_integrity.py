import numpy as np
import pytest


def _check_mdp_integrity(mdp):
    # Row-stochastic: sum over s' equals 1 for all (s,a)
    assert np.allclose(mdp.P.sum(axis=-1), 1.0)

    # Absorbing terminals (if specified as absorbing)
    # For any terminal state i, verify self-loop prob 1 across actions
    term_idx = np.where(mdp.terminal_mask)[0]
    if term_idx.size > 0:
        P_term = mdp.P[term_idx, :, :]
        for k, i in enumerate(term_idx):
            # at least one action should be self-loop with prob 1 in our builders
            assert np.allclose(P_term[k, :, i], 1.0), f"Terminal {i} not absorbing"


def test_toy2state_integrity():
    from src.rlx.envs.tabular.toy2state import build

    mdp = build(gamma=0.99, p_loop=0.5, r_reward=1.0)
    _check_mdp_integrity(mdp)


@pytest.mark.grid
def test_4room_integrity():
    from src.rlx.envs.tabular.gridworld import build_4room

    mdp = build_4room(gamma=0.99, slip=0.0)
    _check_mdp_integrity(mdp)


