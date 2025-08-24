# src/rlx/envs/tabular/toy2state.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np

Array = np.ndarray

@dataclass
class TabularMDP:
    """
    Minimal tabular MDP container.

    P: shape (S, A, S) transition probabilities
    R: shape (S, A) expected immediate rewards
    gamma: discount in [0,1)
    terminal_mask: shape (S,), True for terminal/absorbing states
    state_names / action_names: optional labels for nicer plots/logs
    """
    P: Array
    R: Array
    gamma: float
    terminal_mask: Array
    state_names: Optional[List[str]] = None
    action_names: Optional[List[str]] = None
    extras: Optional[Dict] = None  # free-form metadata

def build(gamma: float = 0.99, p_loop: float = 0.5, r_reward: float = 1.0, seed: int = 0) -> TabularMDP:
    """
    Two states (s0, s1), two actions (a0, a1).

    - s0, a0: gives reward r_reward and with prob p_loop stays in s0,
              otherwise goes to s1 (absorbing, zero reward).
    - s0, a1: zero reward, stays in s0 (a "do nothing" action).
    - s1: absorbing with zero reward for both actions.

    This yields a nice closed form: always taking a0 at s0 achieves
        V*(s0) = r_reward / (1 - gamma * p_loop),  V*(s1) = 0
    which you can use to test your VI/PI implementations.
    """
    rng = np.random.default_rng(seed)
    S, A = 2, 2
    P = np.zeros((S, A, S), dtype=np.float64)
    R = np.zeros((S, A), dtype=np.float64)

    # State 0 transitions
    P[0, 0, 0] = p_loop      # a0 loops to s0 with prob p
    P[0, 0, 1] = 1.0 - p_loop
    R[0, 0] = r_reward

    P[0, 1, 0] = 1.0         # a1 stays in s0
    P[0, 1, 1] = 0.0
    R[0, 1] = 0.0

    # State 1 absorbing under both actions
    P[1, :, 1] = 1.0
    R[1, :] = 0.0

    terminal_mask = np.array([False, True], dtype=bool)

    # sanity: rows sum to 1
    assert np.allclose(P.sum(axis=-1), 1.0), "Transition rows must sum to 1."

    mdp = TabularMDP(
        P=P,
        R=R,
        gamma=float(gamma),
        terminal_mask=terminal_mask,
        state_names=["s0", "s1"],
        action_names=["a0", "a1"],
        extras={"desc": "Toy2State(p_loop={}, r_reward={})".format(p_loop, r_reward)},
    )
    return mdp

def closed_form_values(gamma: float, p_loop: float, r_reward: float) -> np.ndarray:
    """
    Optimal values for the Toy2State MDP when always taking a0 at s0.
    V*(s0) = r / (1 - gamma * p),  V*(s1) = 0
    (Since a1 yields zero forever, it's suboptimal for r>0.)
    """
    v0 = r_reward / (1.0 - gamma * p_loop)
    return np.array([v0, 0.0], dtype=np.float64)
