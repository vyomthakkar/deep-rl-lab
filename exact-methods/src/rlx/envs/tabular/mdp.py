from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np

Array = np.ndarray


@dataclass
class TabularMDP:
    """
    Minimal tabular MDP container used across tabular environments.

    Attributes:
        P: Transition probabilities, shape (S, A, S) with rows summing to 1.
        R: Expected immediate rewards, shape (S, A).
        gamma: Discount factor in [0, 1).
        terminal_mask: Boolean mask over states, shape (S,), True for terminals.
        state_names: Optional list of human-readable state labels.
        action_names: Optional list of action labels.
        extras: Optional dict for environment-specific metadata (e.g., grid shape, walls).
    """
    P: Array
    R: Array
    gamma: float
    terminal_mask: Array
    state_names: Optional[List[str]] = None
    action_names: Optional[List[str]] = None
    extras: Optional[Dict] = None
