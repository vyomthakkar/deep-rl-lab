from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

Array = np.ndarray

@dataclass
class TabularMDP:
    P: Array           # (S, A, S)
    R: Array           # (S, A)
    gamma: float
    terminal_mask: Array  # (S,)
    state_names: Optional[List[str]] = None
    action_names: Optional[List[str]] = None
    extras: Optional[Dict] = None

ACTIONS = [( -1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
ACTION_NAMES = ["U", "R", "D", "L"]

def _rc_to_idx(r:int, c:int, rc_to_idx:Dict[Tuple[int,int], int]) -> int:
    return rc_to_idx[(r,c)]

def _neighbors(h:int, w:int, walls:set, r:int, c:int):
    nxt = []
    for dr, dc in ACTIONS:
        rr, cc = r+dr, c+dc
        if 0 <= rr < h and 0 <= cc < w and (rr,cc) not in walls:
            nxt.append((rr,cc))
        else:
            nxt.append((r,c))  # bump into wall/outside ⇒ stay
    return nxt  # in U,R,D,L order

def _build_P_R_from_layout(layout: Array, gamma: float, step_penalty: float,
                           goal_cells:set, goal_reward: float,
                           pit_cells:set, pit_reward: float,
                           slip: float, absorbing_terminals: bool,
                           wind_cols: Optional[Dict[int,int]] = None,
                           seed:int = 0) -> TabularMDP:
    rng = np.random.default_rng(seed)
    h, w = layout.shape
    walls = {(r,c) for r in range(h) for c in range(w) if layout[r,c] == 1}
    traversable = [(r,c) for r in range(h) for c in range(w) if layout[r,c] == 0]
    rc_to_idx = {rc:i for i, rc in enumerate(traversable)}
    idx_to_rc = {i:rc for rc,i in rc_to_idx.items()}
    S, A = len(traversable), 4

    P = np.zeros((S, A, S), dtype=np.float64)
    R = np.zeros((S, A), dtype=np.float64)
    terminal_mask = np.zeros(S, dtype=bool)

    for rc in goal_cells | pit_cells:
        if rc not in rc_to_idx:
            raise ValueError(f"Terminal {rc} is not a traversable cell.")

    # mark terminals
    for i in range(S):
        r, c = idx_to_rc[i]
        if (r,c) in goal_cells or (r,c) in pit_cells:
            terminal_mask[i] = True

    for i in range(S):
        r, c = idx_to_rc[i]
        is_term = terminal_mask[i]

        for a, (dr, dc) in enumerate(ACTIONS):
            if is_term and absorbing_terminals:
                P[i, a, i] = 1.0
                R[i, a] = 0.0
                continue

            # intended next
            nbrs = _neighbors(h, w, walls, r, c)
            intended = nbrs[a]

            # slip model: with prob slip, move to a perpendicular action (split evenly)
            # perpendiculars: if a in {U,D} -> {L,R}; if a in {L,R} -> {U,D}
            if a in (0,2):
                perps = [3,1]  # L, R
            else:
                perps = [0,2]  # U, D

            branches = []
            branches.append(((1.0 - slip), intended))
            if slip > 0:
                for pa in perps:
                    branches.append((slip/2.0, nbrs[pa]))

            # apply wind (column-based upward drift) AFTER movement, if specified
            probs_to_sprime: Dict[int, float] = {}
            for p, (rr, cc) in branches:
                if wind_cols:
                    up = int(wind_cols.get(cc, 0))
                    rr = max(0, rr - up)
                    if (rr, cc) in walls:
                        rr, cc = r, c  # blown into wall ⇒ stay
                j = rc_to_idx[(rr, cc)]
                probs_to_sprime[j] = probs_to_sprime.get(j, 0.0) + p

            # normalize (just in case of numeric drift)
            total = sum(probs_to_sprime.values())
            for j, p in probs_to_sprime.items():
                P[i, a, j] = p / total

            # expected immediate reward = step_penalty + expected terminal bonus/penalty on entry
            exp_term = 0.0
            for j, p in probs_to_sprime.items():
                rr, cc = idx_to_rc[j]
                if (rr,cc) in goal_cells:
                    exp_term += p * goal_reward
                elif (rr,cc) in pit_cells:
                    exp_term += p * pit_reward
            R[i, a] = step_penalty + exp_term

    state_names = [f"({r},{c})" for (r,c) in traversable]
    return TabularMDP(
        P=P, R=R, gamma=gamma, terminal_mask=terminal_mask,
        state_names=state_names, action_names=ACTION_NAMES,
        extras={"shape": (h,w), "walls": list(walls)}
    )

# ---------- Public builders ----------

def build_4room(height:int=11, width:int=11, doors:List[List[int]]=((5,1),(1,5),(9,5),(5,9)),
                goal_pos:List[int]=[9,9], goal_reward:float=1.0,
                pit_positions:List[List[int]] = None, pit_reward:float=-1.0,
                step_penalty:float=0.01, slip:float=0.1, gamma:float=0.99,
                absorbing_terminals:bool=True, seed:int=0) -> TabularMDP:
    pit_positions = pit_positions or []
    layout = np.ones((height, width), dtype=int)  # 1=wall, 0=empty
    layout[1:-1,1:-1] = 0

    # draw the 4-room cross walls
    mid_r, mid_c = height//2, width//2
    layout[mid_r,1:-1] = 1
    layout[1:-1,mid_c] = 1

    # carve doorways
    for r,c in doors:
        layout[r, c] = 0

    goal_cells = {tuple(goal_pos)}
    pit_cells  = {tuple(p) for p in pit_positions}

    return _build_P_R_from_layout(
        layout=layout, gamma=gamma, step_penalty=step_penalty,
        goal_cells=goal_cells, goal_reward=goal_reward,
        pit_cells=pit_cells, pit_reward=pit_reward,
        slip=slip, absorbing_terminals=absorbing_terminals,
        wind_cols=None, seed=seed
    )

def build_from_ascii(gamma:float, ascii_map:str, legend:Dict[str,str],
                     step_penalty:float=0.01, goal_reward:float=1.0, pit_reward:float=-1.0,
                     slip:float=0.0, wind_cols:Optional[Dict[int,int]]=None,
                     absorbing_terminals:bool=True, seed:int=0) -> TabularMDP:
    lines = [list(row) for row in ascii_map.strip("\n").splitlines()]
    h, w = len(lines), len(lines[0])
    layout = np.zeros((h,w), dtype=int)
    goal_cells, pit_cells = set(), set()

    for r in range(h):
        for c in range(w):
            ch = lines[r][c]
            if ch == legend.get("wall", "#"):
                layout[r,c] = 1
            elif ch == legend.get("goal", "G"):
                layout[r,c] = 0; goal_cells.add((r,c))
            elif ch == legend.get("pit", "X"):
                layout[r,c] = 0; pit_cells.add((r,c))
            else:
                layout[r,c] = 0

    return _build_P_R_from_layout(layout, gamma, step_penalty,
                                  goal_cells, goal_reward, pit_cells, pit_reward,
                                  slip, absorbing_terminals, wind_cols, seed)

def build_cliff(width:int=12, height:int=4, start_pos:List[int]=[3,0],
                goal_pos:List[int]=[3,11], cliff_cols:List[int]=None,
                step_penalty:float=0.01, cliff_reward:float=-1.0, goal_reward:float=1.0,
                slip:float=0.0, gamma:float=0.99, absorbing_terminals:bool=True,
                seed:int=0) -> TabularMDP:
    cliff_cols = cliff_cols or list(range(1, width-1))
    layout = np.zeros((height,width), dtype=int)
    # outer walls
    layout[0,:]=1; layout[-1,:]=1; layout[:,0]=1; layout[:,-1]=1
    layout[start_pos[0], start_pos[1]] = 0
    layout[goal_pos[0], goal_pos[1]] = 0

    # “cliff” is a line of pits between start and goal on the bottom row
    pit_cells = {(start_pos[0], c) for c in cliff_cols}
    goal_cells = {tuple(goal_pos)}

    return _build_P_R_from_layout(
        layout, gamma, step_penalty, goal_cells, goal_reward,
        pit_cells, pit_reward=cliff_reward, slip=slip,
        absorbing_terminals=absorbing_terminals, wind_cols=None, seed=seed
    )
