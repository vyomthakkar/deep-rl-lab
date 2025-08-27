# src/rlx/scripts/make_4room_figure.py
from __future__ import annotations
import argparse
import re
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt


### How to run: 
# cd /Users/vyomthakkar/Downloads/deep-rl-practice/exact-methods
# PYTHONPATH=src python -m rlx.scripts.make_4room_figure --outfile runs/figs/4room_vi.png

# Env + algo
from rlx.envs.tabular.gridworld import build_4room
try:
    # uses your implementation
    from rlx.algos.dp.value_iteration import run_vi  # expected: returns dict with 'V' or just V
    HAS_RUN_VI = True
except Exception:
    HAS_RUN_VI = False

# ----- helpers -----

ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
ARROW_VECS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}  # in (x,y) screen coords

def parse_state_rc(state_name: str) -> Tuple[int, int]:
    # expects "(r,c)"
    m = re.match(r"\((\-?\d+),\s*(\-?\d+)\)", state_name)
    if not m:
        raise ValueError(f"Cannot parse state name '{state_name}' into (r,c).")
    return int(m.group(1)), int(m.group(2))

def q_from_v(P: np.ndarray, R: np.ndarray, gamma: float, V: np.ndarray) -> np.ndarray:
    # Q[s,a] = R[s,a] + gamma * sum_s' P[s,a,s'] * V[s']
    return R + gamma * (P @ V)

def greedy_from_q(Q: np.ndarray) -> np.ndarray:
    return np.argmax(Q, axis=1)

def value_grid_from_V(V: np.ndarray, state_names: List[str], shape: Tuple[int, int], walls: List[Tuple[int,int]]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), np.nan, dtype=float)
    # fill traversable cells with V
    for idx, name in enumerate(state_names):
        r, c = parse_state_rc(name)
        grid[r, c] = V[idx]
    # walls remain NaN (will show as blank in imshow)
    return grid

# ----- plotting -----

def plot_values_and_policy(mdp, V: np.ndarray, Q: np.ndarray, pi: np.ndarray, outfile: str, title: str = ""):
    shape = tuple(mdp.extras.get("shape", ()))
    walls = set(map(tuple, mdp.extras.get("walls", [])))
    h, w = shape

    gridV = value_grid_from_V(V, mdp.state_names, shape, walls)

    fig, ax = plt.subplots(figsize=(w * 0.6, h * 0.6), dpi=140)

    # Heatmap of values; NaNs (walls) masked out
    im = ax.imshow(gridV, interpolation="nearest", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Value V(s)")

    # Draw walls as black squares on top (optional, since NaN shows blank)
    for r in range(h):
        for c in range(w):
            if (r, c) in walls:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black"))

    # Terminals to mark
    terminals_idx = np.where(mdp.terminal_mask)[0]
    terminals_rc = {parse_state_rc(mdp.state_names[i]) for i in terminals_idx}

    # Quiver arrows for greedy actions
    xs, ys, us, vs = [], [], [], []
    for i, name in enumerate(mdp.state_names):
        r, c = parse_state_rc(name)
        if (r, c) in walls:
            continue
        if (r, c) in terminals_rc:
            continue  # no arrow on terminal
        a = int(pi[i])
        u, v = ARROW_VECS[a]
        xs.append(c); ys.append(r)
        us.append(u); vs.append(v)

    # Put arrows at cell centers; invert y so (0,0) is top-left like the grid indices
    ax.quiver(
        np.array(xs), np.array(ys),
        np.array(us), np.array(vs),
        angles="xy", scale_units="xy", scale=2.5, width=0.012, headwidth=3, headlength=4
    )
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)  # invert to align with matrix indexing
    ax.set_xticks(range(w)); ax.set_yticks(range(h))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_aspect("equal")
    ax.grid(which="both", color=(0,0,0,0.15), linewidth=0.5)

    # mark terminals
    for (r, c) in terminals_rc:
        ax.scatter(c, r, marker="*", s=140, edgecolor="k")

    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    print(f"[saved] {outfile}")

# ----- main -----

def main():
    parser = argparse.ArgumentParser(description="Make a 4-Room value/policy figure.")
    parser.add_argument("--height", type=int, default=11)
    parser.add_argument("--width", type=int, default=11)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--slip", type=float, default=0.10)
    parser.add_argument("--step_penalty", type=float, default=0.01)
    parser.add_argument("--goal_r", type=int, default=9)
    parser.add_argument("--goal_c", type=int, default=9)
    parser.add_argument("--goal_reward", type=float, default=1.0)
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--outfile", type=str, default="runs/figs/4room_vi.png")
    args = parser.parse_args()

    doors = [(5,1),(1,5),(9,5),(5,9)]
    mdp = build_4room(
        height=args.height, width=args.width, doors=doors,
        goal_pos=[args.goal_r, args.goal_c], goal_reward=args.goal_reward,
        pit_positions=[], pit_reward=-1.0,
        step_penalty=args.step_penalty, slip=args.slip, gamma=args.gamma,
        absorbing_terminals=True, seed=0
    )

    # Run VI (use your implementation if available; else a tiny fallback)
    if HAS_RUN_VI:
        out = run_vi(mdp, tol=args.tol, max_iters=args.max_iters, logger=None)
        if isinstance(out, dict) and "V" in out:
            V = np.asarray(out["V"], dtype=np.float64)
        else:
            V = np.asarray(out, dtype=np.float64)
    else:
        # Fallback mini-VI so the script still runs if import path differs
        V = np.zeros(mdp.P.shape[0], dtype=np.float64)
        last = V.copy()
        for _ in range(args.max_iters):
            Q = mdp.R + mdp.gamma * (mdp.P @ V)
            V = np.max(Q, axis=1)
            if np.max(np.abs(V - last)) < args.tol:
                break
            last = V

    Q = q_from_v(mdp.P, mdp.R, mdp.gamma, V)
    pi = greedy_from_q(Q)

    title = f"4-Room Gridworld — VI (γ={args.gamma}, slip={args.slip})"
    plot_values_and_policy(mdp, V, Q, pi, args.outfile, title)

if __name__ == "__main__":
    main()
