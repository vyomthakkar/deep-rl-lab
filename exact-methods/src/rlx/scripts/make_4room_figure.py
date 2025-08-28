# src/rlx/scripts/make_4room_figure.py
from __future__ import annotations
import argparse
import re
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


### How to run: 
# cd /Users/vyomthakkar/Downloads/deep-rl-practice/exact-methods
# PYTHONPATH=src python -m rlx.scripts.make_4room_figure --outfile runs/figs/4room_vi.png

#

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

def plot_values_and_policy(
    mdp,
    V: np.ndarray,
    Q: np.ndarray,
    pi: np.ndarray,
    outfile: str,
    title: str = "",
    annotate: bool = False,
    decimals: int = 2,
    annotate_fontsize: int = 8,
    annotate_contrast: bool = True,
    annotate_box_alpha: float = 0.0,
    arrow_alpha: float = 0.9,
    hide_arrows_on_annotate: bool = False,
    fig_scale: float = 0.6,
    dpi: int = 140,
):
    shape = tuple(mdp.extras.get("shape", ()))
    walls = set(map(tuple, mdp.extras.get("walls", [])))
    h, w = shape

    gridV = value_grid_from_V(V, mdp.state_names, shape, walls)

    fig, ax = plt.subplots(figsize=(w * fig_scale, h * fig_scale), dpi=dpi)

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
    if not (hide_arrows_on_annotate and annotate):
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
            angles="xy", scale_units="xy", scale=2.5, width=0.012, headwidth=3, headlength=4,
            alpha=arrow_alpha, zorder=2
        )
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)  # invert to align with matrix indexing
    ax.set_xticks(range(w)); ax.set_yticks(range(h))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_aspect("equal")
    ax.grid(which="both", color=(0,0,0,0.15), linewidth=0.5)

    # Annotate V(s) values if requested
    if annotate:
        for r in range(h):
            for c in range(w):
                val = gridV[r, c]
                if np.isnan(val):
                    continue
                text_color = "black"
                if annotate_contrast:
                    rgba = im.cmap(im.norm(val))
                    rr, gg, bb, _ = rgba
                    # Perceptual luminance for contrast selection
                    luminance = 0.299 * rr + 0.587 * gg + 0.114 * bb
                    text_color = "black" if luminance > 0.5 else "white"
                path_effects = [pe.withStroke(linewidth=1.2, foreground=(1,1,1) if text_color=="black" else (0,0,0))]
                bbox = None
                if annotate_box_alpha and annotate_box_alpha > 0.0:
                    bbox = dict(boxstyle="round,pad=0.2", fc=(1,1,1, annotate_box_alpha), ec="none")
                ax.text(
                    c, r, f"{val:.{decimals}f}",
                    ha="center", va="center",
                    fontsize=annotate_fontsize, color=text_color,
                    fontweight="semibold", zorder=3,
                    path_effects=path_effects, bbox=bbox,
                )

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
    parser.add_argument("--annotate", action="store_true", help="Overlay numeric V(s) on cells")
    parser.add_argument("--print_values", action="store_true", help="Print the V-grid to stdout")
    parser.add_argument("--save_values", type=str, default="", help="Path to save V-grid (.csv, .npy, or .npz)")
    parser.add_argument("--decimals", type=int, default=2, help="Decimal places for display/save")
    parser.add_argument("--annotate_fontsize", type=int, default=8, help="Font size for value labels")
    parser.add_argument("--annotate_contrast", action="store_true", help="Auto-pick text color for contrast")
    parser.add_argument("--annotate_box_alpha", type=float, default=0.0, help="Alpha for value label background box")
    parser.add_argument("--arrow_alpha", type=float, default=0.9, help="Alpha for policy arrows")
    parser.add_argument("--hide_arrows_on_annotate", action="store_true", help="Hide arrows when annotating values")
    parser.add_argument("--fig_scale", type=float, default=0.6, help="Figure scale per grid cell")
    parser.add_argument("--dpi", type=int, default=140, help="Figure DPI")
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
            # Prefer returned Q/pi when available; otherwise derive from V
            if "Q" in out:
                Q = np.asarray(out["Q"], dtype=np.float64)
            else:
                Q = q_from_v(mdp.P, mdp.R, mdp.gamma, V)
            if "pi" in out:
                pi = np.asarray(out["pi"], dtype=np.int64)
            else:
                pi = greedy_from_q(Q)
            # Extract iteration count/time from logs if available
            iters_run = None
            elapsed_sec = None
            logs = out.get("logs", None)
            if isinstance(logs, list) and len(logs) > 0:
                last = logs[-1]
                # logs use 0-based iteration indexing
                try:
                    iters_run = int(last.get("i", len(logs) - 1)) + 1
                except Exception:
                    iters_run = len(logs)
                try:
                    elapsed_sec = float(last.get("wall_clock_time", None))
                except Exception:
                    elapsed_sec = None
        else:
            V = np.asarray(out, dtype=np.float64)
            Q = q_from_v(mdp.P, mdp.R, mdp.gamma, V)
            pi = greedy_from_q(Q)
    else:
        # Fallback mini-VI so the script still runs if import path differs
        V = np.zeros(mdp.P.shape[0], dtype=np.float64)
        last = V.copy()
        iters_run = 0
        for _ in range(args.max_iters):
            Q = mdp.R + mdp.gamma * (mdp.P @ V)
            V = np.max(Q, axis=1)
            if np.max(np.abs(V - last)) < args.tol:
                break
            last = V
            iters_run += 1
        # Derive policy from final Q
        pi = greedy_from_q(Q)

    # If Q/pi not set for any reason (safety), compute from V
    if 'Q' not in locals():
        Q = q_from_v(mdp.P, mdp.R, mdp.gamma, V)
    if 'pi' not in locals():
        pi = greedy_from_q(Q)

    # Prepare value grid for optional printing/saving
    shape = tuple(mdp.extras.get("shape", ()))
    walls = set(map(tuple, mdp.extras.get("walls", [])))
    gridV = value_grid_from_V(V, mdp.state_names, shape, walls)

    if args.print_values:
        np.set_printoptions(precision=args.decimals, suppress=False, linewidth=200)
        print("V-grid (NaN = wall):")
        print(gridV)

    if args.save_values:
        path = args.save_values
        lower = path.lower()
        if lower.endswith('.csv'):
            fmt = f"%.{args.decimals}f"
            np.savetxt(path, gridV, delimiter=",", fmt=fmt)
            print(f"[saved values] {path}")
        elif lower.endswith('.npz'):
            np.savez_compressed(path, values=gridV)
            print(f"[saved values] {path}")
        else:
            # default to .npy if not csv/npz
            if not lower.endswith('.npy'):
                path = path + '.npy'
            np.save(path, gridV)
            print(f"[saved values] {path}")

    # Compose title with convergence info if available
    title = f"4-Room Gridworld — VI (γ={args.gamma}, slip={args.slip})"
    try:
        if 'iters_run' in locals() and iters_run is not None:
            title += f" — iters={int(iters_run)}"
        if 'elapsed_sec' in locals() and elapsed_sec is not None:
            title += f" — t={elapsed_sec:.3f}s"
    except Exception:
        pass
    plot_values_and_policy(
        mdp, V, Q, pi, args.outfile, title,
        annotate=args.annotate, decimals=args.decimals,
        annotate_fontsize=args.annotate_fontsize,
        annotate_contrast=args.annotate_contrast,
        annotate_box_alpha=args.annotate_box_alpha,
        arrow_alpha=args.arrow_alpha,
        hide_arrows_on_annotate=args.hide_arrows_on_annotate,
        fig_scale=args.fig_scale,
        dpi=args.dpi,
    )

if __name__ == "__main__":
    main()
