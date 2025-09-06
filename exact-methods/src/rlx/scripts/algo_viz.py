from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure 'src' is on sys.path when running this file directly
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
_SRC_DIR = os.path.join(_ROOT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from rlx.envs.tabular.gridworld import build_4room
from rlx.algos.dp.soft_value_iteration import run_soft_vi, soft_bellman_backup
from rlx.algos.dp.value_iteration import run_vi


#https://chatgpt.com/c/68b962cb-f4f4-8321-8588-9fd4858270ca

def soft_vi():
    print("=============== SOFT VI =============== ")
    mdp = build_4room(gamma=0.99, slip=0.0)
    shape = tuple(mdp.extras.get("shape", ()))
    print(
        f"Built 4-room MDP with shape={shape}, states={mdp.P.shape[0]}, actions={mdp.P.shape[1]}"
    )
    tau = 1e-6
    tol = 1e-8
    # tol = 1e-5
    max_iters = 1000
    result = run_soft_vi(mdp, tau=tau, tol=tol, max_iters=max_iters, logger=None)
    logs = result["logs"]
    lastV = result["V"]
    converged = result["converged"]
    print(f"{converged=}")
    print(f"{len(logs)=}")
    
    delta = [log["delta"] for log in logs]
    policy_l1_change = [log["policy_l1_change"] for log in logs]
    entropy = [log["entropy"] for log in logs]
    
    # plt.plot(entropy)
    # plt.show()
    
    print(f"{lastV=}")
    return delta
    
def vi():
    print("=============== VI =============== ")
    mdp = build_4room(gamma=0.99, slip=0.0)
    shape = tuple(mdp.extras.get("shape", ()))
    print(
        f"Built 4-room MDP with shape={shape}, states={mdp.P.shape[0]}, actions={mdp.P.shape[1]}"
    )
    tol = 1e-8
    # tol = 1e-5
    max_iters = 1000
    result = run_vi(mdp, tol=tol, max_iters=max_iters, logger=None)
    logs = result["logs"]
    lastV = result["V"]
    converged = result["converged"]
    print(f"{converged=}")
    print(f"{len(logs)=}")
    delta = [log["delta"] for log in logs]
    
    print(f"{lastV=}")
    return delta

### Assignment Plots (Part C) ###

def convergence_curves():
    # --- Hyperparameters ---
    tau = 0.1
    tol = 1e-5
    max_iters = 1000
    gamma = 0.99
    slip = 0.1  # For stochastic environment as per instructions
    seeds = [1, 2, 3, 4, 5]  # Using 5 seeds as requested

    # --- Data Storage ---
    # To store results for each seed
    vi_deltas_seeds = []
    vi_policy_changes_seeds = []
    soft_vi_deltas_seeds = []
    soft_vi_policy_changes_seeds = []

    # --- Run Experiments ---
    for seed in seeds:
        print(f"--- Running with seed: {seed} ---")
        np.random.seed(seed)  # Set the seed for reproducibility

        # Build the environment with slight variations per seed
        # Add small random perturbation to slip probability to create variance
        slip_varied = slip + np.random.normal(0, 0.02)  # Small variation around base slip
        slip_varied = np.clip(slip_varied, 0.0, 0.3)  # Keep within reasonable bounds
        mdp = build_4room(gamma=gamma, slip=slip_varied)
        print(f"  Using slip = {slip_varied:.4f}")

        # --- Value Iteration ---
        print("Running Value Iteration...")
        vi_result = run_vi(mdp, tol=tol, max_iters=max_iters, logger=None)
        vi_logs = vi_result["logs"]
        vi_deltas = [log["delta"] for log in vi_logs]
        vi_policy_changes = [log["policy_l1_change"] for log in vi_logs]
        vi_deltas_seeds.append(vi_deltas)
        vi_policy_changes_seeds.append(vi_policy_changes)

        # --- Soft Value Iteration ---
        print("Running Soft Value Iteration...")
        soft_vi_result = run_soft_vi(mdp, tau=tau, tol=tol, max_iters=max_iters, logger=None)
        soft_vi_logs = soft_vi_result["logs"]
        soft_vi_deltas = [log["delta"] for log in soft_vi_logs]
        soft_vi_policy_changes = [log["policy_l1_change"] for log in soft_vi_logs]
        soft_vi_deltas_seeds.append(soft_vi_deltas)
        soft_vi_policy_changes_seeds.append(soft_vi_policy_changes)

    print("\n--- All seeds completed ---")

    # --- Process and Plot Results ---
    def process_and_plot(data_seeds, title, ax):
        # Pad shorter sequences to the length of the longest one for aggregation
        max_len = max(len(x) for x in data_seeds)
        
        # Convert to float arrays first to handle NaN padding
        padded_data = []
        for x in data_seeds:
            x_float = np.array(x, dtype=float)
            if len(x_float) < max_len:
                padded = np.pad(x_float, (0, max_len - len(x_float)), 'constant', constant_values=np.nan)
            else:
                padded = x_float
            padded_data.append(padded)
        
        mean = np.nanmean(padded_data, axis=0)
        std = np.nanstd(padded_data, axis=0)
        
        # Debug: print std statistics
        print(f"{title}: mean std = {np.nanmean(std):.6f}, max std = {np.nanmax(std):.6f}")
        
        iterations = np.arange(max_len)
        ax.plot(iterations, mean, label=title, linewidth=2)
        # Make shading more visible with higher alpha and ensure it's plotted
        if np.nanmax(std) > 0:
            ax.fill_between(iterations, mean - std, mean + std, alpha=0.3, label=f'{title} ±1σ')
        else:
            print(f"Warning: {title} has zero standard deviation - no shading will be visible")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Value")
        ax.set_title(f"{title} vs. Iteration")
        ax.legend()
        ax.grid(True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Convergence Curves (5 Seeds, slip=0.1)')

    process_and_plot(vi_deltas_seeds, 'VI ||V_{k+1}-V_k||∞', axes[0, 0])
    process_and_plot(soft_vi_deltas_seeds, 'Soft-VI ||V_{k+1}-V_k||∞', axes[0, 1])
    process_and_plot(vi_policy_changes_seeds, 'VI Policy Change ||π_{k+1}-π_k||₁', axes[1, 0])
    process_and_plot(soft_vi_policy_changes_seeds, 'Soft-VI Policy Change ||π_{k+1}-π_k||₁', axes[1, 1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    
    
    
    
    
    
    
    


if __name__ == "__main__":
    convergence_curves()


