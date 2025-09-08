from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Ensure 'src' is on sys.path when running this file directly
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
_SRC_DIR = os.path.join(_ROOT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from rlx.envs.tabular.gridworld import build_4room
from rlx.algos.dp.soft_value_iteration import run_soft_vi, soft_bellman_backup
from rlx.algos.dp.value_iteration import run_vi
from rlx.algos.dp.policy_iteration import run_pi


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


### Part C.1 ###
###
### Convergence curves
###
### Plot ||V_{k+1}-V_k||∞ vs iteration (VI, Soft-VI @ τ=0.1).
### Plot policy change Σ_s ||π_{k+1}(.|s)−π_k(.|s)||₁ vs iteration.
###
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
    

### Part C.2 ###
### VI vs PI agreement
###
### After convergence, report ||V_VI−V_PI||∞ and whether greedy policies match (document tie-break rule).
###
def vi_pi_agreement():
    # --- Hyperparameters ---
    tol = 1e-8
    eval_tol = 1e-8  # Policy evaluation tolerance for PI
    max_iters = 1000
    max_eval_iters = 1000  # Max evaluation iterations for PI
    gamma = 0.99
    slip = 0.4  # Use deterministic environment for agreement testing
    seeds = [1, 2, 3, 4, 5]  # 5 seeds for robustness
    
    # --- Data Storage ---
    value_differences = []  # Store ||V_VI - V_PI||∞ for each seed
    policy_agreements = []  # Store policy match counts for each seed
    total_states = 0  # Will be set after first MDP creation
    
    # --- Run Experiments ---
    for seed in seeds:
        print(f"--- Running with seed: {seed} ---")
        np.random.seed(seed)  # Set the seed for reproducibility

        # Build deterministic environment for VI/PI agreement testing
        mdp = build_4room(gamma=gamma, slip=slip, seed=seed)
        if total_states == 0:
            total_states = len(mdp.state_names)
        print(f"  Environment: {total_states} states, slip = {slip}")
        
        # --- Value Iteration ---
        print("Running Value Iteration...")
        vi_result = run_vi(mdp, tol=tol, max_iters=max_iters, logger=None)
        V_vi = vi_result["V"]
        pi_vi = vi_result["pi"]
        Q_vi = vi_result["Q"]
        
        # --- Policy Iteration ---
        print("Running Policy Iteration...")
        pi_result = run_pi(mdp, eval_tol=eval_tol, max_eval_iters=max_eval_iters, logger=None)
        V_pi = pi_result["V"]
        pi_pi = pi_result["pi"]
        Q_pi = pi_result["Q"]
        
        # 1. Compute ||V_VI - V_PI||∞ 
        # 2. Check policy agreement (count matching actions)
        # 3. Handle tie-breaking cases (detect when Q-values are equal)
        # 4. Store results for aggregation across seeds
        value_diff = np.max(np.abs(V_vi - V_pi))
        policy_agreement_pctg = (np.sum(pi_vi == pi_pi) / total_states) * 100
        value_differences.append(value_diff)
        policy_agreements.append(policy_agreement_pctg)
        policy_tie_breaking(Q_vi, Q_pi, pi_vi, pi_pi, policy_agreement_pctg)
        
        
    # Results aggregation and reporting
    # 1. Compute mean/std of value differences across seeds
    # 2. Compute mean/std of policy agreement percentages
    # 3. Print summary statistics
    # 4. Document tie-breaking behavior (in the report)
    print(f"Value difference: {value_differences}")
    print(f"Policy agreement: {policy_agreements}")
    print(f"Value difference: {np.mean(value_differences):.6f} ± {np.std(value_differences):.6f}")
    print(f"Policy agreement: {np.mean(policy_agreements):.6f} ± {np.std(policy_agreements):.6f}")
    
    
def policy_tie_breaking(Q_vi, Q_pi, pi_vi, pi_pi, policy_agreement_pctg):
  print("### Tie Breaking analysis (start) ###\n")
  # After computing VI and PI results
  total_states = len(Q_vi)
  tie_states = []
  epsilon = 1e-12  # Very small threshold for detecting ties

  for state in range(total_states):
      q_vi = Q_vi[state]
      q_pi = Q_pi[state]

      # Sort Q-values to find potential ties
      q_sorted = np.sort(q_vi)[::-1]  # Descending order

      # Check if top 2 Q-values are very close
      if len(q_sorted) > 1 and (q_sorted[0] - q_sorted[1]) < epsilon:
          tie_states.append(state)

          # Check if policies differ for this tied state
          if pi_vi[state] != pi_pi[state]:
              print(f"Tie-break conflict at state {state}:")
              print(f"  Q-values (vi): {q_vi}")
              print(f"  Q-values (pi): {q_pi}")
              print(f"  VI choice: {pi_vi[state]}, PI choice: {pi_pi[state]}")

  print(f"\nTie analysis:")
  print(f"States with tied Q-values: {len(tie_states)}")
  print(f"States with policy disagreement: {total_states - int(policy_agreement_pctg/100 * total_states)}")
  print("### Tie Breaking analysis (end) ###\n")
  
  

### Part C.3 ###
###
### Sweep γ∈{0.90,0.99} × p_slip∈{0,0.1,0.3}.
### Record iterations to tolerance and wall-clock for VI vs PI (bar chart).
###
def gamma_slip_sensitivity():
    # Sweep params
    gamma_values = [0.9, 0.99]
    p_slip_values = [0.0, 0.1, 0.3]
    
    # Algo params
    tol = 1e-8
    max_iters = 1000
    max_eval_iters = 1000 #for pi
    eval_tol = 1e-8 #for pi
    
    # Data collection structures
    results = []
    
    print("=== Gamma & Slip Sensitivity Analysis ===")
    
    for gamma, p_slip in product(gamma_values, p_slip_values):
        print(f"\n--- Running γ={gamma}, slip={p_slip} ---")
        mdp = build_4room(gamma=gamma, slip=p_slip)
        print(f"Environment: {len(mdp.state_names)} states")
        
        # --- Value Iteration ---
        print("Running Value Iteration...")
        vi_result = run_vi(mdp, tol=tol, max_iters=max_iters, logger=None)
        vi_iters = len(vi_result['logs'])
        vi_converged = vi_result['converged']
        vi_time = vi_result['run_time']
        print(f"  VI: {vi_iters} iters, {vi_time:.4f}s, converged={vi_converged}")
        
        # --- Policy Iteration ---
        print("Running Policy Iteration...")
        pi_result = run_pi(mdp, eval_tol=eval_tol, max_eval_iters=max_eval_iters, logger=None)
        pi_iters = len(pi_result['logs'])  # outer iterations
        pi_converged = pi_result['converged']
        pi_time = pi_result['run_time']
        print(f"  PI: {pi_iters} iters, {pi_time:.4f}s, converged={pi_converged}")
        
        # Store results
        results.append({
            'gamma': gamma,
            'slip': p_slip,
            'vi_iters': vi_iters,
            'vi_time': vi_time,
            'vi_converged': vi_converged,
            'pi_iters': pi_iters,
            'pi_time': pi_time,
            'pi_converged': pi_converged,
        })
    
    print(f"\n=== Creating Bar Charts ===")
    
    # Create grouped bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data for plotting
    conditions = [f"γ={r['gamma']}, slip={r['slip']}" for r in results]
    vi_iters_data = [r['vi_iters'] for r in results]
    pi_iters_data = [r['pi_iters'] for r in results]
    vi_times_data = [r['vi_time'] for r in results]
    pi_times_data = [r['pi_time'] for r in results]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    # Iterations bar chart
    bars1 = ax1.bar(x - width/2, vi_iters_data, width, label='VI', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, pi_iters_data, width, label='PI', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Conditions (γ, slip)')
    ax1.set_ylabel('Iterations to Convergence')
    ax1.set_title('Iterations to Tolerance: VI vs PI')
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    # Wall-clock time bar chart
    bars3 = ax2.bar(x - width/2, vi_times_data, width, label='VI', alpha=0.8, color='skyblue')
    bars4 = ax2.bar(x + width/2, pi_times_data, width, label='PI', alpha=0.8, color='lightcoral')
    
    ax2.set_xlabel('Conditions (γ, slip)')
    ax2.set_ylabel('Wall-Clock Time (seconds)')
    ax2.set_title('Wall-Clock Time: VI vs PI')
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print(f"\n=== Summary Table ===")
    print(f"{'Condition':<15} {'VI Iters':<10} {'PI Iters':<10} {'VI Time':<10} {'PI Time':<10}")
    print("-" * 65)
    for r in results:
        condition = f"γ={r['gamma']}, slip={r['slip']}"
        print(f"{condition:<15} {r['vi_iters']:<10} {r['pi_iters']:<10} {r['vi_time']:<10.4f} {r['pi_time']:<10.4f}")
        
    return results


### DIAGNOSTIC FUNCTION ###
def debug_vi_pi_convergence():
    """Debug why VI and PI take the same number of iterations in deterministic case."""
    print("=== DEBUGGING VI vs PI Convergence ===")
    
    # Use deterministic case
    gamma = 0.99
    slip = 0.0
    tol = 1e-8
    eval_tol = 1e-8
    max_iters = 1000
    max_eval_iters = 1000
    
    mdp = build_4room(gamma=gamma, slip=slip)
    print(f"Environment: {len(mdp.state_names)} states, γ={gamma}, slip={slip}")
    
    print("\n--- VALUE ITERATION DETAILED ---")
    vi_result = run_vi(mdp, tol=tol, max_iters=max_iters, logger=None)
    vi_logs = vi_result["logs"]
    print(f"VI converged in {len(vi_logs)} iterations")
    print("First 5 VI iterations:")
    for i, log in enumerate(vi_logs[:5]):
        print(f"  Iter {i}: delta={log['delta']:.2e}, policy_changes={log['policy_l1_change']}")
    print("Last 5 VI iterations:")
    for i, log in enumerate(vi_logs[-5:], len(vi_logs)-5):
        print(f"  Iter {i}: delta={log['delta']:.2e}, policy_changes={log['policy_l1_change']}")
    
    print(f"\n--- POLICY ITERATION DETAILED ---")
    pi_result = run_pi(mdp, eval_tol=eval_tol, max_eval_iters=max_eval_iters, logger=None)
    pi_logs = pi_result["logs"]
    print(f"PI converged in {len(pi_logs)} outer iterations")
    print("All PI outer iterations:")
    for i, log in enumerate(pi_logs):
        print(f"  Outer {i}: inner_iters={log['inner_iter']}, delta={log['delta']:.2e}, policy_changes={log['policy_l1_change']}")
    
    print(f"\n--- VALUE COMPARISON ---")
    V_vi = vi_result["V"]
    V_pi = pi_result["V"]
    pi_vi = vi_result["pi"]
    pi_pi = pi_result["pi"]
    
    value_diff = np.max(np.abs(V_vi - V_pi))
    policy_agreement = np.sum(pi_vi == pi_pi) / len(pi_vi) * 100
    
    print(f"||V_VI - V_PI||∞ = {value_diff:.2e}")
    print(f"Policy agreement: {policy_agreement:.1f}%")
    
    # Check if policies are optimal from start
    print(f"\n--- INITIAL POLICY CHECK ---")
    # Check what the initial policy improvement would yield
    initial_V = np.zeros(len(mdp.state_names))
    EV_initial = mdp.P @ initial_V
    Q_initial = mdp.R + mdp.gamma * EV_initial
    optimal_pi_from_zero = Q_initial.argmax(axis=1)
    
    print(f"Optimal policy from V=0: first 10 actions = {optimal_pi_from_zero[:10]}")
    print(f"Final VI policy: first 10 actions = {pi_vi[:10]}")
    print(f"Final PI policy: first 10 actions = {pi_pi[:10]}")
    
    return {
        'vi_iters': len(vi_logs),
        'pi_iters': len(pi_logs),
        'value_diff': value_diff,
        'policy_agreement': policy_agreement
    }


if __name__ == "__main__":
    # convergence_curves()
    # vi_pi_agreement()
    # gamma_slip_sensitivity()
    debug_vi_pi_convergence()
