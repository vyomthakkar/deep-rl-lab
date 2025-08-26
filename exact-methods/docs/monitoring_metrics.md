# Monitoring Metrics for RL Algorithms

This document describes the key metrics tracked during training of exact RL methods (Value Iteration, Policy Iteration, Soft Value Iteration).

## Core Convergence Metrics

### 1. ΔV∞ (Delta V Infinity-norm)
**Formula**: `||V_{k+1} - V_k||_∞ = max_s |V_{k+1}(s) - V_k(s)|`

**Purpose**: Primary convergence criterion for Value Iteration

**Interpretation**:
- Large values (>0.1): Algorithm still learning significantly
- Medium values (0.001-0.1): Approaching convergence
- Small values (<1e-6): Near convergence
- When < tolerance: Stop VI iteration

**Usage**: Main stopping criterion for VI algorithms

---

### 2. Bellman Residual
**Formula**: `||T*V - V||_∞` where T* is the Bellman optimality operator

**Purpose**: Measures how far current value function is from satisfying Bellman optimality equation

**Note**: For standard VI, this equals ΔV∞ because VI performs `V_{k+1} = T*V_k`

**Interpretation**:
- 0: Perfect Bellman optimality (converged)
- >0: Still violating Bellman equation
- Useful for comparing different algorithms

---

### 3. Policy L1-change
**Formula**: `∑_s ||π_{k+1}(·|s) - π_k(·|s)||_1`

**For deterministic policies**: Count of states where action changed

**Purpose**: Track policy stability independent of value changes

**Interpretation**:
- High values: Policy still changing significantly
- 0: Policy has stabilized (even if values still changing)
- Can achieve optimal policy before value convergence

**Usage**: Early stopping criterion, policy stability analysis

---

## Algorithm-Specific Metrics

### 4. Entropy
**Formula**: `H(π) = -∑_s ∑_a π(a|s) log π(a|s)`

**For deterministic VI/PI**: Always 0 (since π(a|s) ∈ {0,1})

**For soft VI**: Measures policy randomness/exploration

**Interpretation**:
- 0: Completely deterministic policy
- log(|A|): Uniform random policy (maximum entropy)
- Higher values: More exploratory
- Lower values: More greedy/deterministic

**Usage**: Exploration-exploitation trade-off analysis in soft methods

---

## Performance Metrics

### 5. Wall-clock Time
**What**: Actual elapsed time in seconds

**Components**:
- `iter_time_sec`: Time per iteration
- `cumulative_time_sec`: Total time since start

**Purpose**: Real-world efficiency comparison

**Why critical**:
- Iterations-to-convergence doesn't tell full story
- Different algorithms have different per-iteration costs
- Essential for VI vs PI vs Soft-VI comparisons
- Real-world deployment constraint

---

## Logging Best Practices

### Memory Efficiency
- **Store scalars only** in iteration logs (not full arrays)
- Log summary statistics instead of raw vectors
- Save full V, Q, π only at convergence

### Frequency
- **Every iteration**: Core metrics (ΔV∞, policy change, time)
- **Periodic**: Expensive metrics (entropy calculations)
- **Final**: Complete state (V, Q, π, full logs)

### Example Log Entry
```python
{
    "iteration": 42,
    "delta_v_inf": 0.00123,
    "bellman_residual": 0.00123,  # Same as delta_v_inf for VI
    "policy_l1_change": 0,        # Policy stable
    "entropy": 0.0,               # Deterministic policy  
    "iter_time_sec": 0.001,
    "cumulative_time_sec": 0.045,
    "v_mean": 15.7,               # Summary stats
    "v_std": 8.2,
    "converged": False
}
```

## Convergence Analysis

### Typical Convergence Pattern
1. **Early iterations**: High ΔV∞, frequent policy changes
2. **Mid iterations**: Decreasing ΔV∞, policy stabilizing
3. **Late iterations**: Low ΔV∞, stable policy
4. **Convergence**: ΔV∞ < tolerance, policy_l1_change = 0

### Diagnostic Uses
- **Slow convergence**: High ΔV∞ after many iterations
- **Oscillation**: ΔV∞ not monotonically decreasing
- **Policy instability**: policy_l1_change > 0 when ΔV∞ is small
- **Algorithm comparison**: Wall-clock time vs iterations trade-offs

## References
- Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 4
- Puterman, "Markov Decision Processes", Chapter 6
