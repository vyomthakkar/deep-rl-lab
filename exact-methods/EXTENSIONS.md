## Extensions roadmap

Short, actionable ideas to extend the exact methods module later. Keep everything tabular and NumPy-first unless noted.

## Policy evaluation variants

- Linear-solve evaluation (exact V^π)
  - Solve (I − γ P^π) V = r^π exactly.
  - Snippet (shape-safe):
```python
import numpy as np
S = mdp.P.shape[0]
idx = np.arange(S)
P_pi = mdp.P[idx, pi, :]    # (S, S)
r_pi = mdp.R[idx, pi]       # (S,)
V = np.linalg.solve(np.eye(S) - mdp.gamma * P_pi, r_pi)
```
  - Notes: γ < 1 ⇒ matrix is non-singular. Fall back to `lstsq` if needed.

- Gauss–Seidel iterative evaluation (in-place)
  - Faster convergence than Jacobi sweeps:
```python
for it in range(max_eval_iters):
    V_old = V.copy()
    for s in range(S):
        a = pi[s]
        V[s] = R[s, a] + gamma * (P[s, a, :] @ V)  # in-place uses latest V
    if np.max(np.abs(V - V_old)) < eval_tol:
        break
```

- Modified Policy Iteration (MPI)
  - Do k evaluation sweeps (Jacobi or GS) per improvement step. k∈{1,3,5} often works well.
  - Config keys: `pi.mpi_k`, `pi.eval_mode ∈ {jacobi, gs, linear}`.

- Successive Over-Relaxation (SOR)
  - Optional speedup for evaluation: update `V[s] ← (1−ω) V[s] + ω · T_π V[s]`, ω∈(0,2).

## Policy improvement and stopping

- Tie-breaking policy
  - Keep `argmax` first-max; optionally support stable argmax via small action priors.
  - Deterministic reproducibility: document seed and tie-break.

- Stopping criteria
  - Current: stop when π stops changing.
  - Optional: stop when `max_a Q(s,a) − Q(s,π(s)) < ε_improve` for all s.

## Value iteration variants

- Asynchronous VI
  - Update states in a sweep order (e.g., row-major) or random permutation each iter.

- Prioritized sweeping
  - Maintain a priority queue on states by Bellman error; update largest first.

- Anderson acceleration (advanced)
  - Extrapolate iterates to speed fixed-point convergence of VI.

## Soft / Max-Ent extensions

- Soft Policy Iteration (τ > 0)
  - Soft evaluation: `V_τ(s) = τ·log ∑_a exp(Q_τ(s,a)/τ)`.
  - Soft improvement: `π_τ(a|s) ∝ exp(Q_τ(s,a)/τ)`.
  - Implement with `scipy.special.logsumexp` for stability (or manual trick).

- Temperature sweeps and schedules
  - Grid τ∈{0.05, 0.1, 0.5, 1.0}; verify τ→0 recovers hard control within 1e−4.

## Analysis and tests

- PI ↔ VI agreement
  - After convergence on 4Rooms (slip=0): `||V_VI − V_PI||∞ < 1e−6`; greedy policies match.

- Linear-solve vs iterative eval
  - For fixed π, compare `V^π` from linear solve vs (Jacobi/GS) within 1e−8.

- Monotone policy improvement
  - Track `J(π_k)` or `max_a Q_k(s,a)` monotonicity across improvements.

- Random MDP fuzzer
  - Sample small MDPs (S≤12, A≤5); empirically verify contraction/evaluation/improvement properties.

## Performance and engineering

- Vectorization first
  - Precompute `EV = P @ V` and reuse; avoid Python loops except GS inner loop.

- Optional JIT
  - If desired, add `numba` for GS/MPI inner loops (feature-gated, not default).

- Logging/metrics
  - Track eval residual (||T_π V − V||∞), optimality residual (||T*V − V||∞), policy L1-change, entropy (for soft), wall-clock.

## Wiring (configs)

- Hydra/CLI knobs to add when implementing
  - `algo=pi`: `eval_mode={jacobi,gs,linear}`, `mpi_k=int`, `sor_omega=float`.
  - `algo=soft_vi`: `tau=float`, `tau_schedule={const,decay}`, `max_iters`, `tol`.


## Setting the right temperature and Entropy for Soft Value Iteration:

- It is very important (if you set temperature) to visualize and understand the entropy (average entropy for all states or ideally average entropy for non-terminal states) and how it changes over iterations.
- Also, if you have a deterministic number of states and actions in the environment, you can actually compute the entropy of a uniform/random policy, which is just: S*log(|A|)
- Comparing this theoretical bound of the max entropy of the environment given the set of valid states and actions, with the current entropy give us an indication of how "random/stochastic" the optimal policy is, when compared to the uniform random policy!
- Plotting Temperature vs Policy Entropy allows us to visualize this and choose an appropriate setting of temperature.
