### Goals
- Observe and compare VI/PI training dynamics.
- Log standard metrics consistently.
- Produce reproducible, aggregated plots across seeds/configs.

### What to log (per iteration)
Unify keys across algorithms to simplify plotting:
- Common
  - iter: VI iteration index (k) or PI outer iteration (k)
  - bellman_residual: ||T V − V||∞ for VI; policy-eval residual for PI
  - policy_l1_change: number of states whose greedy action changed since previous iter
  - wall_clock_time: seconds since start
- VI-only
  - algo: "vi"
- PI-only
  - algo: "pi"
  - inner_eval_iters: number of Jacobi sweeps in that outer iteration

Optional (nice-to-have):
- entropy: 0.0 for deterministic PI/VI; >0 for soft methods
- max_q: max_s,a Q(s,a) for monotonicity diagnostics
- seed, gamma, slip, env_name, timestamp: include as run metadata and duplicate in each row if convenient

### Where to write logs
- Write both CSV and JSONL for convenience:
  - CSV rows: columns above
  - JSONL: each row as a dict for flexible parsing
- Directory layout for reproducibility:
  - runs/logs/{env}/{algo}/gamma={γ}_slip={p}_seed={s}/metrics.csv
  - runs/meta/{...}/config.yaml (Hydra/OmegaConf dump or custom metadata.json)
- Keep figures under:
  - runs/figs/{env}/{algo}/...

### How to produce the plots (scripts)
Create a small plotting script that can consume one or many logs and emit standard figures; typical panels:

- Residual vs iteration (semi-log y)
  - y: bellman_residual
  - x: iter
  - Lines for VI and PI; optionally shaded mean ± 95% CI across seeds

- Policy L1-change vs iteration
  - y: policy_l1_change
  - x: iter
  - Useful to visualize when policy stabilizes (hits 0)

- Iterations to tolerance and wall-clock
  - Bar chart by algo (VI vs PI) for:
    - iterations_to_tol
    - wall_clock_time_to_tol
  - Compute iterations_to_tol from first iter where residual < tol

- Softness & entropy (if/when added)
  - y1: mean policy entropy vs τ
  - y2: return vs τ (optional)

- Sensitivity panels (small multiples)
  - Loop over γ ∈ {0.90, 0.99}, slip ∈ {0, 0.1, 0.3}
  - Residual and policy-change curves per condition

### Aggregation across seeds
- A loader that accepts a list of CSVs (glob pattern per condition):
  - Align by iter (truncate to min length if runs diverge)
  - Compute mean, std/sem, and 95% CI per iter
  - For PI, iter already corresponds to outer iterations; also report mean inner_eval_iters per outer iter

### CLI ergonomics
- Add flags to the plotting script:
  - --logs 'runs/logs/gridworld_4room/value_iteration/gamma=0.99_slip=0.1_seed=*/metrics.csv'
  - --algo {vi,pi,soft_vi,auto}
  - --plot {residual,policy_change,iterations,wall_clock,all}
  - --outdir runs/figs/...
  - --xmax/--ymin/--ymax, --semilog-y for residual
  - --groupby 'algo' to overlay VI and PI

### Standard RL practice notes
- Use semi-log for residuals; linear for policy-change.
- Plot means with shaded CI across seeds; include per-run faint lines optionally.
- Keep consistent color/style per algo across all plots.
- Record and display tol used; annotate the plot with iterations_to_tol and final residual.
- Ensure deterministic ordering and tie-breaking for fair VI/PI comparisons.

### Minimal workflow
- Train (or just run VI/PI) with logging enabled → writes CSV/JSONL per run.
- Call plotting script with a glob for the condition(s) you want to visualize.
- Save multi-panel figure(s) to `runs/figs/...`.

- Plan: standardize log schema, write CSV/JSONL per run under `runs/logs/...`, and add a dedicated plotting script that creates semi-log residual and policy-change curves, plus iteration/time bar charts, with optional seed aggregation.