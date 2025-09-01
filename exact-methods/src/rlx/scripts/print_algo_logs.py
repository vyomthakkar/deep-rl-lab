from __future__ import annotations
import argparse
import json
from typing import Any, Dict
import numpy as np
import os
import sys

# ------------------------------------------------------------
# Edit these defaults in-code if you prefer not to pass CLI args
# ------------------------------------------------------------
ALGO: str = "vi"           # {"vi", "pi"}
ENV_NAME: str = "4room"    # {"4room", "toy2"}
GAMMA: float = 0.99
SLIP: float = 0.10          # used only for 4room
SEED: int = 0

# VI params
TOL: float = 1e-8
MAX_ITERS: int = 2000

# PI params
EVAL_TOL: float = 1e-8
MAX_EVAL_ITERS: int = 1000

# Ensure 'src' is on sys.path when running this file directly
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def build_env(env: str, gamma: float, slip: float, seed: int):
    if env == "4room":
        from rlx.envs.tabular.gridworld import build_4room
        mdp = build_4room(gamma=gamma, slip=slip, seed=seed)
    elif env == "toy2":
        from rlx.envs.tabular.toy2state import build
        # toy2 has no slip; keep signature simple
        mdp = build(gamma=gamma, seed=seed)
    else:
        raise ValueError(f"Unknown env: {env}")
    return mdp


def run_algo_and_get_logs(algo: str, mdp, args: argparse.Namespace) -> Dict[str, Any]:
    if algo == "vi":
        from rlx.algos.dp.value_iteration import run_vi
        out = run_vi(mdp, tol=args.tol, max_iters=args.max_iters, logger=None)
    elif algo == "pi":
        from rlx.algos.dp.policy_iteration import run_pi
        out = run_pi(mdp, eval_tol=args.eval_tol, max_eval_iters=args.max_eval_iters, logger=None)
    else:
        raise ValueError(f"Unknown algo: {algo}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Run VI/PI and print logs (JSONL).")
    parser.add_argument("--algo", type=str, default=ALGO, choices=["vi", "pi"], help="Algorithm to run")
    parser.add_argument("--env", type=str, default=ENV_NAME, choices=["4room", "toy2"], help="Environment builder")
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--slip", type=float, default=SLIP, help="Slip probability (4room only)")
    parser.add_argument("--seed", type=int, default=SEED)
    # VI params
    parser.add_argument("--tol", type=float, default=TOL)
    parser.add_argument("--max_iters", type=int, default=MAX_ITERS)
    # PI params
    parser.add_argument("--eval_tol", type=float, default=EVAL_TOL)
    parser.add_argument("--max_eval_iters", type=int, default=MAX_EVAL_ITERS)
    args = parser.parse_args()

    mdp = build_env(args.env, gamma=args.gamma, slip=args.slip, seed=args.seed)
    out = run_algo_and_get_logs(args.algo, mdp, args)

    meta = {
        "algo": args.algo,
        "env": args.env,
        "gamma": float(args.gamma),
        "slip": float(args.slip),
        "seed": int(args.seed),
    }
    print(json.dumps({"meta": meta, "message": "begin_logs"}))
    logs = out.get("logs", [])
    for rec in logs:
        # Ensure JSON-serializable (convert numpy types)
        safe = {k: (v.item() if isinstance(v, (np.generic,)) else v) for k, v in rec.items()}
        print(json.dumps(safe))
    print(json.dumps({"meta": meta, "message": "end_logs", "num_records": len(logs)}))


if __name__ == "__main__":
    main()


