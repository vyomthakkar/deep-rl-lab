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


def main():
    mdp = build_4room(gamma=0.99, slip=0.0)
    shape = tuple(mdp.extras.get("shape", ()))
    print(
        f"Built 4-room MDP with shape={shape}, states={mdp.P.shape[0]}, actions={mdp.P.shape[1]}"
    )
    tau = 0.1
    tol = 1e-8
    max_iters = 1000
    result = run_soft_vi(mdp, tau=tau, tol=tol, max_iters=max_iters, logger=None)
    print(result)
    print(result["logs"])
    logs = result["logs"]
    plt.plot([log["delta"] for log in logs])
    plt.show()


if __name__ == "__main__":
    main()


