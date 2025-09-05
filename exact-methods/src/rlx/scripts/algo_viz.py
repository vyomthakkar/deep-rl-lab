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
    


if __name__ == "__main__":
    delta_soft_vi = soft_vi()
    delta_vi = vi()
    
    # print(f"{delta_soft_vi[:100]=}")
    # print(f"{delta_vi[:100]=}")

    # plt.plot(delta_soft_vi[:15], label="Soft VI")
    # plt.plot(delta_vi[:15], label="VI")
    # plt.legend()
    # plt.show()


