# rlx/scripts/train.py (sketch)
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1) Build env/MDP
    mdp = instantiate(cfg.env)  # calls toy2state.build(...)

    # 2) Dispatch algo
    if cfg.algo.name == "value_iteration":
        from rlx.algos.dp.value_iteration import run_vi
        out = run_vi(mdp, tol=cfg.algo.tol, max_iters=cfg.algo.max_iters, logger=None)
    elif cfg.algo.name == "policy_iteration":
        from rlx.algos.dp.policy_iteration import run_pi
        out = run_pi(mdp, eval_tol=cfg.algo.eval_tol, max_eval_iters=cfg.algo.max_eval_iters, logger=None)
    elif cfg.algo.name == "soft_value_iteration":
        from rlx.algos.dp.soft_value_iteration import run_soft_vi
        out = run_soft_vi(mdp, tau=cfg.algo.tau, tol=cfg.algo.tol, max_iters=cfg.algo.max_iters, logger=None)
    else:
        raise ValueError(f"Unknown algo: {cfg.algo.name}")

    # 3) Save artifacts / plots here…

if __name__ == "__main__":
    main()
