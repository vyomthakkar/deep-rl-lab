#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere; cd to repo root (this script's directory)
cd "$(dirname "$0")"

export PYTHONPATH=src
mkdir -p runs/figs

echo "[1/3] Generating 4room VI figure..."
python -m rlx.scripts.make_4room_figure --algo vi --tol 1e-8 --outfile runs/figs/4room_vi.png "$@"

echo "[2/3] Generating 4room PI figure (deterministic slip=0)..."
python -m rlx.scripts.make_4room_figure --algo pi --slip 0.0 --eval_tol 1e-8 --max_eval_iters 1000 --outfile runs/figs/4room_pi.png "$@"

echo "[3/3] Generating 4room Soft-VI figure (tau=0.1)..."
python -m rlx.scripts.make_4room_figure --algo soft_vi --tau 0.1 --tol 1e-8 --outfile runs/figs/4room_soft_vi.png "$@"

echo "[done] Figures saved in runs/figs"


