PYTHON := python
EXPORT := PYTHONPATH=src
ARGS ?=

.PHONY: fig-vi fig-pi fig-soft fig-all fig-vi-slip03 fig-pi-slip03

fig-vi:
	@mkdir -p runs/figs
	@$(EXPORT) $(PYTHON) -m rlx.scripts.make_4room_figure --algo vi --tol 1e-8 --outfile runs/figs/4room_vi.png $(ARGS)

fig-pi:
	@mkdir -p runs/figs
	@$(EXPORT) $(PYTHON) -m rlx.scripts.make_4room_figure --algo pi --eval_tol 1e-8 --max_eval_iters 1000 --outfile runs/figs/4room_pi.png $(ARGS)

fig-soft:
	@mkdir -p runs/figs
	@$(EXPORT) $(PYTHON) -m rlx.scripts.make_4room_figure --algo soft_vi --tau 0.1 --tol 1e-8 --outfile runs/figs/4room_soft_vi.png $(ARGS)

fig-all: fig-vi fig-pi fig-soft
	@echo "Saved figures in runs/figs"

fig-vi-slip03:
	@$(MAKE) fig-vi ARGS="$(ARGS) --slip 0.3 --outfile runs/figs/4room_vi_slip03.png"

fig-pi-slip03:
	@$(MAKE) fig-pi ARGS="$(ARGS) --slip 0.3 --outfile runs/figs/4room_pi_slip03.png"


