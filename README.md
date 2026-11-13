# hedging under volatility uncertainty: robust PDE hedge against a learned recurrent hedger

code for the experiment of `docs/experiment_plan.md` (section 6 of the report). it asks when learning an unknown
volatility from the price path beats guarding against the worst volatility in a band, on simulated data and on real
spy data. everything runs on a laptop cpu (about 2 cpu-hours for all training).

## hedgers

| label | what it is | trained |
|---|---|---|
| `bsb` | black-scholes-barenblatt (uncertain volatility) hedge, finite-difference pde | no |
| `gru` | recurrent deep hedger, trained across a volatility prior (meta-rl) | yes |
| `posterior_mean` | black-scholes delta at the posterior-mean volatility (certainty-equivalent, not bayes-optimal) | no |
| `mlp` | memoryless ablation with the same inputs as the gru | yes |
| `bs_ref`, `bs_vix`, `oracle` | references: black-scholes at the prior mean, at vix/100 (real data), at the true sigma (simulation) | no |

## definitions

- the seller starts with zero cash and trades self-financed: `X = g(S_n) - sum_k delta_k (S_{k+1} - S_k)`.
- required premium = cvar95 of `X`, in dollars. every hedger is scored from the same zero start.
- prior loss: cvar95 over paths whose volatility is drawn from the prior. worst-case loss: largest cvar95 over a grid of 9 constant volatilities in the band.
- the band and the prior are fitted on spy 2010-2017 only (`run_calibrate.py`); the 2018-2025 windows are never used for tuning.

## setup

```
python -m venv .venv && .venv\Scripts\activate        # python 3.11
pip install -r requirements.txt
```

## reproduce, in order

```
python -m pytest tests -q            # unit tests (seconds)
python run_calibrate.py              # writes data/calibration.json
python run_go_no_go.py               # free hedges only, tells whether the effect is measurable (about 2 minutes)
python run_train.py --scale main     # 12 networks, 2 single-thread workers (about 1 hour)
python run_evaluate.py --scale main  # simulation tests and decision rules
python run_real.py --scale main      # real-data test
python run_probe.py --scale main     # optional hidden-state probe
python run_figures.py --scale main   # figures and tables in results/main/
```

use `--scale smoke` anywhere for a seconds-long check of the pipeline, and `--scale long` (batch 8192, 4000 steps)
if the pilot shows the loss is still falling. finished training runs are skipped, so interrupted runs resume.

## reproducibility

- all randomness is seeded: networks use seeds 0 to 2, test sets use seed 12345 (never used for training), bootstraps use fixed seeds.
- results were produced on cpu with one torch thread per training run; a different thread count or torch version can change low-order bits.
- `data/spy_vix.csv` is committed; `data/download_data.py` re-downloads it (yahoo may adjust history).
- the bsb finite-difference table is recomputed in seconds and cached in `data/cache/` (not committed).

## layout

```
hedging/       library: config, payoffs, simulate, bsb, baselines, networks, losses, train, evaluate, calibration
run_*.py       one script per step of the plan
tests/         unit tests
data/          spy and vix closes, calibration.json
results/       json results, figures and tables per scale
runs/          checkpoints and training logs per scale
```

## known simplifications (see section 3 of the plan)

volatility is fixed within each simulated path; the posterior-mean hedge is not bayes-optimal; no transaction costs;
real-data windows overlap and are few (93 windows, 31 non-overlapping quarters), so that test is exploratory.
in discrete daily trading the bsb hedge's required premium can exceed its continuous-time guaranteed price,
most visibly for the butterfly at low volatility.
