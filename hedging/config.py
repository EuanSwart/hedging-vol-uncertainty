"""shared constants, file locations and the calibration loader."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# ---- locations
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SPY_CSV = DATA_DIR / "spy_vix.csv"
CALIB_FILE = DATA_DIR / "calibration.json"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"

# ---- market and contract
S0 = 100.0
T = 0.25                    # maturity in years
N_STEPS = 63                # daily rebalancing dates
ALPHA = 0.95                # cvar level
PAYOFFS = {
    "call": [(100.0, 1.0)],                                    # (strike, number of calls)
    "butterfly": [(90.0, 1.0), (100.0, -2.0), (110.0, 1.0)],
}

# ---- data split: the prior is fitted on the first half, real-data tests use the second
CALIBRATION_END = "2017-12-31"
TEST_START = "2018-01-01"

# ---- testing
TEST_SEED = 12345           # never used for training
N_GRID = 9                  # points on the true-sigma grid
N_PRIOR_TEST = 100_000
N_GRID_TEST = 50_000

SEEDS = (0, 1, 2)
PRIORS = ("calibrated", "uniform", "minimax")   # training priors: the spy-fitted one, a uniform one with a heavy
# upper tail, and 'minimax': the worst cvar over the test grid of constant sigma (a robust, not a bayesian, objective)
MINIMAX_TAU = 0.25         # temperature of the smooth maximum over sigma in the minimax objective (dollars)


@dataclass(frozen=True)
class Scale:
    """training budget: paths per step and number of adam steps."""
    name: str
    batch: int
    steps: int


SCALES = {
    "smoke": Scale("smoke", 256, 30),      # seconds, only checks that the code runs
    "main": Scale("main", 4096, 2500),     # the planned run, about 8 minutes per network
    "long": Scale("long", 8192, 4000),     # fallback if the pilot loss is still falling
}


@dataclass(frozen=True)
class Calibration:
    """volatility band and inverse-gamma prior on sigma^2, fitted by run_calibrate.py."""
    band: tuple
    a0: float
    b0: float
    sigma_ref: float        # prior-mean sigma, the reference volatility of the networks


@lru_cache(maxsize=None)
def load_calibration(path: str = str(CALIB_FILE)) -> Calibration:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run `python run_calibrate.py` first")
    c = json.loads(p.read_text())
    return Calibration(tuple(c["band"]), c["a0"], c["b0"], c["sigma_ref"])
