"""evaluation helpers: test sets, hedge registry, losses on prior and grid sets, bootstrap."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from hedging.baselines import bs_delta_path, posterior_mean_delta
from hedging.bsb import bsb_delta_path
from hedging.config import ALPHA, Calibration, N_GRID, N_STEPS, TEST_SEED
from hedging.losses import cvar, shortfall
from hedging.networks import load_network, network_delta
from hedging.simulate import gbm_from_noise, make_generator, simulate_prior_paths


# ------------------------------------------------------------------ test sets
@dataclass
class SimTestSets:
    """prior paths, and paths at fixed true volatilities that all share the same noise (common random numbers)."""
    prior_S: Tensor
    prior_sigma: Tensor
    grid_sigma: list
    grid_S: list


def build_test_sets(calib: Calibration, n_prior: int, n_grid_paths: int, seed: int = TEST_SEED,
                    prior: str = "calibrated") -> SimTestSets:
    gen = make_generator(seed)
    prior_S, prior_sigma = simulate_prior_paths(n_prior, gen, calib, prior=prior)
    z = torch.randn(n_grid_paths, N_STEPS, generator=gen)
    grid_sigma = np.linspace(calib.band[0], calib.band[1], N_GRID).round(4).tolist()
    return SimTestSets(prior_S, prior_sigma, grid_sigma, [gbm_from_noise(z, s) for s in grid_sigma])


# ------------------------------------------------------------------ hedge registry
def free_hedges(name: str, calib: Calibration) -> dict:
    """hedges that need no training, as functions (S, sigma) -> deltas; sigma is only used by the oracle."""
    return {
        "bs_ref": lambda S, sigma: bs_delta_path(name, S, calib.sigma_ref),
        "bsb": lambda S, sigma: bsb_delta_path(name, S, calib.band),
        "posterior_mean": lambda S, sigma: posterior_mean_delta(name, S, calib),
        "oracle": lambda S, sigma: bs_delta_path(name, S, sigma),
    }


def network_hedges(name: str, calib: Calibration, scale: str, kinds=("gru", "mlp"), seeds=(0, 1, 2),
                   prior: str = "calibrated") -> dict:
    """trained networks that exist on disk, labelled like 'gru_s0'."""
    from hedging.networks import checkpoint_path
    out = {}
    for kind in kinds:
        for seed in seeds:
            if checkpoint_path(scale, kind, name, seed, prior).exists():
                model = load_network(scale, kind, name, seed, calib, prior)
                out[f"{kind}_s{seed}"] = (lambda m: lambda S, sigma: network_delta(m, S))(model)
    return out


# ------------------------------------------------------------------ losses
def sim_shortfalls(name: str, hedge, sets: SimTestSets) -> dict:
    """shortfalls of one hedge on the prior set and on every grid set."""
    x_prior = shortfall(name, sets.prior_S, hedge(sets.prior_S, sets.prior_sigma))
    x_grid = [shortfall(name, S, hedge(S, torch.full((S.shape[0],), s))) for s, S in zip(sets.grid_sigma, sets.grid_S)]
    return dict(prior=x_prior, grid=x_grid)


def summarize(x: dict, grid_sigma: list) -> dict:
    """prior loss and worst-case loss (largest cvar95 over the true-sigma grid), plus std as a secondary check."""
    grid_cvar = [cvar(g) for g in x["grid"]]
    grid_std = [float(g.std()) for g in x["grid"]]
    worst = int(np.argmax(grid_cvar))
    return dict(prior_cvar=cvar(x["prior"]), prior_std=float(x["prior"].std()),
                worst_cvar=grid_cvar[worst], worst_sigma=grid_sigma[worst], worst_std=max(grid_std),
                grid_cvar=grid_cvar, grid_std=grid_std)


# ------------------------------------------------------------------ bootstrap over test paths
def bootstrap_losses(xs: dict, n_boot: int = 200, seed: int = 0) -> dict:
    """paired bootstrap of prior and worst-case cvar for several hedges at once.

    all hedges and all grid points share the same resampled path indices, because they share their noise.
    returns {label: {"prior": array(n_boot), "worst": array(n_boot)}}."""
    rng = np.random.default_rng(seed)
    n_prior = next(iter(xs.values()))["prior"].numel()
    n_grid = next(iter(xs.values()))["grid"][0].numel()
    out = {k: dict(prior=np.zeros(n_boot), worst=np.zeros(n_boot)) for k in xs}
    for b in range(n_boot):
        ip = torch.from_numpy(rng.integers(0, n_prior, n_prior))
        ig = torch.from_numpy(rng.integers(0, n_grid, n_grid))
        for k, x in xs.items():
            out[k]["prior"][b] = cvar(x["prior"][ip])
            out[k]["worst"][b] = max(cvar(g[ig]) for g in x["grid"])
    return out


def interval(samples: np.ndarray) -> tuple:
    """95% percentile interval of a bootstrap sample."""
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def mean_and_se(values) -> tuple:
    """mean and standard error over seeds."""
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
