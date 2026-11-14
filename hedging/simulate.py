"""price paths: simulated geometric brownian motion with a per-path volatility, and real spy windows."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from scipy.special import gammaincc, gammainccinv
from torch import Tensor

from hedging.config import Calibration, N_STEPS, S0, SPY_CSV, T


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def sample_prior_sigma(n_paths: int, gen: torch.Generator, calib: Calibration, prior: str = "calibrated") -> Tensor:
    """draw sigma from the prior: the inverse-gamma on sigma^2 truncated to the band (inverse-cdf sampling),
    or, for prior='uniform', uniform on the band, which puts far more mass on high volatility."""
    lo, hi = calib.band
    if prior == "uniform":
        return (lo + (hi - lo) * torch.rand(n_paths, generator=gen)).float()
    if prior != "calibrated":
        raise ValueError(f"unknown prior {prior!r}")
    a, b = calib.a0, calib.b0
    cdf = lambda v: gammaincc(a, b / v)                # cdf of the inverse gamma at variance v
    u = torch.rand(n_paths, generator=gen, dtype=torch.float64).numpy()
    u = cdf(lo ** 2) + (cdf(hi ** 2) - cdf(lo ** 2)) * u    # uniform on the truncated cdf range
    variance = b / gammainccinv(a, u)
    return torch.from_numpy(np.sqrt(variance)).float()


def gbm_from_noise(z: Tensor, sigma, T: float = T) -> Tensor:
    """paths (N, n+1) from standard normal noise z (N, n); sigma is a scalar or a vector of length N."""
    n = z.shape[1]
    dt = T / n
    sigma = torch.as_tensor(sigma, dtype=z.dtype)
    if sigma.dim() == 1:
        sigma = sigma[:, None]
    log_ret = -0.5 * sigma ** 2 * dt + sigma * math.sqrt(dt) * z    # the drift makes S a martingale
    log_s = torch.cat([torch.zeros_like(log_ret[:, :1]), log_ret.cumsum(1)], 1)
    return S0 * torch.exp(log_s)


def simulate_prior_paths(n_paths: int, gen: torch.Generator, calib: Calibration, n_steps: int = N_STEPS,
                         prior: str = "calibrated"):
    """paths whose volatility is drawn once per path from the prior; returns (S, sigma)."""
    sigma = sample_prior_sigma(n_paths, gen, calib, prior)
    z = torch.randn(n_paths, n_steps, generator=gen)
    return gbm_from_noise(z, sigma), sigma


def load_real_windows(start: str, n_steps: int = N_STEPS, stride: int = 21, csv=SPY_CSV) -> dict:
    """overlapping windows of daily spy closes, each rebased to S0, with the vix on the first day."""
    df = pd.read_csv(csv, parse_dates=["date"]).dropna()
    df = df[df["date"] >= start].reset_index(drop=True)
    spy, vix = df["spy"].to_numpy(float), df["vix"].to_numpy(float)
    starts = list(range(0, len(spy) - n_steps, stride))
    S = torch.tensor(np.stack([spy[i:i + n_steps + 1] / spy[i] * S0 for i in starts]), dtype=torch.float32)
    log_ret = torch.log(S[:, 1:] / S[:, :-1])
    realised_vol = torch.sqrt(252.0 * (log_ret ** 2).mean(1))   # annualised, zero-mean convention
    return dict(S=S, vix_start=torch.tensor(vix[starts], dtype=torch.float32),
                realised_vol=realised_vol, start_date=[str(df["date"][i].date()) for i in starts])
