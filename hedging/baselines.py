"""hedges that need no training: black-scholes at a chosen volatility, and the posterior-mean hedge."""
from __future__ import annotations

import torch
from torch import Tensor

from hedging.config import Calibration, T
from hedging.payoffs import bs_delta


def remaining_time(n_steps: int, like: Tensor, T: float = T) -> Tensor:
    """time to maturity at hedging dates k = 0..n-1."""
    return T - torch.arange(n_steps, dtype=like.dtype) * (T / n_steps)


def bs_delta_path(name: str, S: Tensor, sigma, T: float = T) -> Tensor:
    """black-scholes delta along paths at a fixed volatility; sigma is a scalar or a vector with one value per path."""
    n = S.shape[1] - 1
    sigma = torch.as_tensor(sigma, dtype=S.dtype)
    if sigma.dim() == 1:
        sigma = sigma[:, None]
    return bs_delta(name, S[:, :n], sigma, remaining_time(n, S, T)[None])


def posterior_over_sigma(S: Tensor, calib: Calibration, n_grid: int = 81, T: float = T):
    """posterior weights (N, n, n_grid) on a sigma grid given the returns seen up to each hedging date.

    with an inverse-gamma prior the posterior on sigma^2 is inverse gamma again: the shape grows by
    k/2 and the scale by half the realised variance divided by dt (see eq. posterior of the report)."""
    n = S.shape[1] - 1
    dt = T / n
    r = torch.log(S[:, 1:] / S[:, :-1])
    seen = torch.cat([torch.zeros_like(r[:, :1]), (r ** 2).cumsum(1)[:, :n - 1]], 1)   # sum of squared returns before date k
    k = torch.arange(n, dtype=S.dtype)[None, :]
    a = calib.a0 + k / 2
    b = calib.b0 + seen / (2 * dt)
    sigma = torch.linspace(calib.band[0], calib.band[1], n_grid, dtype=S.dtype)    # truncation to the band
    v = sigma ** 2
    log_w = -(a[..., None] + 1) * torch.log(v) - b[..., None] / v + torch.log(sigma)   # density in sigma, up to a constant
    return torch.softmax(log_w, -1), sigma


def posterior_mean_sigma(S: Tensor, calib: Calibration) -> Tensor:
    """posterior mean of sigma at each hedging date, shape (N, n)."""
    w, sigma = posterior_over_sigma(S, calib)
    return (w * sigma).sum(-1)


def posterior_mean_delta(name: str, S: Tensor, calib: Calibration) -> Tensor:
    """black-scholes delta at the posterior-mean volatility.

    this is a certainty-equivalent rule: it ignores the posterior spread and the value of future learning,
    so it is a reference and not the bayes-optimal hedge."""
    n = S.shape[1] - 1
    sigma_hat = posterior_mean_sigma(S, calib)
    return bs_delta(name, S[:, :n], sigma_hat, remaining_time(n, S)[None])
