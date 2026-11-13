"""black-scholes-barenblatt (uncertain volatility) price and hedge by explicit finite differences."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from hedging.config import CACHE_DIR, N_STEPS, S0, T
from hedging.payoffs import payoff


def solve_bsb(name: str, band: tuple, n_steps: int = N_STEPS, T: float = T, n_grid: int = 600):
    """solve V_tau = 0.5 * s(Gamma)^2 * S^2 * Gamma with s = sigma_hi if Gamma > 0 else sigma_lo.

    returns the price grid, the delta table (n_steps, n_grid) at the hedging dates,
    and the price at time zero on the grid."""
    lo, hi = band
    s_grid = np.exp(np.linspace(math.log(30.0), math.log(400.0), n_grid))
    x = np.log(s_grid)
    dx = x[1] - x[0]
    v = payoff(name, torch.from_numpy(s_grid)).numpy().astype(np.float64)   # value at maturity

    def delta_and_s2gamma(v):
        vx = np.gradient(v, x)
        vxx = np.zeros_like(v)
        vxx[1:-1] = (v[2:] - 2 * v[1:-1] + v[:-2]) / dx ** 2
        return np.gradient(v, s_grid), vxx - vx           # delta, and S^2 * Gamma in log coordinates

    dt = T / n_steps
    n_sub = int(math.ceil(dt * hi ** 2 / (0.4 * dx ** 2)))   # substeps keep the explicit scheme stable
    h = dt / n_sub
    delta = np.zeros((n_steps, n_grid))
    for j in range(n_steps):                               # march backwards in time, tau = (j+1) dt
        for _ in range(n_sub):
            _, g = delta_and_s2gamma(v)
            v_new = v.copy()
            v_new[1:-1] = v[1:-1] + h * 0.5 * np.where(g > 0, hi ** 2, lo ** 2)[1:-1] * g[1:-1]
            for edge, a, b in ((0, 1, 2), (n_grid - 1, n_grid - 2, n_grid - 3)):   # linear extrapolation
                v_new[edge] = v_new[a] + (v_new[a] - v_new[b]) * (s_grid[edge] - s_grid[a]) / (s_grid[a] - s_grid[b])
            v = v_new
        delta[n_steps - 1 - j] = delta_and_s2gamma(v)[0]    # remaining time (j+1) dt is hedging date n-1-j
    return s_grid, delta, v


def load_bsb(name: str, band: tuple, n_steps: int = N_STEPS, T: float = T, n_grid: int = 600):
    """cached solve: the table is small and cheap to recompute, so the cache is not committed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"bsb_{name}_{band[0]}_{band[1]}_n{n_steps}_T{T}_M{n_grid}.npz"
    if f.exists():
        d = np.load(f)
        return d["s_grid"], d["delta"], d["price0"]
    s_grid, delta, price0 = solve_bsb(name, band, n_steps, T, n_grid)
    np.savez(f, s_grid=s_grid, delta=delta, price0=price0)
    return s_grid, delta, price0


def bsb_price(name: str, band: tuple, s0: float = S0, **kw) -> float:
    """capital that hedges without loss against every volatility path in the band."""
    s_grid, _, price0 = load_bsb(name, band, **kw)
    return float(np.interp(s0, s_grid, price0))


def bsb_delta_path(name: str, S: Tensor, band: tuple) -> Tensor:
    """bsb delta along price paths S (N, n+1), linearly interpolated in log price; returns (N, n)."""
    n = S.shape[1] - 1
    s_grid, table, _ = load_bsb(name, band, n_steps=n)
    xg = torch.log(torch.from_numpy(s_grid)).float()
    d = torch.from_numpy(table).float()                       # (n, grid)
    x = torch.log(S[:, :n].clamp(float(s_grid[0]), float(s_grid[-1])))
    idx = torch.searchsorted(xg, x.contiguous()).clamp(1, len(xg) - 1)
    w = (x - xg[idx - 1]) / (xg[idx] - xg[idx - 1])
    k = torch.arange(n)[None, :].expand_as(idx)
    return d[k, idx - 1] * (1 - w) + d[k, idx] * w
