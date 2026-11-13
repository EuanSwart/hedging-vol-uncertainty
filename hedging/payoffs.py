"""european payoffs and black-scholes prices and deltas (zero rate, zero drift)."""
from __future__ import annotations

import torch
from torch import Tensor

from hedging.config import PAYOFFS

_EPS = 1e-8


def _as_tensor(x, like: Tensor) -> Tensor:
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


def _d1_and_sd(S: Tensor, strike: float, sigma, tau):
    sigma, tau = _as_tensor(sigma, S), _as_tensor(tau, S)
    sd = sigma * tau.clamp_min(_EPS).sqrt()          # total standard deviation of log price
    d1 = (torch.log(S / strike) + 0.5 * sd ** 2) / sd
    return d1, sd, tau


def bs_call_price(S: Tensor, strike: float, sigma, tau) -> Tensor:
    d1, sd, tau = _d1_and_sd(S, strike, sigma, tau)
    price = S * torch.special.ndtr(d1) - strike * torch.special.ndtr(d1 - sd)
    return torch.where(tau > _EPS, price, (S - strike).clamp_min(0.0))   # intrinsic value at maturity


def bs_call_delta(S: Tensor, strike: float, sigma, tau) -> Tensor:
    d1, _, tau = _d1_and_sd(S, strike, sigma, tau)
    return torch.where(tau > _EPS, torch.special.ndtr(d1), (S > strike).to(S.dtype))


def payoff(name: str, S_T: Tensor) -> Tensor:
    """terminal payoff g(S_T) as a sum of call payoffs."""
    return sum(c * (S_T - k).clamp_min(0.0) for k, c in PAYOFFS[name])


def bs_price(name: str, S: Tensor, sigma, tau) -> Tensor:
    return sum(c * bs_call_price(S, k, sigma, tau) for k, c in PAYOFFS[name])


def bs_delta(name: str, S: Tensor, sigma, tau) -> Tensor:
    return sum(c * bs_call_delta(S, k, sigma, tau) for k, c in PAYOFFS[name])


def reference_strike(name: str) -> float:
    """strike used to scale log-moneyness: the call strike, or the middle butterfly strike."""
    strikes = [k for k, _ in PAYOFFS[name]]
    return strikes[len(strikes) // 2]
