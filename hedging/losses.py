"""shortfall and cvar. the seller starts with zero cash, so cvar95 of the shortfall is the required premium."""
from __future__ import annotations

import torch
from torch import Tensor

from hedging.config import ALPHA
from hedging.payoffs import payoff


def shortfall(name: str, S: Tensor, delta: Tensor) -> Tensor:
    """X = g(S_n) - sum_k delta_k (S_{k+1} - S_k); positive X is a loss for the seller."""
    gains = (delta * (S[:, 1:] - S[:, :-1])).sum(1)
    return payoff(name, S[:, -1]) - gains


def cvar_training_loss(x: Tensor, w: Tensor, alpha: float = ALPHA) -> Tensor:
    """rockafellar-uryasev form: w + E[(X - w)+] / (1 - alpha); its minimum over w is cvar_alpha(X)."""
    return w + torch.relu(x - w).mean() / (1 - alpha)


@torch.no_grad()
def cvar(x: Tensor, alpha: float = ALPHA) -> float:
    """empirical cvar: the mean of the worst (1 - alpha) share of the sample."""
    k = max(1, round((1 - alpha) * x.numel()))
    return float(torch.topk(x.flatten(), k).values.mean())
