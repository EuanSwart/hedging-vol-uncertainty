"""neural hedgers. both see (log-moneyness, time to maturity, latest return); only the gru has a hidden state."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from hedging.baselines import remaining_time
from hedging.config import RUNS_DIR, T, Calibration
from hedging.payoffs import bs_delta, reference_strike

LOG_MONEYNESS_SCALE = 0.1
N_FEATURES = 3


def build_features(S: Tensor, name: str, sigma_ref: float, T: float = T) -> Tensor:
    """features (N, n, 3) at hedging dates k = 0..n-1; they use prices up to S_k only (no look-ahead)."""
    n = S.shape[1] - 1
    dt = T / n
    log_moneyness = torch.log(S[:, :n] / reference_strike(name)) / LOG_MONEYNESS_SCALE
    time_left = (1 - torch.arange(n, dtype=S.dtype) / n).expand_as(log_moneyness)
    r = torch.log(S[:, 1:] / S[:, :-1])                               # r[:, j] is the return into date j+1
    latest = torch.cat([torch.zeros_like(r[:, :1]), r[:, :n - 1]], 1)   # the return into date k, zero at k = 0
    latest = latest / (sigma_ref * math.sqrt(dt))                       # scaled to roughly unit size
    return torch.stack([log_moneyness, time_left, latest], -1)


class Hedger(nn.Module):
    """delta_k = black-scholes delta at the reference volatility + a learned correction.

    the output layer starts at zero, so training starts from the plain black-scholes hedge."""

    def __init__(self, name: str, sigma_ref: float, T: float = T):
        super().__init__()
        self.name, self.sigma_ref, self.T = name, sigma_ref, T

    def base_delta(self, S: Tensor) -> Tensor:
        n = S.shape[1] - 1
        return bs_delta(self.name, S[:, :n], self.sigma_ref, remaining_time(n, S, self.T)[None])

    def features(self, S: Tensor) -> Tensor:
        return build_features(S, self.name, self.sigma_ref, self.T)

    def correction(self, features: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, S: Tensor) -> Tensor:
        return self.base_delta(S) + self.correction(self.features(S))


def _zero_init(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class MLPHedger(Hedger):
    """memoryless: every date is processed on its own, so it cannot accumulate evidence about sigma."""

    def __init__(self, name: str, sigma_ref: float, hidden: int = 32, T: float = T):
        super().__init__(name, sigma_ref, T)
        self.body = nn.Sequential(nn.Linear(N_FEATURES, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, 1)
        _zero_init(self.head)

    def correction(self, features: Tensor) -> Tensor:
        return self.head(self.body(features)).squeeze(-1)


class GRUHedger(Hedger):
    """recurrent: the hidden state can carry a running summary of past returns, such as realised variance."""

    def __init__(self, name: str, sigma_ref: float, hidden: int = 16, T: float = T):
        super().__init__(name, sigma_ref, T)
        self.gru = nn.GRU(N_FEATURES, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        _zero_init(self.head)

    def hidden_states(self, S: Tensor) -> Tensor:
        """hidden state (N, n, H) after reading the features of each date."""
        return self.gru(self.features(S))[0]

    def correction(self, features: Tensor) -> Tensor:
        return self.head(self.gru(features)[0]).squeeze(-1)


def make_hedger(kind: str, name: str, calib: Calibration, hidden: int | None = None) -> Hedger:
    """kind is 'gru' (hidden 16) or 'mlp' (two layers of 32); the sizes give comparable parameter counts."""
    if kind == "gru":
        return GRUHedger(name, calib.sigma_ref, hidden or 16)
    if kind == "mlp":
        return MLPHedger(name, calib.sigma_ref, hidden or 32)
    raise ValueError(f"unknown hedger kind {kind!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_tag(kind: str, name: str, seed: int, prior: str = "calibrated") -> str:
    """file stem of a run; the default prior keeps the names of the main experiment unchanged."""
    return f"{kind}_{name}_s{seed}" if prior == "calibrated" else f"{kind}_{name}_{prior}_s{seed}"


def checkpoint_path(scale: str, kind: str, name: str, seed: int, prior: str = "calibrated"):
    return RUNS_DIR / scale / "ckpt" / f"{run_tag(kind, name, seed, prior)}.pt"


def load_network(scale: str, kind: str, name: str, seed: int, calib: Calibration, prior: str = "calibrated") -> Hedger:
    model = make_hedger(kind, name, calib)
    model.load_state_dict(torch.load(checkpoint_path(scale, kind, name, seed, prior), weights_only=True))
    return model.eval()


@torch.no_grad()
def network_delta(model: Hedger, S: Tensor, chunk: int = 20000) -> Tensor:
    """deltas for many paths, in chunks to keep memory low."""
    return torch.cat([model(S[i:i + chunk]) for i in range(0, S.shape[0], chunk)])
