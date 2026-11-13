"""optional probe (h2): does the gru hidden state linearly encode the posterior-mean volatility?

fit on half of the test paths, score r^2 on the other half, and compare with an untrained gru of the same seed."""
import argparse
import json

import numpy as np
import torch

from hedging.baselines import posterior_mean_sigma
from hedging.config import RESULTS_DIR, SCALES, SEEDS, TEST_SEED, load_calibration
from hedging.networks import checkpoint_path, load_network, make_hedger
from hedging.simulate import make_generator, simulate_prior_paths


def ridge_r2(train_x, train_y, test_x, test_y, penalty: float = 1e-6) -> float:
    """linear regression with an intercept and a tiny ridge penalty; returns held-out r^2."""
    add_one = lambda a: np.hstack([a, np.ones((len(a), 1))])
    a, b = add_one(train_x), add_one(test_x)
    coef = np.linalg.solve(a.T @ a + penalty * np.eye(a.shape[1]), a.T @ train_y)
    resid = ((test_y - b @ coef) ** 2).sum()
    return float(1 - resid / ((test_y - test_y.mean()) ** 2).sum())


@torch.no_grad()
def probe_one(model, S, target) -> float:
    h = model.hidden_states(S)[:, 1:].reshape(-1, model.gru.hidden_size).numpy()   # dates k >= 1, pooled
    y = target[:, 1:].reshape(-1).numpy()
    half = len(h) // 2          # rows of the same path are adjacent, so the split keeps paths apart
    return ridge_r2(h[:half], y[:half], h[half:], y[half:])


def main(scale: str, name: str, n_paths: int) -> dict:
    calib = load_calibration()
    S, _ = simulate_prior_paths(n_paths, make_generator(TEST_SEED + 1), calib)
    target = posterior_mean_sigma(S, calib)
    out = {}
    for seed in SEEDS:
        if not checkpoint_path(scale, "gru", name, seed).exists():
            continue
        torch.manual_seed(seed)
        untrained = make_hedger("gru", name, calib)            # same architecture, random weights
        out[seed] = dict(trained=probe_one(load_network(scale, "gru", name, seed, calib), S, target),
                         untrained=probe_one(untrained, S, target))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=list(SCALES), default="main")
    ap.add_argument("--n-paths", type=int, default=20000)
    a = ap.parse_args()
    res = {name: main(a.scale, name, a.n_paths) for name in ("butterfly", "call")}
    out = RESULTS_DIR / a.scale
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
