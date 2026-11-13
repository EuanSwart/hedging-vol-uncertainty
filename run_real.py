"""test 2 (real spy data, no retraining): hedging error on overlapping 63-day windows of 2018-2025.

the networks and the band come from simulation and the 2010-2017 calibration, so this period is out of sample.
with so few windows the main metric is the standard deviation of the shortfall; cvar95 is shown for completeness."""
import argparse
import json

import numpy as np
import torch

from hedging.baselines import bs_delta_path, posterior_mean_delta
from hedging.bsb import bsb_delta_path
from hedging.config import RESULTS_DIR, SCALES, SEEDS, TEST_START, load_calibration
from hedging.evaluate import interval
from hedging.losses import cvar, shortfall
from hedging.networks import checkpoint_path, load_network, network_delta
from hedging.simulate import load_real_windows

BLOCK = 3            # windows per bootstrap block, about one non-overlapping quarter at stride 21


def window_shortfalls(name: str, win: dict, calib, scale: str) -> dict:
    """shortfall of every hedger on every real window; networks are kept per seed as 'gru_s0', ..."""
    S = win["S"]
    x = {
        "bs_ref": shortfall(name, S, bs_delta_path(name, S, calib.sigma_ref)),
        "bs_vix": shortfall(name, S, bs_delta_path(name, S, win["vix_start"] / 100)),    # market practice: implied volatility
        "bsb": shortfall(name, S, bsb_delta_path(name, S, calib.band)),
        "posterior_mean": shortfall(name, S, posterior_mean_delta(name, S, calib)),
    }
    for kind in ("gru", "mlp"):
        for seed in SEEDS:
            if checkpoint_path(scale, kind, name, seed).exists():
                model = load_network(scale, kind, name, seed, calib)
                x[f"{kind}_s{seed}"] = shortfall(name, S, network_delta(model, S))
    return {k: v.numpy() for k, v in x.items()}


def collapse_seeds(stats: dict) -> dict:
    """average the per-seed statistics of each network kind into one number per hedger."""
    out = {k: v for k, v in stats.items() if "_s" not in k}
    for kind in ("gru", "mlp"):
        vals = [v for k, v in stats.items() if k.startswith(kind + "_s")]
        if vals:
            out[kind] = float(np.mean(vals))
    return out


def std_by_hedger(x: dict, idx) -> dict:
    return collapse_seeds({k: float(v[idx].std()) for k, v in x.items()})


def block_bootstrap(x: dict, n_boot: int, seed: int = 0) -> dict:
    """moving-block bootstrap over windows: overlapping windows are dependent, so whole blocks are resampled."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(x.values())))
    n_blocks = int(np.ceil(n / BLOCK))
    draws = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - BLOCK + 1, n_blocks)
        idx = (starts[:, None] + np.arange(BLOCK)).ravel()[:n]
        draws.append(std_by_hedger(x, idx))
    return {k: np.array([d[k] for d in draws]) for k in draws[0]}


def evaluate_real(name: str, calib, scale: str, n_boot: int) -> dict:
    win = load_real_windows(TEST_START)
    x = window_shortfalls(name, win, calib, scale)
    n = len(win["start_date"])
    allidx = np.arange(n)
    boot = block_bootstrap(x, n_boot)
    point = std_by_hedger(x, allidx)
    top = np.argsort(win["realised_vol"].numpy())[-(n // 3):]          # top third by realised volatility (ex post)
    vol = win["realised_vol"].numpy()
    res = dict(n_windows=n, share_outside_band=float(((vol < calib.band[0]) | (vol > calib.band[1])).mean()),
               hedgers={}, top_third_std=std_by_hedger(x, top))
    for k, v in point.items():
        diff = boot[k] - boot["bsb"]                                     # difference in std against bsb (negative: better)
        res["hedgers"][k] = dict(std=v, std_ci95=interval(boot[k]), std_minus_bsb=float(v - point["bsb"]),
                                 std_minus_bsb_ci95=interval(diff),
                                 mean=float(np.mean([m.mean() for kk, m in x.items() if kk == k or kk.startswith(k + "_s")])),
                                 cvar95=float(np.mean([cvar(torch.from_numpy(m)) for kk, m in x.items()
                                                       if kk == k or kk.startswith(k + "_s")])))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=list(SCALES), default="main")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    calib = load_calibration()
    res = {name: evaluate_real(name, calib, a.scale, a.n_boot) for name in ("butterfly", "call")}
    out = RESULTS_DIR / a.scale
    out.mkdir(parents=True, exist_ok=True)
    (out / "real_results.json").write_text(json.dumps(res, indent=1))
    for name, r in res.items():
        print(f"\n{name}: {r['n_windows']} windows, {100 * r['share_outside_band']:.0f}% outside the band")
        for k, h in r["hedgers"].items():
            lo, hi = h["std_minus_bsb_ci95"]
            print(f"  {k:15s} std {h['std']:6.3f}  mean {h['mean']:+7.3f}  vs bsb {h['std_minus_bsb']:+.3f} [{lo:+.3f}, {hi:+.3f}]  top third {r['top_third_std'][k]:6.3f}")
