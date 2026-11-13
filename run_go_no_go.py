"""step 3 of the plan: can the free hedges be told apart at all? run before any training.

compares bsb with the posterior-mean hedge on common random numbers with a paired bootstrap. if the interval of
the butterfly difference contains zero, the experiment cannot show a gain from learning and needs rethinking."""
import argparse
import json

from hedging.bsb import bsb_price
from hedging.config import N_GRID_TEST, RESULTS_DIR, load_calibration
from hedging.evaluate import (bootstrap_losses, build_test_sets, free_hedges, interval, sim_shortfalls, summarize)


def main(n_prior: int, n_grid_paths: int, n_boot: int) -> dict:
    calib = load_calibration()
    report = dict(band=calib.band, n_prior=n_prior, n_grid_paths=n_grid_paths)
    for name in ("butterfly", "call"):
        sets = build_test_sets(calib, n_prior, n_grid_paths)
        hedges = free_hedges(name, calib)
        xs = {k: sim_shortfalls(name, h, sets) for k, h in hedges.items()}
        boot = bootstrap_losses(xs, n_boot)
        table = {k: summarize(x, sets.grid_sigma) for k, x in xs.items()}
        diff = {}
        for loss in ("prior", "worst"):
            d = boot["bsb"][loss] - boot["posterior_mean"][loss]          # positive: bsb is worse than the posterior-mean hedge
            diff[loss] = dict(mean=float(d.mean()), ci95=interval(d), resolvable=bool(interval(d)[0] > 0))
        report[name] = dict(bsb_price=bsb_price(name, calib.band), hedges=table, bsb_minus_posterior_mean=diff)
        print(f"\n{name}: bsb guaranteed price {report[name]['bsb_price']:.3f}")
        for k, t in table.items():
            print(f"  {k:15s} prior cvar95 {t['prior_cvar']:7.3f}   worst-case cvar95 {t['worst_cvar']:7.3f} (sigma {t['worst_sigma']:.3f})")
        for loss, d in diff.items():
            print(f"  bsb - posterior_mean, {loss:5s}: {d['mean']:+.3f}  95% ci [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  resolvable: {d['resolvable']}")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "go_no_go.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prior", type=int, default=200_000)
    ap.add_argument("--n-grid-paths", type=int, default=N_GRID_TEST)
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args()
    main(a.n_prior, a.n_grid_paths, a.n_boot)
