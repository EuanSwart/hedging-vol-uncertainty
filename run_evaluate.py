"""test 1 (simulation): prior loss and worst-case loss of every hedger, differences, and the decision rules h1 and h2."""
import argparse
import json

from hedging.config import N_GRID_TEST, N_PRIOR_TEST, RESULTS_DIR, SCALES, SEEDS, load_calibration
from hedging.bsb import bsb_price
from hedging.evaluate import (bootstrap_losses, build_test_sets, free_hedges, interval, mean_and_se,
                              network_hedges, sim_shortfalls, summarize)

METRICS = ("prior_cvar", "worst_cvar", "prior_std", "worst_std")


def seed_values(table: dict, kind: str, metric: str) -> list:
    return [t[metric] for label, t in table.items() if label.startswith(kind + "_s")]


def evaluate_payoff(name: str, calib, scale: str, n_prior: int, n_grid_paths: int, n_boot: int) -> dict:
    sets = build_test_sets(calib, n_prior, n_grid_paths)
    hedges = {**free_hedges(name, calib), **network_hedges(name, calib, scale, seeds=SEEDS)}
    xs = {label: sim_shortfalls(name, h, sets) for label, h in hedges.items()}
    table = {label: summarize(x, sets.grid_sigma) for label, x in xs.items()}

    # per-kind mean and standard error over seeds
    kinds = {}
    for kind in ("gru", "mlp"):
        if seed_values(table, kind, "prior_cvar"):
            kinds[kind] = {m: mean_and_se(seed_values(table, kind, m)) for m in METRICS}

    # differences in cvar (positive means the first-named hedger is worse), per seed then averaged
    diffs = {}
    for loss in ("prior_cvar", "worst_cvar"):
        ref = {"bsb": table["bsb"][loss], "posterior_mean": table["posterior_mean"][loss]}
        if "gru" in kinds:
            g = seed_values(table, "gru", loss)
            diffs[f"value_of_learning_{loss}"] = mean_and_se([ref["bsb"] - v for v in g])           # bsb - gru
            diffs[f"gru_minus_posterior_mean_{loss}"] = mean_and_se([v - ref["posterior_mean"] for v in g])
            if "mlp" in kinds:
                m = seed_values(table, "mlp", loss)
                diffs[f"value_of_memory_{loss}"] = mean_and_se([a - b for a, b in zip(m, g)])       # mlp - gru, seed by seed
    boot = {}
    if n_boot:   # test-sampling error: paired bootstrap over test paths
        b = bootstrap_losses(xs, n_boot)
        boot = {label: dict(prior_ci95=interval(v["prior"]), worst_ci95=interval(v["worst"])) for label, v in b.items()}
    return dict(bsb_price=bsb_price(name, calib.band), hedgers=table, per_kind=kinds, differences=diffs, bootstrap=boot)


def decision_rules(res: dict) -> dict:
    """the rules fixed in the plan (section 7); a value of None means the networks are missing."""
    out = {}
    d_but, d_call = res["butterfly"]["differences"], res["call"]["differences"]
    two_se = lambda pair: 2 * pair[1]
    has_se = lambda pair: pair[1] > 0              # a standard error needs at least two seeds
    if d_but and has_se(d_but["value_of_learning_prior_cvar"]):
        out["H1_butterfly_prior"] = d_but["value_of_learning_prior_cvar"][0] > two_se(d_but["value_of_learning_prior_cvar"])
        out["H1_butterfly_worst"] = d_but["value_of_learning_worst_cvar"][0] > two_se(d_but["value_of_learning_worst_cvar"])
    if d_call and has_se(d_call["value_of_learning_worst_cvar"]):
        v = d_call["value_of_learning_worst_cvar"]
        out["H1_call_worst_within_2se_of_zero"] = abs(v[0]) <= two_se(v)
    if d_but and "value_of_memory_prior_cvar" in d_but:
        out["H2_gru_not_worse_than_mlp"] = d_but["value_of_memory_prior_cvar"][0] >= 0
        g = d_but["gru_minus_posterior_mean_prior_cvar"]
        if has_se(g):
            out["H2_gru_not_worse_than_posterior_mean"] = g[0] <= two_se(g)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=list(SCALES), default="main")
    ap.add_argument("--n-prior", type=int, default=N_PRIOR_TEST)
    ap.add_argument("--n-grid-paths", type=int, default=N_GRID_TEST)
    ap.add_argument("--n-boot", type=int, default=100)
    a = ap.parse_args()
    calib = load_calibration()
    res = {name: evaluate_payoff(name, calib, a.scale, a.n_prior, a.n_grid_paths, a.n_boot) for name in ("butterfly", "call")}
    res["decision_rules"] = decision_rules(res)
    out = RESULTS_DIR / a.scale
    out.mkdir(parents=True, exist_ok=True)
    (out / "sim_results.json").write_text(json.dumps(res, indent=1))
    for name in ("butterfly", "call"):
        print(f"\n{name} (bsb guaranteed price {res[name]['bsb_price']:.3f})")
        for label, t in res[name]["hedgers"].items():
            print(f"  {label:15s} prior {t['prior_cvar']:7.3f}  worst {t['worst_cvar']:7.3f}  (sigma {t['worst_sigma']:.3f})")
        for k, v in res[name]["differences"].items():
            print(f"  {k:42s} {v[0]:+.3f} +/- {v[1]:.3f}")
    print("\ndecision rules:", json.dumps(res["decision_rules"], indent=1))
