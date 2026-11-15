"""does the training prior explain the call's worst-case loss? compare networks trained on the calibrated prior
with networks trained on a uniform prior (far more mass on high volatility).

every hedger is scored on three things: the loss under the calibrated prior, the loss under the uniform prior,
and the worst case over the constant-sigma grid."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hedging.config import N_GRID_TEST, N_PRIOR_TEST, PRIORS, RESULTS_DIR, SCALES, SEEDS, TEST_SEED, load_calibration
from hedging.evaluate import build_test_sets, free_hedges, mean_and_se, network_hedges, sim_shortfalls, summarize
from hedging.losses import cvar, shortfall
from hedging.simulate import make_generator, simulate_prior_paths



def main(scale: str, n_prior: int, n_grid_paths: int, PAYOFF: str = "call") -> dict:
    calib = load_calibration()
    sets = build_test_sets(calib, n_prior, n_grid_paths)                       # calibrated prior and the sigma grid
    S_u, sigma_u = simulate_prior_paths(n_prior, make_generator(TEST_SEED + 7), calib, prior="uniform")

    hedges = dict(free_hedges(PAYOFF, calib))
    for prior in PRIORS:
        for label, h in network_hedges(PAYOFF, calib, scale, seeds=SEEDS, prior=prior).items():
            hedges[f"{label.split('_s')[0]}_{prior}_s{label.split('_s')[1]}"] = h

    table = {}
    for label, h in hedges.items():
        s = summarize(sim_shortfalls(PAYOFF, h, sets), sets.grid_sigma)
        table[label] = dict(calibrated_prior=s["prior_cvar"], worst_case=s["worst_cvar"], grid_cvar=s["grid_cvar"],
                            uniform_prior=cvar(shortfall(PAYOFF, S_u, h(S_u, sigma_u))))
    oracle = np.array(table["oracle"]["grid_cvar"])
    for row in table.values():                     # worst-case regret to the known-sigma hedge, over the grid
        row["worst_regret"] = float(np.max(np.array(row["grid_cvar"]) - oracle))

    per_group = {}                                                              # mean and s.e. over seeds
    for kind in ("gru", "mlp"):
        for prior in PRIORS:
            rows = [t for k, t in table.items() if k.startswith(f"{kind}_{prior}_s")]
            if rows:
                per_group[f"{kind}_{prior}"] = {m: mean_and_se([r[m] for r in rows])
                                                for m in ("calibrated_prior", "uniform_prior", "worst_case", "worst_regret")}
    result = dict(grid_sigma=sets.grid_sigma, hedgers=table, per_group=per_group)

    out = RESULTS_DIR / scale
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if PAYOFF == "call" else f"_{PAYOFF}"
    (out / f"prior_ablation{suffix}.json").write_text(json.dumps(result, indent=1))
    make_figure(result, out / f"fig_prior_ablation{suffix}.png", PAYOFF)
    write_latex_table(result, out / f"tab_prior{suffix}.tex")
    return result


def write_latex_table(result: dict, path) -> None:
    """tabular fragment for the report: call losses of networks by training prior."""
    t, g = result["hedgers"], result["per_group"]
    pm = lambda x: f"{x[0]:.2f}\\,$\\pm$\\,{x[1]:.2f}"
    lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r"Hedger & trained on & calibrated prior & uniform prior & worst case & worst regret \\", r"\midrule"]
    for name, key in (("BSB", "bsb"), ("Posterior-mean", "posterior_mean"), ("Oracle", "oracle")):
        lines.append(f"{name} & -- & {t[key]['calibrated_prior']:.2f} & {t[key]['uniform_prior']:.2f} & {t[key]['worst_case']:.2f} & {t[key]['worst_regret']:.2f} \\\\")
    for kind in ("gru", "mlp"):
        for prior in PRIORS:
            v = g.get(f"{kind}_{prior}")
            if v:
                label = "worst case over $\\sigma$" if prior == "minimax" else f"{prior} prior"
                lines.append(f"{kind.upper()} & {label} & {pm(v['calibrated_prior'])} & {pm(v['uniform_prior'])} & {pm(v['worst_case'])} & {pm(v['worst_regret'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def make_figure(result: dict, path, PAYOFF: str = "call") -> None:
    """call: required premium against true sigma for networks trained on each prior."""
    t, sigma = result["hedgers"], result["grid_sigma"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for label, colour, ls in (("bsb", "tab:red", "-"), ("oracle", "black", ":"), ("posterior_mean", "tab:green", "--")):
        ax.plot(sigma, t[label]["grid_cvar"], color=colour, ls=ls, label={"bsb": "BSB", "oracle": "oracle",
                                                                           "posterior_mean": "posterior-mean"}[label])
    for kind, colour in (("gru", "tab:blue"), ("mlp", "tab:orange")):
        for prior, ls in (("calibrated", "-"), ("uniform", "--"), ("minimax", ":")):
            curves = np.array([v["grid_cvar"] for k, v in t.items() if k.startswith(f"{kind}_{prior}_s")])
            if len(curves):
                ax.plot(sigma, curves.mean(0), color=colour, ls=ls, alpha=0.9 if prior == "calibrated" else 0.6,
                        label=f"{kind.upper()} ({prior} prior)")
    ax.set_xlabel("true volatility"); ax.set_ylabel("required premium (cvar95)"); ax.set_title(PAYOFF)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=list(SCALES), default="main")
    ap.add_argument("--n-prior", type=int, default=N_PRIOR_TEST)
    ap.add_argument("--n-grid-paths", type=int, default=N_GRID_TEST)
    ap.add_argument("--payoff", choices=["call", "butterfly"], default="call")
    a = ap.parse_args()
    res = main(a.scale, a.n_prior, a.n_grid_paths, a.payoff)
    for label in ("bsb", "posterior_mean", "oracle"):
        t = res["hedgers"][label]
        print(f"{label:16s} calibrated prior {t['calibrated_prior']:.3f}  uniform prior {t['uniform_prior']:.3f}  worst case {t['worst_case']:.3f}")
    for group, v in res["per_group"].items():
        print(f"{group:16s} " + "  ".join(f"{m} {x[0]:.3f}+/-{x[1]:.3f}" for m, x in v.items()))
