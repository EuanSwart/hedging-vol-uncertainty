"""figures and tables from the saved results: loss against true sigma, hedge ratios on two paths, real-data bars."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from hedging.baselines import bs_delta_path, posterior_mean_delta
from hedging.bsb import bsb_delta_path
from hedging.config import N_STEPS, RESULTS_DIR, SCALES, SEEDS, load_calibration
from hedging.networks import checkpoint_path, load_network, network_delta
from hedging.simulate import gbm_from_noise, make_generator

LABELS = {"bs_ref": "BS at prior mean", "bsb": "BSB (robust)", "posterior_mean": "posterior-mean",
          "mlp": "MLP", "gru": "GRU", "oracle": "oracle", "bs_vix": "BS at VIX"}
STYLE = {"bs_ref": ("grey", ":"), "bsb": ("tab:red", "-"), "posterior_mean": ("tab:green", "--"),
         "mlp": ("tab:orange", "-"), "gru": ("tab:blue", "-"), "oracle": ("black", ":"), "bs_vix": ("tab:purple", "--")}


def curve_by_label(hedgers: dict, key: str) -> dict:
    """grid curves, with network seeds averaged and shown as a band."""
    out = {}
    for label, t in hedgers.items():
        kind = label.split("_s")[0] if "_s" in label and label[:3] in ("gru", "mlp") else label
        out.setdefault(kind, []).append(t[key])
    return {k: np.array(v) for k, v in out.items()}


def fig_worst_case(sim: dict, calib, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    sigma = np.linspace(calib.band[0], calib.band[1], 9)
    for ax, name in zip(axes, ("butterfly", "call")):
        for kind, v in curve_by_label(sim[name]["hedgers"], "grid_cvar").items():
            colour, ls = STYLE[kind]
            ax.plot(sigma, v.mean(0), color=colour, ls=ls, label=LABELS[kind])
            if len(v) > 1:
                ax.fill_between(sigma, v.mean(0) - v.std(0, ddof=1) / np.sqrt(len(v)),
                                v.mean(0) + v.std(0, ddof=1) / np.sqrt(len(v)), color=colour, alpha=0.2)
        ax.axhline(sim[name]["bsb_price"], color="tab:red", lw=0.7, alpha=0.6)    # bsb guaranteed capital
        ax.set_title(name); ax.set_xlabel("true volatility"); ax.set_ylabel("required premium (cvar95)")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_hedge_paths(calib, scale: str, path):
    """hedge ratio over time on a low- and a high-volatility butterfly path that share the same noise."""
    z = torch.randn(1, N_STEPS, generator=make_generator(7))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, sigma in zip(axes, (0.08, 0.28)):
        S = gbm_from_noise(z, sigma)
        curves = {"bsb": bsb_delta_path("butterfly", S, calib.band)[0], "oracle": bs_delta_path("butterfly", S, sigma)[0],
                  "posterior_mean": posterior_mean_delta("butterfly", S, calib)[0]}
        if checkpoint_path(scale, "gru", "butterfly", SEEDS[0]).exists():
            curves["gru"] = network_delta(load_network(scale, "gru", "butterfly", SEEDS[0], calib), S)[0]
        for kind, d in curves.items():
            colour, ls = STYLE[kind]
            ax.plot(d.numpy(), color=colour, ls=ls, label=LABELS[kind])
        ax.set_title(f"true volatility {sigma:.2f}"); ax.set_xlabel("trading day")
    axes[0].set_ylabel("hedge ratio (butterfly)"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_real(real: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, name in zip(axes, ("butterfly", "call")):
        h = real[name]["hedgers"]
        labels = list(h)
        y = [h[k]["std"] for k in labels]
        err = [[h[k]["std"] - h[k]["std_ci95"][0] for k in labels], [h[k]["std_ci95"][1] - h[k]["std"] for k in labels]]
        ax.bar(range(len(labels)), y, yerr=err, color=[STYLE[k][0] for k in labels], capsize=3)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels([LABELS[k] for k in labels], rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{name}: spy 2018-2025"); ax.set_ylabel("std of shortfall")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def write_tables(sim: dict, real: dict, path):
    """markdown tables of the numbers the report quotes."""
    lines = []
    for name in ("butterfly", "call"):
        s = sim[name]
        lines += [f"### simulation, {name} (bsb guaranteed price {s['bsb_price']:.3f})", "",
                  "| hedger | prior cvar95 | worst-case cvar95 | worst sigma |", "|---|---|---|---|"]
        for label, t in s["hedgers"].items():
            lines.append(f"| {label} | {t['prior_cvar']:.3f} | {t['worst_cvar']:.3f} | {t['worst_sigma']:.3f} |")
        lines += ["", "| difference (cvar95) | mean | s.e. over seeds |", "|---|---|---|"]
        lines += [f"| {k} | {v[0]:+.3f} | {v[1]:.3f} |" for k, v in s["differences"].items()] + [""]
    for name in ("butterfly", "call"):
        r = real[name]
        lines += [f"### real data, {name} ({r['n_windows']} windows, {100 * r['share_outside_band']:.0f}% outside the band)", "",
                  "| hedger | std | 95% ci | std minus bsb | 95% ci | top-third std |", "|---|---|---|---|---|---|"]
        for k, h in r["hedgers"].items():
            lines.append(f"| {k} | {h['std']:.3f} | [{h['std_ci95'][0]:.3f}, {h['std_ci95'][1]:.3f}] | {h['std_minus_bsb']:+.3f} | "
                         f"[{h['std_minus_bsb_ci95'][0]:+.3f}, {h['std_minus_bsb_ci95'][1]:+.3f}] | {r['top_third_std'][k]:.3f} |")
        lines.append("")
    path.write_text("\n".join(lines))


def pm(mean_se: tuple) -> str:
    """'mean $\\pm$ se' with two decimals."""
    return f"{mean_se[0]:.2f}\\,$\\pm$\\,{mean_se[1]:.2f}"


def write_latex_tables(sim: dict, real: dict, out) -> None:
    """tabular fragments for the report (the report wraps them in table environments)."""
    rows = [("BS at prior mean", "bs_ref"), ("BSB", "bsb"), ("Posterior-mean", "posterior_mean"),
            ("MLP", "mlp"), ("GRU", "gru"), ("Oracle", "oracle")]
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Butterfly} & \multicolumn{2}{c}{Call} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Hedger & prior & worst case & prior & worst case \\", r"\midrule"]
    for name, key in rows:
        cells = []
        for payoff in ("butterfly", "call"):
            for metric in ("prior_cvar", "worst_cvar"):
                if key in ("mlp", "gru"):
                    cells.append(pm(sim[payoff]["per_kind"][key][metric]))
                else:
                    cells.append(f"{sim[payoff]['hedgers'][key][metric]:.2f}")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_sim.tex").write_text("\n".join(lines) + "\n")

    rows = [("BSB", "bsb"), ("BS at VIX", "bs_vix"), ("BS at prior mean", "bs_ref"), ("MLP", "mlp"), ("GRU", "gru"),
            ("Posterior-mean", "posterior_mean")]
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Butterfly} & \multicolumn{2}{c}{Call} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Hedger & std & minus BSB & std & minus BSB \\", r"\midrule"]
    for name, key in rows:
        cells = []
        for payoff in ("butterfly", "call"):
            h = real[payoff]["hedgers"][key]
            cells.append(f"{h['std']:.2f}")
            lo, hi = h["std_minus_bsb_ci95"]
            cells.append("--" if key == "bsb" else f"{h['std_minus_bsb']:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_real.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=list(SCALES), default="main")
    a = ap.parse_args()
    out = RESULTS_DIR / a.scale
    calib = load_calibration()
    sim = json.loads((out / "sim_results.json").read_text())
    real = json.loads((out / "real_results.json").read_text())
    fig_worst_case(sim, calib, out / "fig_worst_case.png")
    fig_hedge_paths(calib, a.scale, out / "fig_hedge_paths.png")
    fig_real(real, out / "fig_real.png")
    write_tables(sim, real, out / "tables.md")
    write_latex_tables(sim, real, out)
    print("wrote figures and tables to", out)
