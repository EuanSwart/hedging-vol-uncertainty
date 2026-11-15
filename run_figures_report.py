"""report-style versions of fig_worst_case and fig_hedge_paths (serif, PDF + PNG), matching fig_training.

uses the saved sim_results.json and the seed-0 GRU checkpoint; nothing is retrained.
writes results/<scale>/report/fig_worst_case.{pdf,png} and fig_hedge_paths.{pdf,png}
usage: python run_figures_report.py [--scale main]
"""
import argparse
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

import run_training_figures_report as rep          # sets the shared report rcParams on import
from hedging.baselines import bs_delta_path, posterior_mean_delta
from hedging.bsb import bsb_delta_path
from hedging.config import N_STEPS, RESULTS_DIR, SEEDS, load_calibration
from hedging.networks import checkpoint_path, load_network, network_delta
from hedging.simulate import gbm_from_noise, make_generator
from run_figures import curve_by_label

# same colours as fig_training for the networks; black/grey line styles for the references
LABELS = {"bs_ref": "BS at prior mean", "bsb": "BSB (robust)", "posterior_mean": "Posterior-mean",
          "oracle": "Oracle", "mlp": "MLP", "gru": "GRU"}
STYLE = {"bs_ref": dict(color="0.55", ls="-.", lw=1.0), "bsb": dict(color="tab:red", ls="-", lw=1.3),
         "posterior_mean": dict(color="tab:green", ls="--", lw=1.3), "oracle": dict(color="k", ls=":", lw=1.4),
         "mlp": dict(color=rep.COL["mlp"], ls="-", lw=1.3), "gru": dict(color=rep.COL["gru"], ls="-", lw=1.3)}
ORDER = ("bs_ref", "bsb", "posterior_mean", "mlp", "gru", "oracle")


def fig_worst_case(sim, calib, out):
    sigma = np.linspace(calib.band[0], calib.band[1], 9)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
    for ax, name, t in zip(axes, ("butterfly", "call"), ("(a) Butterfly", "(b) Call")):
        curves = curve_by_label(sim[name]["hedgers"], "grid_cvar")
        for kind in ORDER:
            v = curves[kind]
            ax.plot(sigma, v.mean(0), label=LABELS[kind], **STYLE[kind])
            if len(v) > 1:
                se = v.std(0, ddof=1) / np.sqrt(len(v))
                ax.fill_between(sigma, v.mean(0) - se, v.mean(0) + se, color=STYLE[kind]["color"], alpha=0.2, lw=0)
        ax.axhline(sim[name]["bsb_price"], color="0.4", ls=(0, (6, 3)), lw=0.7, label="BSB price (continuous time)")
        ax.set_title(t); ax.set_xlabel(r"True volatility $\sigma$")
    axes[0].set_ylabel(r"Required premium, CVaR$_{0.95}$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
               handlelength=2.2, columnspacing=1.2)
    fig.tight_layout()
    rep.save(fig, out, "fig_worst_case")


def fig_hedge_paths(calib, scale, out):
    """hedge ratio over time on a low- and a high-volatility butterfly path that share the same noise (as run_figures)."""
    z = torch.randn(1, N_STEPS, generator=make_generator(7))
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), sharey=True)
    for ax, sigma, t in zip(axes, (0.08, 0.28), "ab"):
        S = gbm_from_noise(z, sigma)
        curves = {"bsb": bsb_delta_path("butterfly", S, calib.band)[0],
                  "posterior_mean": posterior_mean_delta("butterfly", S, calib)[0],
                  "oracle": bs_delta_path("butterfly", S, sigma)[0]}
        if checkpoint_path(scale, "gru", "butterfly", SEEDS[0]).exists():
            curves["gru"] = network_delta(load_network(scale, "gru", "butterfly", SEEDS[0], calib), S)[0]
        for kind in ("bsb", "posterior_mean", "gru", "oracle"):
            if kind in curves:
                ax.plot(curves[kind].detach().numpy(), label=LABELS[kind], **STYLE[kind])
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(rf"({t}) True volatility $\sigma={sigma:.2f}$"); ax.set_xlabel("Trading day")
    axes[0].set_ylabel("Hedge ratio")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, handlelength=2.2)
    fig.tight_layout()
    rep.save(fig, out, "fig_hedge_paths")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="main")
    a = ap.parse_args()
    calib = load_calibration()
    sim = json.loads((RESULTS_DIR / a.scale / "sim_results.json").read_text())
    out = RESULTS_DIR / a.scale / "report"
    out.mkdir(parents=True, exist_ok=True)
    fig_worst_case(sim, calib, out)
    fig_hedge_paths(calib, a.scale, out)
    print("figures in", out)


if __name__ == "__main__":
    main()
