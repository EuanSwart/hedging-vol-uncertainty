"""report-quality training figures (PDF + PNG) built from the same data as run_training_curves.py.

writes results/<scale>/training/report/fig_training.{pdf,png} and fig_training_prior.{pdf,png}
usage: python run_training_figures_report.py [--scale main] [--window 5]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_training_curves import ROOT, COL, PAYOFFS, load, stack, smooth

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.titlesize": 9, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": True,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
try:  # prefer a Computer-Modern-like serif if installed
    from matplotlib import font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    for cand in ("CMU Serif", "STIXGeneral"):
        if cand in names:
            plt.rcParams["font.serif"] = [cand]; break
except Exception:
    pass

YREL = "CVaR rel. to BS delta (%)"


def tag(ax, s):
    ax.text(0.02, 0.97, s, transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")


def rel_curves(runs, kind, payoff, prior, window):
    step, y, ref, _ = stack(runs, kind, payoff, prior)
    return step, 100 * (smooth(y, window) / ref[:, None] - 1)


def band(ax, step, v, color, label=None, lw=1.3, alpha=0.2):
    ax.plot(step, v.mean(0), color=color, lw=lw, label=label)
    ax.fill_between(step, v.min(0), v.max(0), color=color, alpha=alpha, lw=0)


def tight_ylim(ax, curves, pad=0.06, top=0.06):
    lo = min(c.min() for c in curves); hi = max(c.max() for c in curves)
    lo, hi = min(lo, 0), max(hi, 0)
    p = pad * (hi - lo)
    ax.set_ylim(lo - p, hi + top * (hi - lo))


def save(fig, out, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(out / f"{name}.{ext}", **kw)
    plt.close(fig)


def fig_training(runs, out, w):
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6), sharex="col")
    for j, payoff in enumerate(PAYOFFS):
        ax = axes[0, j]
        allc = []
        for kind in ("gru", "mlp"):
            step, rel = rel_curves(runs, kind, payoff, "calibrated", w)
            band(ax, step, rel, COL[kind], kind.upper())
            allc += [rel.min(0), rel.max(0)]
        ax.axhline(0, color="k", ls=":", lw=0.8)
        tight_ylim(ax, allc, top=0.3)
        ax.set_title("Call" if j == 0 else "Butterfly")
        tag(ax, "(a)" if j == 0 else "(b)")
        ax = axes[1, j]
        sg, g, _, _ = stack(runs, "gru", payoff)
        _, m, _, _ = stack(runs, "mlp", payoff)
        d = smooth(g - m, w)
        ax.plot(sg, d.mean(0), color="k", lw=1.3)
        ax.fill_between(sg, d.min(0), d.max(0), color="0.6", alpha=0.35, lw=0)
        ax.axhline(0, color="k", ls=":", lw=0.8)
        tight_ylim(ax, [d.min(0), d.max(0)])
        ax.set_xlabel("Training step")
        tag(ax, "(c)" if j == 0 else "(d)")
    axes[0, 0].set_ylabel(YREL)
    axes[1, 0].set_ylabel("GRU $-$ MLP CVaR")
    h, l = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(h, l, loc="upper right", frameon=False, handlelength=1.5, ncol=2, columnspacing=1.0, borderaxespad=0.2)
    fig.tight_layout()
    save(fig, out, "fig_training")


def fig_prior(runs, out, w):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4), sharey=True)
    allc = []
    for ax, prior, t, ttl in zip(axes, ("calibrated", "uniform"), "ab", ("Calibrated prior", "Uniform prior")):
        for kind in ("gru", "mlp"):
            step, rel = rel_curves(runs, kind, "call", prior, w)
            band(ax, step, rel, COL[kind], kind.upper())
            allc += [rel.min(0), rel.max(0)]
        ax.axhline(0, color="k", ls=":", lw=0.8)
        ax.set_title(ttl); ax.set_xlabel("Training step"); tag(ax, f"({t})")
    tight_ylim(axes[0], allc, top=0.3)
    axes[0].set_ylabel(YREL)
    axes[1].legend(loc="upper right", frameon=False, handlelength=1.5, ncol=2, columnspacing=1.0, borderaxespad=0.2)
    fig.tight_layout()
    save(fig, out, "fig_training_prior")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="main")
    ap.add_argument("--window", type=int, default=5)
    a = ap.parse_args()
    runs = load(a.scale)
    out = ROOT / "results" / a.scale / "training" / "report"
    out.mkdir(parents=True, exist_ok=True)
    fig_training(runs, out, a.window)
    fig_prior(runs, out, a.window)
    print("figures in", out)


if __name__ == "__main__":
    main()
