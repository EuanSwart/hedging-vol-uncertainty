"""training-progress figures from runs/<scale>/logs/*.json (train_cvar every 50 steps).

Several views of the same data, written to results/<scale>/training/:
  1 raw           raw curves per seed, mean, black-scholes reference line
  2 smoothed      moving-average curves, gap to the black-scholes reference in percent
  3 paired        gru minus mlp at the same seed and step (same noise stream, so batch noise cancels)
  4 progress      fraction of total improvement achieved, with steps to 50 / 90 / 99 percent
  5 priors        calibrated against uniform prior for the call, relative to the starting value
  6 seeds         final value per seed and between-seed spread over training
plus summary.md with the numbers behind the figures.

usage: python run_training_curves.py [--scale main] [--window 5]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
COL = {"gru": "tab:blue", "mlp": "tab:orange"}
PAYOFFS = ("call", "butterfly")


def load(scale: str) -> dict:
    """(kind, payoff, prior, seed) -> dict(step, cvar, ref, seconds, n_parameters)."""
    runs = {}
    for f in sorted((ROOT / "runs" / scale / "logs").glob("*.json")):
        d = json.loads(f.read_text())
        h = d["history"]
        prior = d.get("prior", "uniform" if "uniform" in f.stem else "calibrated")
        runs[(d["kind"], d["payoff"], prior, d["seed"])] = dict(
            step=np.array([p["step"] for p in h]), cvar=np.array([p["train_cvar"] for p in h]),
            ref=d["last_batch_bs_reference_cvar"], seconds=d["seconds"], n_parameters=d["n_parameters"])
    return runs


def stack(runs, kind, payoff, prior="calibrated"):
    keys = sorted(k for k in runs if k[:3] == (kind, payoff, prior))
    return (runs[keys[0]]["step"], np.array([runs[k]["cvar"] for k in keys]),
            np.array([runs[k]["ref"] for k in keys]), keys)


def smooth(y, w):
    """trailing-centred moving average along the last axis, edges shrunk instead of padded."""
    if w <= 1:
        return y
    k = np.ones(w)
    num = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), -1, y)
    den = np.convolve(np.ones(y.shape[-1]), k, "same")
    return num / den


def crossing(step, frac, level):
    """first step at which the progress fraction reaches level (nan if never)."""
    idx = np.where(frac >= level)[0]
    return step[idx[0]] if len(idx) else np.nan


def fig_raw(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, payoff in zip(axes, PAYOFFS):
        for kind in ("mlp", "gru"):
            step, y, ref, _ = stack(runs, kind, payoff)
            for r in y:
                ax.plot(step, r, color=COL[kind], alpha=0.25, lw=0.8)
            ax.plot(step, y.mean(0), color=COL[kind], lw=2, label=f"{kind.upper()} (mean of {len(y)} seeds)")
        ax.axhline(np.mean(ref), color="k", ls=":", label="BS delta (final-batch CVaR, seed mean)")
        ax.set(title=payoff, xlabel="training step", ylabel="training CVaR$_{95}$ of shortfall")
        ax.legend(fontsize=8)
    fig.suptitle("1. raw training objective (every 50 steps, fresh batch each step)")
    fig.tight_layout(); fig.savefig(out / "1_raw.png", dpi=150); plt.close(fig)


def fig_smoothed(runs, out, window):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for j, payoff in enumerate(PAYOFFS):
        for kind in ("mlp", "gru"):
            step, y, ref, _ = stack(runs, kind, payoff)
            ys = smooth(y, window)
            m, lo, hi = ys.mean(0), ys.min(0), ys.max(0)
            axes[0, j].plot(step, m, color=COL[kind], lw=2, label=kind.upper())
            axes[0, j].fill_between(step, lo, hi, color=COL[kind], alpha=0.2)
            gap = 100 * (ys / ref[:, None] - 1)                      # percent above the BS reference
            axes[1, j].plot(step, gap.mean(0), color=COL[kind], lw=2, label=kind.upper())
            axes[1, j].fill_between(step, gap.min(0), gap.max(0), color=COL[kind], alpha=0.2)
        axes[0, j].axhline(np.mean(ref), color="k", ls=":", label="BS delta")
        axes[1, j].axhline(0, color="k", ls=":")
        axes[0, j].set(title=f"{payoff}: smoothed CVaR (window {window}, band = seed range)", ylabel="CVaR$_{95}$")
        axes[1, j].set(xlabel="training step", ylabel="CVaR vs BS delta, % (negative = better)")
        axes[0, j].legend(fontsize=8); axes[1, j].legend(fontsize=8)
    fig.suptitle("2. smoothed curves and gap to the black-scholes reference")
    fig.tight_layout(); fig.savefig(out / "2_smoothed.png", dpi=150); plt.close(fig)


def fig_paired(runs, out, window):
    """same seed means same noise stream in training, so the gru-mlp difference removes batch noise."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    stats = {}
    for ax, payoff in zip(axes, PAYOFFS):
        s_g, g, _, kg = stack(runs, "gru", payoff)
        s_m, m, _, km = stack(runs, "mlp", payoff)
        assert [k[3] for k in kg] == [k[3] for k in km]
        d = g - m
        raw_sd = float(np.std(g - g.mean(1, keepdims=True)))
        for r, k in zip(d, kg):
            ax.plot(s_g, smooth(r, window), lw=1, alpha=0.6, label=f"seed {k[3]}")
        ax.plot(s_g, smooth(d, window).mean(0), color="k", lw=2.5, label="mean")
        ax.axhline(0, color="grey", ls=":")
        ax.set(title=payoff, xlabel="training step", ylabel="GRU - MLP training CVaR (negative = GRU better)")
        ax.legend(fontsize=8)
        stats[payoff] = dict(raw_sd=raw_sd, paired_sd=float(np.std(d - d.mean(1, keepdims=True))),
                             final=d[:, -10:].mean(1))
    fig.suptitle(f"3. paired GRU - MLP difference (moving average, window {window})")
    fig.tight_layout(); fig.savefig(out / "3_paired.png", dpi=150); plt.close(fig)
    return stats


def fig_progress(runs, out, window):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    table = {}
    for ax, payoff in zip(axes, PAYOFFS):
        for kind in ("mlp", "gru"):
            step, y, _, keys = stack(runs, kind, payoff)
            ys = smooth(y, window)
            start, end = ys[:, :3].mean(1, keepdims=True), ys[:, -5:].mean(1, keepdims=True)
            frac = (start - ys) / (start - end)                      # 0 at the start, 1 at the final level
            ax.plot(step, frac.mean(0), color=COL[kind], lw=2, label=kind.upper())
            ax.fill_between(step, frac.min(0), frac.max(0), color=COL[kind], alpha=0.2)
            table[(kind, payoff)] = dict(
                start=float(start.mean()), end=float(end.mean()), drop_pct=float(100 * (start - end).mean() / start.mean()),
                **{f"t{int(100*q)}": float(np.nanmean([crossing(step, f, q) for f in frac])) for q in (0.5, 0.9, 0.99)})
        for q in (0.5, 0.9):
            ax.axhline(q, color="grey", ls=":", lw=0.8)
        ax.set(title=payoff, xlabel="training step", ylabel="fraction of total CVaR reduction achieved", ylim=(-0.1, 1.1))
        ax.legend(fontsize=8)
    fig.suptitle("4. progress normalised to each run's own start and end (shows speed, not level)")
    fig.tight_layout(); fig.savefig(out / "4_progress.png", dpi=150); plt.close(fig)
    return table


def fig_priors(runs, out, window):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, prior in zip(axes, ("calibrated", "uniform")):
        for kind in ("mlp", "gru"):
            step, y, ref, _ = stack(runs, kind, "call", prior)
            ys = smooth(y, window)
            rel = 100 * (ys / ref[:, None] - 1)
            ax.plot(step, rel.mean(0), color=COL[kind], lw=2, label=kind.upper())
            ax.fill_between(step, rel.min(0), rel.max(0), color=COL[kind], alpha=0.2)
        ax.axhline(0, color="k", ls=":")
        ax.set(title=f"call, {prior} prior", xlabel="training step", ylabel="CVaR vs BS delta, % (negative = better)")
        ax.legend(fontsize=8)
    lo, hi = (min(a.get_ylim()[0] for a in axes), max(a.get_ylim()[1] for a in axes))
    for a in axes:
        a.set_ylim(lo, hi)
    fig.suptitle("5. call: training under the calibrated against the uniform sigma prior (common y-axis)")
    fig.tight_layout(); fig.savefig(out / "5_priors.png", dpi=150); plt.close(fig)


def fig_seeds(runs, out, window):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    final = {}
    for j, payoff in enumerate(PAYOFFS):
        for i, kind in enumerate(("mlp", "gru")):
            step, y, ref, keys = stack(runs, kind, payoff)
            f = y[:, -10:].mean(1)
            for s, v in zip(keys, f):
                axes[0, j].scatter(i, v, color=COL[kind], zorder=3)
                axes[0, j].annotate(f"s{s[3]}", (i, v), textcoords="offset points", xytext=(6, 0), fontsize=7)
            final[(kind, payoff)] = f
            ys = smooth(y, window)
            axes[1, j].plot(step, ys.std(0, ddof=1), color=COL[kind], lw=2, label=kind.upper())
        axes[0, j].axhline(np.mean(ref), color="k", ls=":", label="BS delta")
        axes[0, j].set_xticks([0, 1], ["MLP", "GRU"])
        axes[0, j].set(title=f"{payoff}: final CVaR per seed (mean of last 10 logs)", ylabel="CVaR$_{95}$")
        axes[1, j].set(xlabel="training step", ylabel="between-seed std of smoothed CVaR")
        axes[0, j].legend(fontsize=8); axes[1, j].legend(fontsize=8)
    fig.suptitle("6. seed variability")
    fig.tight_layout(); fig.savefig(out / "6_seeds.png", dpi=150); plt.close(fig)
    return final


def write_summary(runs, prog, paired, final, out, window):
    L = ["# training-curve summary", "", f"smoothing window {window} logs (= {50*window} steps). start = mean of first 3 logs, "
         "end = mean of last 5 logs of the smoothed curve.", "",
         "| net | payoff | params | start | end | drop % | steps to 50% | 90% | 99% | seconds |", "|---|---|---|---|---|---|---|---|---|---|"]
    for payoff in PAYOFFS:
        for kind in ("mlp", "gru"):
            p = prog[(kind, payoff)]
            keys = [k for k in runs if k[:3] == (kind, payoff, "calibrated")]
            sec = np.mean([runs[k]["seconds"] for k in keys])
            L.append(f"| {kind} | {payoff} | {runs[keys[0]]['n_parameters']} | {p['start']:.3f} | {p['end']:.3f} | {p['drop_pct']:.1f} | "
                     f"{p['t50']:.0f} | {p['t90']:.0f} | {p['t99']:.0f} | {sec:.0f} |")
    L += ["", "## final value against the black-scholes reference (last 10 logs, per seed)", "",
          "| net | payoff | final CVaR per seed | seed mean | BS ref (last batch, seed mean) | % vs BS |", "|---|---|---|---|---|---|"]
    for (kind, payoff), f in final.items():
        ref = np.mean([runs[k]["ref"] for k in runs if k[:3] == (kind, payoff, "calibrated")])
        L.append(f"| {kind} | {payoff} | {', '.join(f'{v:.3f}' for v in f)} | {f.mean():.3f} | {ref:.3f} | {100*(f.mean()/ref-1):+.1f} |")
    L += ["", "## paired GRU - MLP (same seed, same noise stream)", "",
          "| payoff | within-run noise sd (gru) | sd of paired difference | final diff per seed (last 10 logs) | mean |", "|---|---|---|---|---|"]
    for payoff, s in paired.items():
        L.append(f"| {payoff} | {s['raw_sd']:.3f} | {s['paired_sd']:.3f} | {', '.join(f'{v:+.3f}' for v in s['final'])} | {s['final'].mean():+.3f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="main")
    ap.add_argument("--window", type=int, default=5)
    a = ap.parse_args()
    runs = load(a.scale)
    out = ROOT / "results" / a.scale / "training"
    out.mkdir(parents=True, exist_ok=True)
    fig_raw(runs, out)
    fig_smoothed(runs, out, a.window)
    paired = fig_paired(runs, out, a.window)
    prog = fig_progress(runs, out, a.window)
    fig_priors(runs, out, a.window)
    final = fig_seeds(runs, out, a.window)
    write_summary(runs, prog, paired, final, out, a.window)
    print("figures in", out)


if __name__ == "__main__":
    main()
