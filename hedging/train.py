"""train one hedger by pathwise gradient descent on cvar95, with fresh simulated paths at every step.

usage: python -m hedging.train --kind gru --payoff butterfly --seed 0 --scale main"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from hedging.baselines import bs_delta_path
from hedging.config import ALPHA, MINIMAX_TAU, N_GRID, N_STEPS, PRIORS, RUNS_DIR, SCALES, load_calibration
from hedging.losses import cvar, cvar_training_loss, shortfall
from hedging.networks import checkpoint_path, count_parameters, make_hedger, run_tag
from hedging.simulate import gbm_from_noise, make_generator, simulate_prior_paths


def train(kind: str, name: str, seed: int, scale: str, prior: str = "calibrated", log_every: int = 50,
          verbose: bool = False) -> dict:
    sc = SCALES[scale]
    ckpt = checkpoint_path(scale, kind, name, seed, prior)
    log_file = RUNS_DIR / scale / "logs" / f"{run_tag(kind, name, seed, prior)}.json"
    if ckpt.exists() and log_file.exists():             # finished runs are never repeated
        return json.loads(log_file.read_text())
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(int(os.environ.get("EXP_THREADS", "1")))
    torch.manual_seed(seed)
    calib = load_calibration()
    model = make_hedger(kind, name, calib)
    gen = make_generator(seed)                          # the training noise stream depends on the seed only

    minimax = prior == "minimax"
    grid = torch.linspace(calib.band[0], calib.band[1], N_GRID)          # the constant-sigma test grid
    per_sigma = sc.batch // N_GRID
    # the cvar threshold is learned with the network; the minimax objective has one threshold per grid sigma
    w = torch.zeros(N_GRID if minimax else (), requires_grad=True)
    opt = torch.optim.Adam([{"params": model.parameters(), "lr": 1e-3}, {"params": [w], "lr": 1e-2}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, sc.steps)

    history, t0 = [], time.time()
    for step in range(sc.steps):
        if minimax:                                         # equal paths at every grid sigma, sigma never shown to the network
            sigma = grid.repeat_interleave(per_sigma)
            S = gbm_from_noise(torch.randn(sigma.numel(), N_STEPS, generator=gen), sigma)
        else:
            S, _ = simulate_prior_paths(sc.batch, gen, calib, prior=prior)   # sigma drawn from the prior
        x = shortfall(name, S, model(S))
        if minimax:
            xs = x.view(N_GRID, per_sigma)
            if step == 0:
                with torch.no_grad():
                    w.copy_(torch.quantile(xs, ALPHA, dim=1))
            ru = w + torch.relu(xs - w[:, None]).mean(1) / (1 - ALPHA)        # rockafellar-uryasev cvar per sigma
            loss = MINIMAX_TAU * torch.logsumexp(ru / MINIMAX_TAU, 0)        # smooth maximum over sigma
        else:
            if step == 0:                                   # start the threshold at the empirical value-at-risk
                with torch.no_grad():
                    w.fill_(torch.quantile(x, ALPHA).item())
            loss = cvar_training_loss(x, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == sc.steps - 1:
            rec = dict(step=step, train_cvar=cvar(x.detach()), seconds=round(time.time() - t0, 1))
            if minimax:
                cs = [cvar(r) for r in xs.detach()]
                rec.update(cvar_by_sigma=cs, worst_sigma=float(grid[int(np.argmax(cs))]), train_worst_cvar=max(cs),
                           thresholds=w.detach().tolist())
            history.append(rec)
            if verbose:
                extra = f" worst {rec['train_worst_cvar']:.4f} at {rec['worst_sigma']:.3f}" if minimax else ""
                print(f"{kind} {name} s{seed} step {step:5d} cvar95 {rec['train_cvar']:.4f}{extra}", flush=True)

    torch.save(model.state_dict(), ckpt)
    reference = cvar(shortfall(name, S, bs_delta_path(name, S, calib.sigma_ref)))   # same batch, plain black-scholes
    meta = dict(kind=kind, payoff=name, seed=seed, scale=scale, prior=prior, batch=sc.batch, steps=sc.steps,
                n_parameters=count_parameters(model), last_batch_bs_reference_cvar=reference,
                seconds=round(time.time() - t0, 1), history=history)
    log_file.write_text(json.dumps(meta, indent=1))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["gru", "mlp"], required=True)
    ap.add_argument("--payoff", choices=["butterfly", "call"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--scale", choices=list(SCALES), default="main")
    ap.add_argument("--prior", choices=list(PRIORS), default="calibrated")
    args = ap.parse_args()
    out = train(args.kind, args.payoff, args.seed, args.scale, args.prior, verbose=True)
    print(f"done in {out['seconds']} s, {out['n_parameters']} parameters")
