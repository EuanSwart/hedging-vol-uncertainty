"""train one hedger by pathwise gradient descent on cvar95, with fresh simulated paths at every step.

usage: python -m hedging.train --kind gru --payoff butterfly --seed 0 --scale main"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from hedging.baselines import bs_delta_path
from hedging.config import ALPHA, PRIORS, RUNS_DIR, SCALES, load_calibration
from hedging.losses import cvar, cvar_training_loss, shortfall
from hedging.networks import checkpoint_path, count_parameters, make_hedger, run_tag
from hedging.simulate import make_generator, simulate_prior_paths


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

    w = torch.zeros((), requires_grad=True)             # the cvar threshold is learned with the network
    opt = torch.optim.Adam([{"params": model.parameters(), "lr": 1e-3}, {"params": [w], "lr": 1e-2}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, sc.steps)

    history, t0 = [], time.time()
    for step in range(sc.steps):
        S, _ = simulate_prior_paths(sc.batch, gen, calib, prior=prior)   # sigma is drawn from the prior, never shown to the network
        x = shortfall(name, S, model(S))
        if step == 0:                                       # start the threshold at the empirical value-at-risk
            with torch.no_grad():
                w.fill_(torch.quantile(x, ALPHA).item())
        loss = cvar_training_loss(x, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == sc.steps - 1:
            history.append(dict(step=step, train_cvar=cvar(x.detach()), seconds=round(time.time() - t0, 1)))
            if verbose:
                print(f"{kind} {name} s{seed} step {step:5d} cvar95 {history[-1]['train_cvar']:.4f}", flush=True)

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
