"""train all networks of the plan, a few single-thread worker processes at a time.

finished runs are skipped, so the script can be interrupted and restarted. the launcher waits for free memory
before starting a new worker, because ram was the limit on the development laptop."""
import argparse
import itertools
import subprocess
import sys
import time

import psutil

from hedging.config import RUNS_DIR, SCALES, SEEDS, ROOT
from hedging.networks import checkpoint_path


def main(kinds, payoffs, seeds, scale, workers, min_free_gb):
    jobs = [(k, p, s) for k, p, s in itertools.product(kinds, payoffs, seeds)
            if not checkpoint_path(scale, k, p, s).exists()]
    print(f"{len(jobs)} runs to do at scale '{scale}'", flush=True)
    running = []
    while jobs or running:
        running = [(j, proc) for j, proc in running if proc.poll() is None]        # drop finished workers
        free_gb = psutil.virtual_memory().available / 1e9
        if jobs and len(running) < workers and free_gb >= min_free_gb:
            kind, name, seed = jobs.pop(0)
            log = open(RUNS_DIR / f"{scale}_{kind}_{name}_s{seed}.out", "w") if RUNS_DIR.exists() else None
            proc = subprocess.Popen([sys.executable, "-m", "hedging.train", "--kind", kind, "--payoff", name,
                                     "--seed", str(seed), "--scale", scale], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env={**__import__("os").environ, "EXP_THREADS": "1"})
            running.append(((kind, name, seed), proc))
            print(f"started {kind} {name} seed {seed} (free ram {free_gb:.1f} GB)", flush=True)
        time.sleep(5)
    print("all runs finished")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["gru", "mlp"])
    ap.add_argument("--payoffs", nargs="+", default=["butterfly", "call"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--scale", choices=list(SCALES), default="main")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-free-gb", type=float, default=1.2)
    a = ap.parse_args()
    RUNS_DIR.mkdir(exist_ok=True)
    main(a.kinds, a.payoffs, a.seeds, a.scale, a.workers, a.min_free_gb)
