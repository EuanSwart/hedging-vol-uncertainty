"""train all networks of the plan, a few single-thread worker processes at a time, with memory gates.

finished runs are skipped, so the script can be interrupted and restarted. memory safety:
  - a new worker starts only if at least --min-free-gb of ram is available;
  - if available ram drops below --pause-gb, the newest worker is suspended and resumed once ram recovers;
  - available ram is logged to runs/memory_log.csv every 30 seconds.
the machine is also kept from idling into sleep while this script runs."""
import argparse
import itertools
import os
import subprocess
import sys
import time

import psutil

from hedging.config import RUNS_DIR, SCALES, SEEDS, ROOT
from hedging.keepawake import KeepAwake
from hedging.networks import checkpoint_path


def main(kinds, payoffs, seeds, scale, workers, min_free_gb, pause_gb):
    jobs = [(k, p, s) for k, p, s in itertools.product(kinds, payoffs, seeds)
            if not checkpoint_path(scale, k, p, s).exists()]
    print(f"{len(jobs)} runs to do at scale '{scale}'", flush=True)
    running, suspended, last_log = [], [], 0.0
    mem_log = open(RUNS_DIR / "memory_log.csv", "a")
    while jobs or running:
        running = [(j, p) for j, p in running if p.poll() is None]                 # drop finished workers
        suspended = [(j, p) for j, p in suspended if p.poll() is None]
        free_gb = psutil.virtual_memory().available / 1e9
        if time.time() - last_log > 30:
            mem_log.write(f"{time.strftime('%H:%M:%S')},{free_gb:.2f},{len(running)},{len(suspended)}\n")
            mem_log.flush()
            last_log = time.time()
        if free_gb < pause_gb and running:                                         # emergency: pause the newest worker
            job, proc = running.pop()
            psutil.Process(proc.pid).suspend()
            suspended.append((job, proc))
            print(f"free ram {free_gb:.2f} GB: suspended {job}", flush=True)
        elif suspended and free_gb >= min_free_gb:                                 # ram recovered: resume
            job, proc = suspended.pop()
            psutil.Process(proc.pid).resume()
            running.append((job, proc))
            print(f"free ram {free_gb:.2f} GB: resumed {job}", flush=True)
        elif jobs and not suspended and len(running) < workers and free_gb >= min_free_gb:
            kind, name, seed = jobs.pop(0)
            log = open(RUNS_DIR / f"{scale}_{kind}_{name}_s{seed}.out", "w")
            proc = subprocess.Popen([sys.executable, "-m", "hedging.train", "--kind", kind, "--payoff", name,
                                     "--seed", str(seed), "--scale", scale], cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, env={**os.environ, "EXP_THREADS": "1"})
            running.append(((kind, name, seed), proc))
            print(f"started {kind} {name} seed {seed} (free ram {free_gb:.1f} GB)", flush=True)
        time.sleep(5)
    print("all runs finished", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["gru", "mlp"])
    ap.add_argument("--payoffs", nargs="+", default=["butterfly", "call"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--scale", choices=list(SCALES), default="main")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-free-gb", type=float, default=1.5)
    ap.add_argument("--pause-gb", type=float, default=0.8)
    a = ap.parse_args()
    RUNS_DIR.mkdir(exist_ok=True)
    with KeepAwake():
        main(a.kinds, a.payoffs, a.seeds, a.scale, a.workers, a.min_free_gb, a.pause_gb)
