"""Grid sweep over lambda_phi_mstep (the M-step phase weight) and lambda_R.

Why this exists: neither weight has a derivation. lambda_phi = 3 is the
document's worked-example value (config.py) and lambda_R is a calibration
(config.py), and yesterday's gradient measurement (timing gradient ~150x the
phase gradient after warm-up) says lambda_phi is probably far too small. The
only way to settle either is to train a grid and score it the way the
baselines are scored.

Two phases, one script:

  python sweep.py launch  --lambda-phi-mstep 30 100 300 --lambda-r 0 4000 \
                          --gpus 0 1 2 3 --per-gpu 2 --fold 0 --max-epochs 40
  python sweep.py collect --lambda-phi-mstep 30 100 300 --lambda-r 0 4000 --fold 0 --gpu 0

`launch` runs train.py once per (lambda_phi, lambda_R) pair, spread over the
given GPUs with at most --per-gpu runs sharing a GPU, and waits for all of
them. Each run is named sweep_lpm{phi}_lr{R} so its checkpoint and log are
identifiable; a run whose final checkpoint already exists is skipped unless
--force. `collect` scores each run's final checkpoint with evaluate.py on
the fold's middle excerpts (the baselines' protocol) and prints one table
sorted by F, also written to sweep_results.csv. Everything not about the
grid (seed, epochs, workers, ...) is passed through to train.py verbatim
after `--`.
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import config

REPO = Path(__file__).resolve().parent
PY = sys.executable                     # the interpreter running this script


def run_name(lp: float, lr: float) -> str:
    return f"sweep_lpm{lp:g}_lr{lr:g}"


def checkpoint_glob(name: str, seed: int, fold: int) -> str:
    # train.py: f"{name} S{seed} {params_str}" with params_str starting "fold{fold} phase-..."
    return str(REPO / "checkpoints" / f"{name} S{seed} fold{fold} *.ckpt")


def train_command(lp, lr, gpu, args, passthrough):
    return [PY, str(REPO / "train.py"),
            "--name", run_name(lp, lr), "--gpu", str(gpu),
            "--fold", str(args.fold), "--seed", str(args.seed),
            "--max-epochs", str(args.max_epochs),
            "--lambda-phi-mstep", str(lp), "--lambda_r", str(lr),
            # one final checkpoint per run, no per-epoch snapshots: the sweep
            # compares end points, and 16 runs x 8 snapshots is 12 GB
            "--snapshot_every", "0", "--val-frequency", str(args.max_epochs),
            "--logger", "none", *passthrough]


def launch(args, passthrough):
    grid = [(lp, lr) for lp in args.lambda_phi_mstep for lr in args.lambda_r]
    log_dir = REPO / "sweep_logs"
    log_dir.mkdir(exist_ok=True)

    todo = []
    for lp, lr in grid:
        existing = glob.glob(checkpoint_glob(run_name(lp, lr), args.seed, args.fold))
        if existing and not args.force:
            print(f"skip {run_name(lp, lr)}: checkpoint exists ({Path(existing[0]).name})")
            continue
        todo.append((lp, lr))
    print(f"{len(todo)} run(s) to launch over GPUs {args.gpus}, {args.per_gpu} per GPU")

    # Simple scheduler: a slot is (gpu, running Popen or None). A run takes
    # the first free slot; when none is free, poll until one finishes.
    slots = [[g, None] for g in args.gpus for _ in range(args.per_gpu)]
    pending = list(todo)
    started = {}
    while pending or any(p is not None for _, p in slots):
        for slot in slots:
            gpu, proc = slot
            if proc is not None and proc.poll() is not None:
                name = started[proc.pid]
                status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
                print(f"[{time.strftime('%H:%M:%S')}] {name} finished: {status}")
                slot[1] = None
            if slot[1] is None and pending:
                lp, lr = pending.pop(0)
                name = run_name(lp, lr)
                cmd = train_command(lp, lr, gpu, args, passthrough)
                if args.dry_run:
                    print(" ".join(cmd))
                    continue
                log = open(log_dir / f"{name}.log", "w")
                env = dict(os.environ, PYTHONBREAKPOINT="0")
                proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=env)
                started[proc.pid] = name
                slot[1] = proc
                print(f"[{time.strftime('%H:%M:%S')}] {name} -> GPU {gpu}, pid {proc.pid}, log {log.name}")
        if args.dry_run:
            return
        time.sleep(args.poll)
    print("all runs finished")


def collect(args, passthrough):
    results = []
    for lp in args.lambda_phi_mstep:
        for lr in args.lambda_r:
            name = run_name(lp, lr)
            pattern = checkpoint_glob(name, args.seed, args.fold)
            if not glob.glob(pattern):
                print(f"{name}: no checkpoint, skipped")
                continue
            out = REPO / "sweep_logs" / f"{name}.eval.csv"
            cmd = [PY, str(REPO / "evaluate.py"), "--checkpoints", pattern,
                   "--split", "middle", "--fold", str(args.fold), "--gpu", str(args.gpu),
                   "--num-workers", str(args.num_workers), "--datasets-only",
                   "--out", str(out), *passthrough]
            if args.dry_run:
                print(" ".join(cmd)); continue
            print(f"scoring {name} ...", flush=True)
            subprocess.run(cmd, cwd=REPO, check=True,
                           stdout=open(REPO / "sweep_logs" / f"{name}.eval.log", "w"),
                           stderr=subprocess.STDOUT)
            with open(out) as handle:
                rows = [r for r in csv.DictReader(handle) if r["group"] == "ALL"]
            if not rows:
                print(f"{name}: no ALL row in {out}"); continue
            # evaluate.py scores every checkpoint matching the glob; with
            # snapshot_every 0 there is exactly one. Keep the last row anyway.
            r = rows[-1]
            results.append({"lambda_phi_mstep": lp, "lambda_R": lr, "n": int(r["n"]),
                            **{k: float(r[k]) for k in ("F", "CMLt", "AMLt", "dbF")}})
    if args.dry_run or not results:
        return
    results.sort(key=lambda r: -r["F"])
    print(f"\n{'lphi_mstep':>10} {'lambda_R':>10} {'n':>5} {'F':>8} {'CMLt':>8} {'AMLt':>8} {'dbF':>8}")
    for r in results:
        print(f"{r['lambda_phi_mstep']:>10g} {r['lambda_R']:>10g} {r['n']:>5} "
              f"{r['F']:>8.4f} {r['CMLt']:>8.4f} {r['AMLt']:>8.4f} {r['dbF']:>8.4f}")
    out = REPO / "sweep_results.csv"
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    print(f"\nwrote {out}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["launch", "collect"])
    # Grid. Defaults: lambda_phi spans the document's 3 up to the ~100x the
    # gradient ratio suggests; lambda_R spans off, a tenth of, and ten times
    # config.py's calibration.
    p.add_argument("--lambda-phi-mstep", type=float, nargs="+", default=[30.0, 100.0, 300.0],
                   help="M-step phase weights to try; the E-step weight stays config.LAMBDA_PHI")
    p.add_argument("--lambda-r", type=float, nargs="+",
                   default=[0.0, config.LAMBDA_R / 10, config.LAMBDA_R, config.LAMBDA_R * 10])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=40)
    # launch
    p.add_argument("--gpus", type=int, nargs="+", default=[0])
    p.add_argument("--per-gpu", type=int, default=1, help="concurrent runs per GPU")
    p.add_argument("--poll", type=float, default=30.0, help="seconds between scheduler polls")
    p.add_argument("--force", action="store_true", help="retrain runs whose checkpoint exists")
    # collect
    p.add_argument("--gpu", type=int, default=0, help="GPU for scoring")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    return p


if __name__ == "__main__":
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    args = build_parser().parse_args(argv)
    {"launch": launch, "collect": collect}[args.phase](args, passthrough)
