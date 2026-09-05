"""Score checkpoints with the standard beat-tracking metrics, per dataset.

Training only logs a loss, which says nothing comparable to the dense or subset
arms' F-measure. This decodes each held-out piece through Section 5 and
Algorithm 10 and scores it with mir_eval.

Built to line up column-for-column with AlignBeat's own scorers
(launch_scripts/score_fold0_dense.py and score_fold0_subset.py), from which the
protocol here is taken: the same fold-0 loader, the same per-piece skip rule
(fewer than 3 annotated events), the same downbeat rule (scored only where
annotated), the same bpm and corpus grouping, and the same F / CMLt / AMLt
columns. A CSV out of this can be concatenated with cache/bygroup_van.csv.

    # head-to-head with the baselines, which score MIDDLE EXCERPTS
    python evaluate.py --checkpoints "checkpoints/phase*.ckpt" --split middle \
        --out ours_bygroup.csv

    # whole pieces: a harder, more realistic measurement, NOT comparable to the above
    python evaluate.py --checkpoints "checkpoints/phase*.ckpt" --split whole

The split matters and is not a detail: the baselines in cache/bygroup_van.csv
were produced with val_dataloader(), which serves only the middle 30 s of each
piece. Whole-piece decoding additionally exercises the fragment stitching and
includes intros and outros. Comparing one against the other measures the
protocol as much as the model.
"""

import argparse
import glob
import inspect
from collections import defaultdict
from pathlib import Path

import mir_eval
import numpy as np
import torch

import config
from pl_module import PLPhaseTimeRegression
from stitching import stitch_piece
from train_and_infer import decode, infer_meter, meter_consistency_correction

MIN_EVENTS = 3          # score_fold0_dense.py's own threshold
TRIM = 5.0              # eval_trim_beats, PLBeatThis's default


def load_model(ckpt_path, device):
    """Rebuild from the checkpoint's hyper_parameters.

    Adapted from launch_scripts/score_fold0_subset.py, including its discipline:
    retired hyperparameters are dropped with a note, but MISSING WEIGHTS are
    fatal. strict=False would otherwise leave part of the model randomly
    initialised and go on to report confident nonsense.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = dict(ckpt.get("hyper_parameters", {}))
    accepted = set(inspect.signature(PLPhaseTimeRegression.__init__).parameters)
    dropped = sorted(k for k in hp if k not in accepted and k != "model_kwargs")
    if dropped:
        print(f"    ignoring retired hyper_parameters: {', '.join(dropped)}")
        hp = {k: v for k, v in hp.items() if k in accepted}

    model = PLPhaseTimeRegression(**hp)
    # train.py --compile wraps parts in torch.compile, which inserts
    # "_orig_mod." into every key of that part; strip it, as beat_this's own
    # loader does (beat_tracker.py:196).
    state_dict = {k.replace("._orig_mod.", ".").removeprefix("_orig_mod."): v
                  for k, v in ckpt["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"    !! {len(missing)} weights absent from this checkpoint: "
              f"{missing[:4]} -- the architecture has changed, so scores would be "
              f"meaningless. Skipping.")
        return None
    if unexpected:
        print(f"    note: {len(unexpected)} unused keys, e.g. {unexpected[:2]}")
    return model.eval().to(device)


def metrics(truth, preds):
    """F, CMLt, AMLt -- the columns the baseline CSVs carry.

    Same calls and same trim as PLBeatThis's own Metrics at step="test".
    """
    truth = mir_eval.beat.trim_beats(np.asarray(truth, dtype=np.float64),
                                     min_beat_time=TRIM)
    preds = mir_eval.beat.trim_beats(np.asarray(preds, dtype=np.float64),
                                     min_beat_time=TRIM)
    if len(truth) == 0:
        return None
    _CMLc, CMLt, _AMLc, AMLt = mir_eval.beat.continuity(truth, preds)
    return {"F": mir_eval.beat.f_measure(truth, preds), "CMLt": CMLt, "AMLt": AMLt}


def decode_excerpt(model, spect, args, device):
    """Section 5 on one fixed-length excerpt. Returns (beats, downbeats) in seconds."""
    with torch.no_grad():
        hat_phi, hat_t, _ = model.model(spect.unsqueeze(0).to(device).float())
    hat_phi, hat_t = hat_phi[0].float(), hat_t[0].float()

    hat_L = infer_meter(hat_phi, args.meter_candidates,
                        torch.tensor(args.pi_M), model.lambda_phi)
    p_hat, d, t_hat, B = decode(hat_phi, hat_t, hat_L, args.tau)
    B = meter_consistency_correction(B, hat_L, p_hat, d, t_hat, args.tau, args.tau_prime)

    seconds = spect.shape[0] / args.fps
    beats = np.array([t * seconds for _, t in B], dtype=np.float64)
    downbeats = np.array([t * seconds for p, t in B if p == 0], dtype=np.float64)
    return np.unique(beats), np.unique(downbeats)


def score_checkpoint(model, loader, args, device):
    """One row per scorable piece, mirroring score_fold0_dense.py's own loop."""
    rows = []
    for index, batch in enumerate(loader):
        if args.limit and index >= args.limit:
            break
        for i in range(len(batch["spect"])):
            truth_b = np.frombuffer(batch["truth_orig_beat"][i])
            if len(truth_b) < MIN_EVENTS:
                continue

            if args.split == "whole":
                beats, downbeats = stitch_piece(
                    batch["spect"][i], model.model, args.train_length, args.border,
                    args.fps, args.meter_candidates, torch.tensor(args.pi_M),
                    args.tau, args.tau_prime, model.lambda_phi, device=device)
            else:
                beats, downbeats = decode_excerpt(model, batch["spect"][i], args, device)

            met_b = metrics(truth_b, beats)
            if met_b is None:
                continue
            path = str(batch["spect_path"][i])
            row = dict(path=path, corpus=path.split("/", 1)[0],
                       bpm=float(60.0 / np.median(np.diff(truth_b))), **met_b)

            truth_db = np.frombuffer(batch["truth_orig_downbeat"][i])
            if bool(batch["downbeat_mask"][i]) and len(truth_db) >= MIN_EVENTS:
                met_d = metrics(truth_db, downbeats)
                if met_d:
                    row.update(dbF=met_d["F"], dbCMLt=met_d["CMLt"],
                               dbAMLt=met_d["AMLt"])
            rows.append(row)
    return rows


def summarize(rows):
    """The grouping score_fold0_dense.py uses, so the tables can sit side by side."""
    groups = {"ALL": lambda r: True,
              "SMC": lambda r: r["corpus"] == "smc",
              "<70 bpm": lambda r: r["bpm"] < 70,
              "SMC <70": lambda r: r["corpus"] == "smc" and r["bpm"] < 70,
              ">=70 bpm": lambda r: r["bpm"] >= 70}
    for lo, hi in ((0, 70), (70, 100), (100, 130), (130, 160), (160, 1e9)):
        name = f"bpm {lo}-{hi:.0f}" if hi < 1e9 else f"bpm {lo}+"
        groups[name] = (lambda l, h: lambda r: l <= r["bpm"] < h)(lo, hi)
    for corpus in sorted({r["corpus"] for r in rows}):
        groups[f"ds:{corpus}"] = (lambda c: lambda r: r["corpus"] == c)(corpus)

    out = {}
    for name, fn in groups.items():
        sel = [r for r in rows if fn(r)]
        if not sel:
            continue
        db = [r for r in sel if "dbF" in r]
        out[name] = {
            "n": len(sel),
            "F": float(np.mean([r["F"] for r in sel])),
            "CMLt": float(np.mean([r["CMLt"] for r in sel])),
            "AMLt": float(np.mean([r["AMLt"] for r in sel])),
            "n_db": len(db),
            "dbF": float(np.mean([r["dbF"] for r in db])) if db else float("nan"),
            "dbCMLt": float(np.mean([r["dbCMLt"] for r in db])) if db else float("nan"),
            "dbAMLt": float(np.mean([r["dbAMLt"] for r in db])) if db else float("nan"),
        }
    return out


def load_baseline(path):
    """cache/bygroup_van.csv, or any CSV in that shape, keyed by group."""
    import csv
    with open(path) as handle:
        return {r["group"]: r for r in csv.DictReader(handle)}


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.pi_M = [config.METER_PRIOR[l] for l in args.meter_candidates]

    from beat_this.dataset import BeatDataModule
    data_dir = (Path(args.data_dir) if args.data_dir
                else Path(__file__).resolve().parent / "data")
    dm = BeatDataModule(
        data_dir, batch_size=1, train_length=args.train_length, spect_fps=args.fps,
        num_workers=args.num_workers, test_dataset="gtzan",
        length_based_oversampling_factor=0.65, augmentations={},
        hung_data=False, no_val=False, fold=args.fold,
        predict_datasplit="val",
    )
    if args.split == "whole":
        dm.setup(stage="predict")
        loader = dm.predict_dataloader()
    else:
        # Exactly what score_fold0_dense.py and score_fold0_subset.py use:
        # the middle excerpt of each piece, not the whole thing.
        dm.setup(stage="fit")
        loader = dm.val_dataloader()
    print(f"{len(loader)} items, fold {args.fold}, split={args.split}")

    baseline = load_baseline(args.baseline) if args.baseline else None

    paths = sorted(glob.glob(args.checkpoints))
    if not paths:
        raise SystemExit(f"no checkpoints matched {args.checkpoints!r}")

    csv_rows = []
    for path in paths:
        name = Path(path).name
        print(f"\n{name}")
        model = load_model(path, device)
        if model is None:
            continue
        rows = score_checkpoint(model, loader, args, device)
        if not rows:
            print("    no scorable pieces")
            continue
        summary = summarize(rows)

        header = f"    {'group':<18}{'n':>5}{'F':>9}{'CMLt':>9}{'AMLt':>9}{'dbF':>9}"
        if baseline:
            header += f"{'F(base)':>10}{'delta':>9}"
        print(header)
        for group in sorted(summary, key=lambda g: (not g.startswith("ds:"), g)):
            if args.datasets_only and not group.startswith("ds:") and group != "ALL":
                continue
            s = summary[group]
            line = (f"    {group:<18}{s['n']:>5}{s['F']:>9.4f}{s['CMLt']:>9.4f}"
                    f"{s['AMLt']:>9.4f}{s['dbF']:>9.4f}")
            if baseline and group in baseline:
                base = float(baseline[group]["F"])
                line += f"{base:>10.4f}{s['F'] - base:>+9.4f}"
            print(line)
            csv_rows.append({"arm": name, "group": group, **s})

    if args.out and csv_rows:
        import csv
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", type=str, required=True,
                        help="glob, e.g. 'checkpoints/phase*.ckpt'")
    parser.add_argument("--split", choices=["middle", "whole"], default="middle",
                        help="'middle' reproduces the baselines' protocol (the middle "
                             "excerpt of each piece) and is what to use for a "
                             "head-to-head; 'whole' decodes entire pieces")
    parser.add_argument("--baseline", type=str, default=None,
                        help="a bygroup CSV to print alongside, e.g. "
                             "../AlignBeat/cache/bygroup_van.csv")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N items (smoke tests)")
    parser.add_argument("--datasets-only", action="store_true",
                        help="print ALL and the ds:* rows, skipping the bpm bands")

    parser.add_argument("--train_length", type=int, default=config.TRAIN_LENGTH)
    parser.add_argument("--fps", type=int, default=config.FPS)
    parser.add_argument("--border", type=int, default=6,
                        help="frames discarded either side of a fragment seam "
                             "(--split whole only)")
    parser.add_argument("--tau", type=float, default=config.TAU)
    parser.add_argument("--tau_prime", type=float, default=config.TAU_PRIME)
    parser.add_argument("--meter-candidates", type=int, nargs="+",
                        default=config.METER_CANDIDATES)

    main(parser.parse_args())
