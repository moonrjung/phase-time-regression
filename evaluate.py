"""Score checkpoints on whole pieces with the standard beat-tracking metrics.

This is the piece training does not do: validation logs a loss, which says
nothing comparable to the dense or subset arms' F-measure. Here each held-out
piece is decoded end to end (Section 5 + Algorithm 10, per fragment, stitched)
and scored against its own annotation with mir_eval, at the same 70 ms
tolerance and the same 5 s trim the Beat This! pipeline uses -- so a number out
of this is directly comparable to one out of score_fold0_subset.py or
score_fold0_dense.py.

    python evaluate.py --checkpoints "checkpoints/phase*.ckpt" --fold 0 \
        --out scores.csv

Whole pieces, batch size 1, exactly as the paper results are computed --
BeatDataModule's val_dataloader deliberately serves only the middle excerpt of
each piece for speed, so it is NOT used here.
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


def load_model(ckpt_path, device):
    """Rebuild from the checkpoint's own hyper_parameters.

    Retired hyperparameters are dropped with a note; MISSING WEIGHTS are fatal.
    strict=False would otherwise leave part of the model randomly initialised
    and go on to report confident nonsense, which is the one failure mode a
    scoring script must not have.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = dict(ckpt.get("hyper_parameters", {}))
    accepted = set(inspect.signature(PLPhaseTimeRegression.__init__).parameters)
    dropped = sorted(k for k in hp if k not in accepted and k != "model_kwargs")
    if dropped:
        print(f"    ignoring retired hyper_parameters: {', '.join(dropped)}")
        hp = {k: v for k, v in hp.items() if k in accepted}

    model = PLPhaseTimeRegression(**hp)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        print(f"    !! {len(missing)} weights absent from this checkpoint: "
              f"{missing[:4]} -- the architecture has changed, so scores would be "
              f"meaningless. Skipping.")
        return None
    if unexpected:
        print(f"    note: {len(unexpected)} unused keys, e.g. {unexpected[:2]}")
    return model.eval().to(device)


def score(truth, preds, trim=5.0):
    """mir_eval F-measure and Cemgil, with the same trim the pipeline uses."""
    truth = mir_eval.beat.trim_beats(np.asarray(truth), min_beat_time=trim)
    preds = mir_eval.beat.trim_beats(np.asarray(preds), min_beat_time=trim)
    if len(truth) == 0:
        return None
    return {"F-measure": mir_eval.beat.f_measure(truth, preds),
            "Cemgil": mir_eval.beat.cemgil(truth, preds)}


def evaluate(model, loader, args, device):
    per_dataset = defaultdict(lambda: defaultdict(list))
    overall = defaultdict(list)

    for index, batch in enumerate(loader):
        if args.limit and index >= args.limit:
            break
        mel = batch["spect"][0]
        dataset = batch["dataset"][0]

        beats, downbeats = stitch_piece(
            mel, model.model, args.train_length, args.border, args.fps,
            args.meter_candidates, torch.tensor(args.pi_M),
            args.tau, args.tau_prime, model.lambda_phi, device=device)

        truth_beat = np.frombuffer(batch["truth_orig_beat"][0])
        truth_downbeat = np.frombuffer(batch["truth_orig_downbeat"][0])
        has_downbeats = bool(batch["downbeat_mask"][0])

        m = score(truth_beat, beats)
        if m:
            for k, v in m.items():
                overall[f"beat_{k}"].append(v)
                per_dataset[dataset][f"beat_{k}"].append(v)

        # A beat-only piece has no downbeat annotation to score against;
        # counting it as zero would understate the model rather than measure it.
        if has_downbeats and len(truth_downbeat):
            m = score(truth_downbeat, downbeats)
            if m:
                for k, v in m.items():
                    overall[f"downbeat_{k}"].append(v)
                    per_dataset[dataset][f"downbeat_{k}"].append(v)

        if args.verbose and index < 10:
            print(f"    [{index}] {dataset:12s} beats {len(beats):4d}/"
                  f"{len(truth_beat):4d}  downbeats {len(downbeats):3d}/"
                  f"{len(truth_downbeat):3d}")

    return overall, per_dataset


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.pi_M = [config.METER_PRIOR[l] for l in args.meter_candidates]

    from beat_this.dataset import BeatDataModule
    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent / "data"
    dm = BeatDataModule(
        data_dir, batch_size=1, train_length=args.train_length, spect_fps=args.fps,
        num_workers=args.num_workers, test_dataset="gtzan",
        length_based_oversampling_factor=0.65, augmentations={},
        hung_data=False, no_val=False, fold=args.fold,
        predict_datasplit="val",
    )
    dm.setup(stage="predict")
    loader = dm.predict_dataloader()
    print(f"{len(loader)} whole pieces from fold {args.fold}")

    paths = sorted(glob.glob(args.checkpoints))
    if not paths:
        raise SystemExit(f"no checkpoints matched {args.checkpoints!r}")

    rows = []
    for path in paths:
        print(f"\n{Path(path).name}")
        model = load_model(path, device)
        if model is None:
            continue
        overall, per_dataset = evaluate(model, loader, args, device)
        if not overall:
            print("    no scorable pieces")
            continue
        summary = {k: float(np.mean(v)) for k, v in overall.items()}
        counts = {k: len(v) for k, v in overall.items()}
        print("    " + "  ".join(f"{k}={v:.4f} (n={counts[k]})"
                                 for k, v in sorted(summary.items())))
        rows.append({"checkpoint": Path(path).name, **summary})

        if args.per_dataset:
            for name in sorted(per_dataset):
                d = per_dataset[name]
                print(f"      {name:12s} " + "  ".join(
                    f"{k}={np.mean(v):.4f}" for k, v in sorted(d.items())
                    if k.endswith("F-measure")))

    if args.out and rows:
        import csv
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", type=str, required=True,
                        help="glob, e.g. 'checkpoints/phase*.ckpt'")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=None, help="write a CSV here")
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N pieces (smoke tests)")
    parser.add_argument("--per-dataset", action="store_true",
                        help="break the F-measures down by source dataset")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--train_length", type=int, default=config.TRAIN_LENGTH,
                        help="fragment length in frames; must match training")
    parser.add_argument("--fps", type=int, default=config.FPS)
    parser.add_argument("--border", type=int, default=6,
                        help="frames discarded either side of a fragment seam")
    parser.add_argument("--tau", type=float, default=config.TAU)
    parser.add_argument("--tau_prime", type=float, default=config.TAU_PRIME)
    parser.add_argument("--meter-candidates", type=int, nargs="+",
                        default=config.METER_CANDIDATES)

    main(parser.parse_args())
