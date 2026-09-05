"""Launch script for the phase-and-time regression arm.

Adapted from launch_scripts/train.py rather than written fresh, so this arm
sees the same data, the same augmentation, the same precision, the same
checkpointing and the same schedule as the dense and subset arms -- which is
what makes the three comparable. Everything below that differs from that
script is marked DIFFERS.

Run it the same way, from the repository root:

    PYTHONPATH=. python phase_time_regression/train.py --name phase --gpu 0 --fold 0

DIFFERS from launch_scripts/train.py, in full:
  * builds PLPhaseTimeRegression, not PLBeatThis -- and therefore imports
    neither beat_this.model.pl_module nor the alignbeat package, both of which
    carry the earlier draft's 3-class head;
  * --head_type, --loss, --omega_db, --dbn, --sum_head and the other
    class-head flags are gone: this formulation has no classifier;
  * no positive weights: get_train_positive_weights() exists for the dense
    head's BCE, and nothing here uses it;
  * no trainer.test(): scoring needs Section 5's decode, which is not wired up
    (see pl_module.py's own header). Training and validation loss only.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# Checkpoints carry numpy scalars in hyper_parameters and torch >= 2.6 loads
# with weights_only=True, which refuses them. Same patch launch_scripts/train.py
# applies, for the same reason and with the same caveat: safe only because these
# are checkpoints this repo wrote. Do NOT copy into anything loading third-party
# checkpoints.
_torch_load = torch.load
def _load_trusted(*a, **k):
    k["weights_only"] = False
    return _torch_load(*a, **k)
torch.load = _load_trusted

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from beat_this.dataset import BeatDataModule

# This package imports its own modules flat (import config, from heads import ...),
# so its directory has to be importable however the script was invoked.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from pl_module import PLPhaseTimeRegression


def main(args):
    seed_everything(args.seed, workers=True)
    print("Starting a new run with the following parameters:")
    print(args)

    params_str = (f"{'noval ' if not args.val else ''}"
                  f"{'fold' + str(args.fold) + ' ' if args.fold is not None else ''}"
                  f"phase-h{args.transformer_dim}"
                  f"-aug{args.tempo_augmentation}{args.pitch_augmentation}{args.mask_augmentation}")

    if args.logger == "wandb":
        wandb_args = (dict(id=args.resume_id, resume="must")
                      if args.resume_checkpoint and args.resume_id else {})
        logger = WandbLogger(project="beat_this",
                             name=f"{args.name} {params_str}".strip(), **wandb_args)
    else:
        logger = None

    if args.force_flash_attention:
        print("Forcing the use of the flash attention.")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)

    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / "data"
    checkpoint_dir = repo_root / "checkpoints"

    augmentations = {}
    if args.tempo_augmentation:
        augmentations["tempo"] = {"min": -20, "max": 20, "stride": 4}
    if args.pitch_augmentation:
        augmentations["pitch"] = {"min": -5, "max": 6}
    if args.mask_augmentation:
        augmentations["mask"] = {"kind": "permute", "min_count": 1, "max_count": 6,
                                 "min_len": 0.1, "max_len": 2,
                                 "min_parts": 5, "max_parts": 9}

    datamodule = BeatDataModule(
        data_dir,
        batch_size=args.batch_size,
        train_length=args.train_length,
        spect_fps=args.fps,
        num_workers=args.num_workers,
        test_dataset="gtzan",
        length_based_oversampling_factor=args.length_based_oversampling_factor,
        augmentations=augmentations,
        hung_data=args.hung_data,
        no_val=not args.val,
        fold=args.fold,
    )
    datamodule.setup(stage="fit")
    # DIFFERS: no get_train_positive_weights() -- that exists for the dense
    # head's BCE, and this loss has no class term to weight.

    pl_model = PLPhaseTimeRegression(
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        fps=args.fps,
        lambda_phi=args.lambda_phi,
        lambda_R=args.lambda_r,
        beat_only_meter=args.beat_only_meter,
        quantize_targets=args.quantize_targets,
        # forwarded to HybridBeatTracker
        spect_dim=args.spect_dim,
        transformer_dim=args.transformer_dim,
        n_layers=args.n_layers,
        head_dim=args.head_dim,
        stem_dim=args.stem_dim,
        dropout={"frontend": args.frontend_dropout,
                 "transformer": args.transformer_dropout},
        fragment_frames=args.train_length,
        b_0=args.b_0,
        warmup_epochs=args.scale_warmup_epochs,
    )

    if args.compile:
        for part in args.compile:
            if hasattr(pl_model.model, part):
                setattr(pl_model.model, part,
                        torch.compile(getattr(pl_model.model, part)))
                print("Will compile model", part)
            else:
                raise ValueError("The model is missing the part", part, "to compile")

    callbacks = [LearningRateMonitor(logging_interval="step")]
    if args.snapshot_every:
        callbacks.append(ModelCheckpoint(
            every_n_epochs=args.snapshot_every, save_top_k=-1,
            save_on_train_epoch_end=True, dirpath=str(checkpoint_dir),
            filename=f"{args.name} S{args.seed} {params_str}".strip() + "-ep{epoch:03d}"))
    else:
        callbacks.append(ModelCheckpoint(
            every_n_epochs=1, dirpath=str(checkpoint_dir),
            filename=f"{args.name} S{args.seed} {params_str}".strip()))

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=[args.gpu],
        num_sanity_val_steps=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        precision="16-mixed",
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.val_frequency,
        limit_train_batches=args.limit_train_batches or 1.0,
        limit_val_batches=args.limit_val_batches or 1.0,
    )

    trainer.fit(pl_model, datamodule, ckpt_path=args.resume_checkpoint)
    # DIFFERS: no trainer.test(). Section 5's decode is not wired up, and
    # reporting the dense head's metrics for this model would be wrong.


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the phase-and-time regression model. Flags mirror "
                    "launch_scripts/train.py; defaults come from config.py, which "
                    "carries AlignBeat's own values.")
    # run identity / hardware
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--logger", type=str, choices=["wandb", "none"], default="none")
    parser.add_argument("--force-flash-attention", default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--compile", type=str, nargs="*",
                        default=["frontend", "transformer_blocks"],
                        help="model parts to torch.compile; pass with no values to disable")

    # data
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--val", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--hung-data", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--tempo-augmentation", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--pitch-augmentation", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--mask-augmentation", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--length-based-oversampling-factor", type=float, default=0.65)
    parser.add_argument("--train_length", "--train-length", dest="train_length",
                        type=int, default=config.TRAIN_LENGTH)
    parser.add_argument("--fps", type=int, default=config.FPS)

    # backbone
    parser.add_argument("--spect-dim", type=int, default=config.SPECT_DIM)
    parser.add_argument("--transformer-dim", type=int, default=config.TRANSFORMER_DIM)
    parser.add_argument("--n-layers", type=int, default=config.N_LAYERS)
    parser.add_argument("--head-dim", type=int, default=config.HEAD_DIM)
    parser.add_argument("--stem-dim", type=int, default=config.STEM_DIM)
    parser.add_argument("--frontend-dropout", type=float, default=config.DROPOUT["frontend"])
    parser.add_argument("--transformer-dropout", type=float, default=config.DROPOUT["transformer"])

    # optimisation
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--warmup-steps", type=int, default=config.WARMUP_STEPS)
    parser.add_argument("--max-epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--accumulate-grad-batches", type=int,
                        default=config.ACCUMULATE_GRAD_BATCHES)
    parser.add_argument("--val-frequency", type=int, default=5)
    parser.add_argument("--snapshot_every", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=0,
                        help="cap training batches per epoch (0 = no cap). Smoke tests only.")
    parser.add_argument("--limit-val-batches", type=int, default=0,
                        help="cap validation batches per epoch (0 = no cap). Smoke tests only.")

    # this formulation's own knobs (no launch_scripts/train.py counterpart)
    parser.add_argument("--lambda_phi", type=float, default=config.LAMBDA_PHI,
                        help="eq. (19): weight on the circular phase term. No "
                             "AlignBeat counterpart -- omega_db weighted a discrete "
                             "class, not a distance. Wants its own sweep.")
    parser.add_argument("--lambda_r", type=float, default=config.LAMBDA_R,
                        help="eq. (48): periodicity regularizer on downbeat spacing")
    parser.add_argument("--b_0", type=float, default=config.B_0,
                        help="eq. (51): fixed timing scale during warm-start, in "
                             "normalized [0,1] fragment time")
    parser.add_argument("--scale-warmup-epochs", type=int, default=config.WARMUP_EPOCHS,
                        help="eq. (51)'s E_0: epochs before ScaleHead takes over from b_0")
    parser.add_argument("--beat-only-meter", type=int, default=4,
                        help="meter assumed for beat-only (ind=1) fragments, pending "
                             "MixedMeterTarget (Algorithm 1)")
    parser.add_argument("--quantize-targets", default=False, action="store_true",
                        help="round ground-truth times to the frame grid, as the dense "
                             "head is necessarily trained on")

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
