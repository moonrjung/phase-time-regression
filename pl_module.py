"""PyTorch Lightning wrapper for the phase-and-time regression model.

Puts HybridBeatTracker in the position PLBeatThis puts BeatThis, so this
formulation trains through the same Trainer, the same BeatDataModule, the same
augmentation, precision and checkpointing as the dense and subset arms -- which
is what makes the three comparable at all. The standalone loop in
train_and_infer.py stays as a correctness harness on synthetic data; this is the
path to real results.

Deliberately a SEPARATE LightningModule rather than a branch inside
PLBeatThis: that class is wired throughout for a 3-class head (its
_subset_targets emits DOWNBEAT/BEAT/CLASS_UNKNOWN labels, its _compute_loss
reads class_logits), all of which the continuous-phase formulation removes.
Adding a third head_type there would mean threading the old vocabulary through
code that no longer needs it, and would put AlignBeat's working subset runs at
risk for no gain here.

Metrics are NOT wired up: validation reports the loss only. Beat/downbeat
F-measure needs the decode of Section 5 run over the validation set and scored
with the standard tolerance, which is a separate piece of work; reporting
loss alone is honest, whereas reusing the dense head's postprocessor would
score a different model's output.
"""

import numpy as np
import torch
from pytorch_lightning import LightningModule

import config
from hybrid_beat_tracker import HybridBeatTracker
from train_and_infer import e_step, m_step_loss, phase_weight


# The most common meter in the corpus (METER_PRIOR puts 0.86 on it). Used only
# as a fallback where a fragment's own annotation cannot supply a bar length.
FALLBACK_METER = 4


def phases_from_downbeats(n_beats: int, downbeat_positions: np.ndarray,
                          fallback_meter: int = FALLBACK_METER) -> np.ndarray:
    """phi_i in [0, 1) for every beat of a fully-labeled fragment (Section 2.5).

    The document treats L as external per-track metadata, but also notes
    (Section 3.5) that on a fully-labeled track it "could in principle be read
    off the annotation directly, as the spacing between consecutive labeled
    downbeats". That is what this does, per bar rather than per track, so a
    fragment whose meter changes mid-window (Section 3.1) still gets correct
    phases without needing the change-point annotated.

    phi_i = (beats since this bar's downbeat) / (this bar's length), so a
    downbeat is exactly 0 and the k-th beat of an L-beat bar is exactly k/L.
    """
    phi = np.zeros(n_beats, dtype=np.float64)
    if len(downbeat_positions) == 0:
        return phi

    db = np.asarray(downbeat_positions, dtype=np.int64)
    for i in range(n_beats):
        # the bar this beat belongs to: the last downbeat at or before it, or,
        # for beats preceding the first downbeat, that first bar extended back
        k = int(np.searchsorted(db, i, side="right")) - 1
        k = max(k, 0)

        if k + 1 < len(db):
            bar_len = int(db[k + 1] - db[k])
        elif k > 0:
            bar_len = int(db[k] - db[k - 1])      # last bar: reuse the previous one
        else:
            bar_len = fallback_meter               # a single downbeat, no spacing to read
        if bar_len <= 0:
            bar_len = fallback_meter

        phi[i] = ((i - int(db[k])) % bar_len) / bar_len
    return phi


def fragment_targets(batch, fps: float, quantize: bool = False):
    """batch -> one (t_true, ind, phi_true) per fragment.

    Mirrors PLBeatThis._subset_targets exactly on everything the two share --
    the same annotation fields, the same (0, 1] window, the same uniqueness
    rule -- and differs only in what it emits: a continuous phi_i per event
    instead of a DOWNBEAT/BEAT/CLASS_UNKNOWN label.
    """
    num_frames = batch["truth_beat"].shape[-1]
    window_seconds = num_frames / fps
    device = batch["spect"].device

    targets = []
    for index in range(len(batch["spect"])):
        beats = np.frombuffer(batch["truth_orig_beat"][index])
        downbeats = np.frombuffer(batch["truth_orig_downbeat"][index])
        has_downbeats = bool(batch["downbeat_mask"][index])

        # eq. (1) maps onto the half-open axis (0, 1], so an event at exactly 0
        # is unreachable by construction -- same filter PLBeatThis applies.
        keep = (beats > 0) & (beats <= window_seconds)
        beats = np.unique(beats[keep])
        if quantize:
            beats = np.round(beats * fps) / fps

        if has_downbeats and len(beats):
            db_positions = np.flatnonzero(np.isin(beats, downbeats))
            phi = phases_from_downbeats(len(beats), db_positions)
            phi_true = torch.as_tensor(phi, dtype=torch.float32, device=device)
            ind = 0
            # For eq. (48): the annotated downbeats and the bar length between
            # each consecutive pair, read off the annotation exactly as
            # phases_from_downbeats does (Section 3.5: on a fully-labeled
            # track L "could in principle be read off the annotation
            # directly"). Taken from the annotation, NOT from phi_true == 0:
            # a beat one bar before the first annotated downbeat also gets
            # phi = 0 by the extended-first-bar rule, and must not count.
            downbeat_idx = torch.as_tensor(db_positions, dtype=torch.long, device=device)
            bar_lengths = (downbeat_idx[1:] - downbeat_idx[:-1]).to(torch.float32)
        else:
            # ind = 1 implies L unknown too (Section 2.5): the two are never
            # observed independently of one another. No downbeats, no bar
            # lengths: the E-step treats L as latent for this fragment.
            phi_true, ind = None, 1
            downbeat_idx, bar_lengths = None, None

        targets.append({
            "t_true": torch.as_tensor(beats / window_seconds,
                                      dtype=torch.float32, device=device),
            "ind": ind,
            "phi_true": phi_true,
            "downbeat_idx": downbeat_idx,     # ind=0 only
            "bar_lengths": bar_lengths,       # ind=0 only, (K-1,)
        })
    return targets


class PLPhaseTimeRegression(LightningModule):
    """HybridBeatTracker + the E-step/M-step loss, as a LightningModule."""

    def __init__(self,
                 lr: float = config.LR,
                 weight_decay: float = config.WEIGHT_DECAY,
                 warmup_steps: int = config.WARMUP_STEPS,
                 max_epochs: int = config.MAX_EPOCHS,
                 fps: int = config.FPS,
                 lambda_phi_estep: float | None = config.LAMBDA_PHI_ESTEP,
                 lambda_R: float = config.LAMBDA_R,
                 meter_candidates: list[int] = tuple(config.METER_CANDIDATES),
                 quantize_targets: bool = False,
                 **model_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = HybridBeatTracker(**model_kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.fps = fps
        # The phase weight is 1 / b_phi, learned per fragment by the model's
        # phase ScaleHead (appendix "Calibrating the loss weights"). This only
        # pins the E-step's copy of it when not None; see config.LAMBDA_PHI_ESTEP.
        self.lambda_phi_estep = lambda_phi_estep
        self.lambda_R = lambda_R
        self.quantize_targets = quantize_targets

        # ind = 1 fragments carry no meter (Section 2.5), and the eq. (48)
        # discussion forbids assigning them a default one. The meter is
        # therefore LATENT for those fragments: e_step resolves phi_0 under
        # every candidate meter and picks the best by cost - log pi_M(L)
        # (hard EM over L), while the M-step marginalises the periodicity
        # term over all of them. The candidate set and its prior are the
        # document's own pi_M (eq. 4), the same ones inference uses (eq. 67).
        self.meter_candidates = [int(L) for L in meter_candidates]
        self.register_buffer("pi_M", torch.tensor(
            [config.METER_PRIOR[L] for L in self.meter_candidates], dtype=torch.float32))
        # pi_L (eq. 25), the distribution of phi_0 given L, would have to be
        # estimated from the fully-labeled part of the corpus. Until that
        # estimate exists e_step uses a uniform pi_L for every L (its
        # pi_L=None default), flagged here rather than silently assumed.

    def forward(self, spect, epoch=None):
        return self.model(spect, epoch=epoch)

    def _compute_loss(self, batch):
        # epoch, not None: None means "ScaleHead is already trained", which
        # would skip eq. (51)'s warm-start entirely for the whole run.
        hat_phi, hat_t, b_e, b_phi = self.model(batch["spect"], epoch=self.current_epoch)
        targets = fragment_targets(batch, self.fps, self.quantize_targets)

        N = hat_t.shape[1]
        total, used, skipped, nonfinite = 0.0, 0, 0, 0

        # b_e is per-fragment already; hat_phi/hat_t come back in whatever dtype
        # the precision plugin chose. The DP reads scalars through .item(), so it
        # is dtype-agnostic, but the circular arithmetic around 0/1 is not:
        # fp16 has ~3 decimal digits there, which is coarse next to tau = 0.2.
        # Compute the loss in fp32 and let autograd cast the gradient back.
        hat_phi, hat_t, b_e, b_phi = hat_phi.float(), hat_t.float(), b_e.float(), b_phi.float()

        # Per-fragment, not batched: M and N differ per fragment and the DP is
        # inherently sequential over its own (M, N). Same structure
        # SubsetCriterion uses (criterion.py:298) -- the batch dimension is a
        # loop, and only the resulting losses are combined.
        for i, target in enumerate(targets):
            t_true = target["t_true"]
            M = t_true.shape[0]

            # An order-preserving injection needs N >= M, and a fragment with
            # fewer than 2 events carries no spacing information. Skipping is
            # what SubsetCriterion does too; a surviving fragment would mask
            # its own cause.
            if M < 2 or M > N:
                skipped += 1
                continue
            # Non-finite outputs (fp16 overflow in the trunk) would poison the
            # E-step's DP; drop the fragment and count it, so a run that starts
            # producing NaN shows it in the logged 'nonfinite' rather than
            # crashing or silently training on nothing.
            if not (torch.isfinite(hat_t[i]).all() and torch.isfinite(hat_phi[i]).all()
                    and torch.isfinite(b_e[i]) and torch.isfinite(b_phi[i])):
                nonfinite += 1
                continue

            lam = phase_weight(b_phi[i], self.lambda_phi_estep)   # 1 / b_phi unless pinned
            if target["ind"] == 0:
                hat_sigma, phi_i, unmatched, hyps = e_step(
                    t_true, hat_t[i], hat_phi[i], b_e[i], lam,
                    ind=0, phi_true=target["phi_true"])
            else:
                hat_sigma, phi_i, unmatched, hyps = e_step(
                    t_true, hat_t[i], hat_phi[i], b_e[i], lam,
                    ind=1, meter_candidates=self.meter_candidates, pi_M=self.pi_M)

            # eq. (48)'s per-fragment gate: annotated downbeats + bar lengths
            # for ind=0 (None for ind=1), meter hypotheses for ind=1 (None
            # for ind=0). m_step_loss applies whichever is present.
            loss = m_step_loss(
                hat_sigma, phi_i, t_true, hat_t[i], hat_phi[i], b_e[i], b_phi[i],
                unmatched_agree=unmatched, lambda_R=self.lambda_R,
                downbeat_idx=target["downbeat_idx"], bar_lengths=target["bar_lengths"],
                meter_hyps=hyps)

            total = total + loss
            used += 1

        if used == 0:
            return None, {"used": 0, "skipped": skipped, "nonfinite": nonfinite}
        return total / used, {"used": used, "skipped": skipped, "nonfinite": nonfinite}

    def training_step(self, batch, batch_idx):
        loss, stats = self._compute_loss(batch)
        if loss is None:
            # Returning None makes Lightning skip the step rather than
            # backpropagate a fabricated zero.
            return None
        batch_size = len(batch["spect"])
        self.log("train_loss", loss, batch_size=batch_size, prog_bar=True)
        self.log("train_fragments_used", float(stats["used"]), batch_size=batch_size)
        self.log("train_fragments_skipped", float(stats["skipped"]), batch_size=batch_size)
        self.log("train_fragments_nonfinite", float(stats["nonfinite"]), batch_size=batch_size,
                 prog_bar=stats["nonfinite"] > 0)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, stats = self._compute_loss(batch)
        if loss is None:
            return None
        batch_size = len(batch["spect"])
        self.log("val_loss", loss, batch_size=batch_size, prog_bar=True)
        self.log("val_fragments_skipped", float(stats["skipped"]), batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        scheduler = CosineWarmupScheduler(
            optimizer, self.warmup_steps, self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing over `max_iters` steps after `warmup` linear steps.

    Copied from beat_this/model/pl_module.py so the three arms share one
    schedule rather than a reimplementation that can drift. Imported rather
    than copied would pull in that module's alignbeat dependencies; the
    raise_last / raise_to branch is dropped, being unused at its defaults.
    """

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = int(max_iters)
        super().__init__(optimizer)

    def get_lr(self):
        factor = self.get_lr_factor(step=self.last_epoch)
        return [base_lr * factor for base_lr in self.base_lrs]

    def get_lr_factor(self, step):
        progress = min(step / max(self.max_num_iters, 1), 1.0)
        factor = 0.5 * (1 + np.cos(np.pi * progress))
        if self.warmup > 0 and step <= self.warmup:
            factor *= step / self.warmup
        return factor
