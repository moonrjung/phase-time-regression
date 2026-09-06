"""
Hybrid model: BeatThis's own, real, working backbone (frontend + RoFormer
transformer blocks, imported directly from the actual CPJKU/beat_this
source, not reimplemented) -- with its task_heads (frame-wise beat/downbeat
BCE classification) REMOVED and replaced by our own downsampling +
candidate-based regression heads (PhaseHead, RegHead, ScaleHead).

This is the concrete result of reconsidering the document's design against
BeatThis's actual, validated architecture: keep what already works
(the backbone) and only replace the final task-specific stage, rather than
building our own encoder from scratch or trying to force our own heads
into BeatThis's frame-wise output format.

BeatThis's encoder is vendored into backbone.py (with roformer.py), copied
from CPJKU/beat_this under its MIT licence, so no external `beat_this`
package needs to be importable.
"""

import torch
import torch.nn as nn

# Vendored locally (backbone.py, roformer.py) rather than imported from the
# `beat_this` package, so this folder is self-contained: it depends on neither
# AlignBeat's forked beat_this nor the earlier-draft `alignbeat` package that
# fork imports transitively. The encoder weights themselves are upstream's,
# unmodified -- see backbone.py's own header.
from backbone import BeatThisBackbone

import config
from downsample import Downsample
from heads import (
    SharedProjection,
    CandidateSelfAttention,
    PhaseHead,
    RegHead,
    ScaleHead,
    mean_pool_candidates,
    monotonic_time_reparam,
    time_reparam,
    scale_with_warmup,
)


class HybridBeatTracker(nn.Module):
    """BeatThis's own frontend + transformer_blocks (their real backbone,
    unmodified, vendored in backbone.py) -> AlignBeat's Downsample ->
    SharedProjection ->
    (CandidateSelfAttention -> PhaseHead) and (RegHead, ScaleHead).

    BeatThis's own task_heads is never instantiated at all -- we build a
    BeatThis instance only to obtain .frontend and .transformer_blocks,
    then discard the rest -- BeatThisBackbone simply never builds them."""

    def __init__(self, spect_dim: int = config.SPECT_DIM,
                 transformer_dim: int = config.TRANSFORMER_DIM,
                 n_layers: int = config.N_LAYERS,
                 head_dim: int = config.HEAD_DIM,
                 stem_dim: int = config.STEM_DIM,
                 N_min: int = config.N_MIN,
                 reduced_dim: int = config.REDUCED_DIM,
                 phase_hidden: int = config.HEAD_HIDDEN,
                 reg_hidden: int = config.HEAD_HIDDEN,
                 scale_hidden: int = config.HEAD_HIDDEN,
                 b_0: float = config.B_0,
                 warmup_epochs: int = config.WARMUP_EPOCHS,
                 ff_mult: int = config.FF_MULT,
                 dropout: dict = None,
                 fragment_frames: int = config.TRAIN_LENGTH,
                 time_reparam: str = config.TIME_REPARAM):
        super().__init__()
        self.time_reparam_kind = time_reparam

        # Instantiate their real model, but keep only the backbone. ff_mult and
        # dropout are BeatThis's own arguments, passed through at AlignBeat's
        # values rather than left at BeatThis's defaults. AlignBeat's --n-heads 16
        # is not a BeatThis argument: the head count is implied by
        # transformer_dim / head_dim = 512 / 32 = 16, which these values already give.
        backbone_source = BeatThisBackbone(
            spect_dim=spect_dim, transformer_dim=transformer_dim,
            n_layers=n_layers, head_dim=head_dim, stem_dim=stem_dim,
            ff_mult=ff_mult,
            dropout=dict(config.DROPOUT) if dropout is None else dropout,
        )
        self.frontend = backbone_source.frontend
        self.transformer_blocks = backbone_source.transformer_blocks
        # No task_heads to delete: BeatThisBackbone never builds any.

        # Our own candidate-based stage, replacing their frame-wise task_heads.
        # AlignBeat's own Downsample (vendored), not heads.EncoderToCandidates:
        # it builds its Conv1d stack in __init__, so the weights are present
        # before an optimizer captures model.parameters().
        self.downsample = Downsample(
            transformer_dim, num_candidates=config.NUM_CANDIDATES,
            mode=config.DOWNSAMPLE_MODE, fragment_frames=fragment_frames,
            stages=config.DOWNSAMPLE_STAGES)
        self.shared_proj = SharedProjection(transformer_dim, reduced_dim)
        self.candidate_attn = CandidateSelfAttention(
            reduced_dim, n_heads=config.ATTENTION_HEADS,
            n_layers=config.ATTENTION_LAYERS, ff_mult=config.ATTENTION_FF_MULT,
            dropout=config.ATTENTION_DROPOUT,
            final_norm=config.ATTENTION_FINAL_NORM)
        self.phase_head = PhaseHead(reduced_dim, phase_hidden)
        self.reg_head = RegHead(reduced_dim, reg_hidden)
        self.scale_head = ScaleHead(reduced_dim, scale_hidden, b_min=config.B_MIN)
        self.b_0 = b_0
        self.warmup_epochs = warmup_epochs

    def forward(self, x: torch.Tensor, epoch: int | None = None):
        # x: (batch, T, spect_dim) -- raw log-mel spectrogram, BeatThis's own input format
        h = self.frontend(x)                       # (batch, T, transformer_dim)
        h = self.transformer_blocks(h)              # (batch, T, transformer_dim), same shape

        z = self.downsample(h)                      # (batch, N, transformer_dim)
        z = self.shared_proj(z)                      # (batch, N, reduced_dim)
        tilde_z = self.candidate_attn(z)             # phase branch only
        hat_phi = self.phase_head(tilde_z)           # (batch, N), in [0, 1)
        r = self.reg_head(z)                         # (batch, N), raw scores
        hat_t = time_reparam(r, self.time_reparam_kind)   # (batch, N), strictly increasing

        z_bar = mean_pool_candidates(z)               # (batch, reduced_dim)
        if epoch is None:
            b_e = self.scale_head(z_bar)
        else:
            b_e = scale_with_warmup(z_bar, self.scale_head, epoch, self.warmup_epochs, self.b_0)

        return hat_phi, hat_t, b_e


if __name__ == "__main__":
    torch.manual_seed(0)
    # AlignBeat's real configuration throughout (config.py) -- T = 1500 frames
    # at 50 fps, transformer_dim 512, 6 layers, N_min 170 -> S = 3, N = 188.
    batch, T, spect_dim = 1, config.TRAIN_LENGTH, config.SPECT_DIM
    N_min = config.N_MIN

    x = torch.randn(batch, T, spect_dim)

    model = HybridBeatTracker()

    hat_phi, hat_t, b_e = model(x, epoch=1)
    print("warm-start (epoch=1): b_e == b_0?",
          torch.allclose(b_e, torch.full_like(b_e, model.b_0)))

    hat_phi, hat_t, b_e = model(x, epoch=10)
    print(f"T={T}, N_min={N_min} -> stages={config.DOWNSAMPLE_STAGES}, "
          f"N={config.NUM_CANDIDATES}, padded={model.downsample.padded_length}")
    print("hat_phi shape:", hat_phi.shape, " hat_t shape:", hat_t.shape)
    print("hat_phi range: [{:.4f}, {:.4f}]".format(hat_phi.min().item(), hat_phi.max().item()))
    diffs = hat_t[:, 1:] - hat_t[:, :-1]
    print("hat_t strictly increasing:", bool((diffs > 0).all()))
    print("b_e (post-warmup):", b_e.item())

    loss = hat_phi.sum() + hat_t.sum() + b_e.sum()
    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_total = len(trainable)
    n_grad = sum(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)
    print(f"Trainable parameters receiving gradient: {n_grad} / {n_total}")
    print(f"Total trainable parameters: {sum(p.numel() for p in trainable):,}")
    n_fixed = sum(1 for p in model.parameters() if not p.requires_grad)
    if n_fixed:
        print(f"(plus {n_fixed} fixed, non-trainable buffer(s), e.g. RoPE frequencies -- by design)")