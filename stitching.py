"""Whole-piece inference: cut a piece into fragments, decode each, reassemble.

The model sees a fixed 1500-frame window, but evaluation is on whole pieces
(~95 s), so a piece has to be cut into overlapping fragments and the
per-fragment decodes joined back into one event sequence.

`fragment_offsets` is vendored from AlignBeat's alignbeat/stitching.py: it is
pure geometry over frame indices, with no reference to what a fragment decodes
into, so it carries over to this formulation unchanged. `stitch_piece` below
does NOT: AlignBeat's version calls its own decode_events, which returns
{downbeat, beat, background} classes, and this one runs Section 5's decode
instead.
"""

import numpy as np
import torch

from train_and_infer import (circ_dist, decode, infer_meter,
                             meter_consistency_correction)


def fragment_offsets(total_frames, fragment_frames, border_frames):
    """Offsets o_1 = 0, o_2 = D - 2*beta, ... covering [0, total_frames).

    Returns (offset, keep_start, keep_end) per fragment, the keep regions
    forming a partition of the piece: every frame is decoded by exactly one
    fragment, and the borders -- where a fragment has least context -- are
    discarded in favour of a neighbour that saw them mid-window.

    Vendored from alignbeat/stitching.py, unchanged.
    """
    if fragment_frames <= 2 * border_frames:
        raise ValueError(
            f"border_frames {border_frames} must be under half the window "
            f"{fragment_frames}; otherwise consecutive fragments cannot meet")

    stride = fragment_frames - 2 * border_frames
    offsets = []
    offset = 0
    while True:
        offsets.append(offset)
        if offset + fragment_frames >= total_frames:
            break
        offset += stride

    # Slide the last fragment left so it ends exactly at the piece end, rather
    # than zero-padding the one fragment that decodes the tail against an input
    # the model never saw in training. It then overlaps its predecessor by more
    # than 2*beta, so keep regions are clamped to a high-water mark below to
    # stay a partition. Padding survives only for a piece shorter than D.
    if len(offsets) > 1 and total_frames - offsets[-1] < fragment_frames:
        offsets[-1] = total_frames - fragment_frames

    fragments = []
    covered_to = 0
    for index, offset in enumerate(offsets):
        first, last = index == 0, index == len(offsets) - 1
        keep_start = 0 if first else max(offset + border_frames, covered_to)
        keep_end = total_frames if last else offset + fragment_frames - border_frames
        keep_end = max(keep_end, keep_start)
        covered_to = keep_end
        fragments.append((offset, keep_start, keep_end))
    return fragments


def stitch_piece(mel, model, fragment_frames, border_frames, fps,
                 candidate_meters, pi_M, tau, tau_prime, lambda_phi,
                 device=None):
    """Decode a whole piece. Returns (beat_seconds, downbeat_seconds).

    Section 5 per fragment -- infer L (eq. 67), nearest-grid-point decode, then
    MeterConsistencyCorrection (Alg. 10) -- then map each fragment's normalized
    t_hat in [0,1] back to absolute frames and keep only the events inside that
    fragment's own keep region.

    Meter is inferred PER FRAGMENT, not per piece. The document infers L per
    fragment too (eq. 67 reads that fragment's own N candidates), but a piece is
    free to disagree with itself across fragments here, which a piece-level
    estimate would not allow. Worth knowing when reading downbeat numbers.
    """
    total_frames = mel.shape[0]
    pad = None
    if total_frames < fragment_frames:
        pad = fragment_frames - total_frames
        mel = torch.nn.functional.pad(mel, (0, 0, 0, pad))

    frames_for_offsets = max(total_frames, fragment_frames)
    fragments = fragment_offsets(frames_for_offsets, fragment_frames, border_frames)

    beats, downbeats = [], []
    for offset, keep_start, keep_end in fragments:
        chunk = mel[offset:offset + fragment_frames].unsqueeze(0)
        if device is not None:
            chunk = chunk.to(device)
        with torch.no_grad():
            hat_phi, hat_t, _ = model(chunk.float())
        hat_phi, hat_t = hat_phi[0].float(), hat_t[0].float()

        hat_L = infer_meter(hat_phi, candidate_meters, pi_M, lambda_phi)
        p_hat, d, t_hat, B = decode(hat_phi, hat_t, hat_L, tau)
        B = meter_consistency_correction(B, hat_L, p_hat, d, t_hat, tau, tau_prime)

        for phase, t_norm in B:
            frame = offset + t_norm * fragment_frames
            if not (keep_start <= frame < keep_end):
                continue
            if pad is not None and frame >= total_frames:
                continue          # inside the zero padding of a short piece
            seconds = frame / fps
            beats.append(seconds)
            if phase == 0:
                downbeats.append(seconds)

    beats = np.unique(np.asarray(beats, dtype=np.float64))
    downbeats = np.unique(np.asarray(downbeats, dtype=np.float64))
    return beats, downbeats
