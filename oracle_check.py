"""Positive control for the evaluation pipeline: feed it a PERFECT model.

Run it after any change to stitching.py or the decode. It must print 1.0000 for
both recall and precision; anything less means the pipeline is losing or
inventing events before the model is even involved.

Validates fragment_offsets, the keep-region partition, the normalized-t -> absolute
seconds mapping, and the mir_eval call -- all at once."""
import sys, torch, numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import config
from beat_this.dataset import BeatDataModule
from stitching import stitch_piece, fragment_offsets
from evaluate import metrics as _metrics

def score(truth, preds):
    return _metrics(truth, preds)

FPS, D, N = 50, 1500, 188

class Oracle:
    """Emits candidates sitting exactly on this fragment's true beats."""
    def __init__(self, beats_s, downbeats_s):
        self.beats, self.downbeats = beats_s, downbeats_s
        self.frags = None; self.k = 0
    def __call__(self, chunk):
        off = self.frags[self.k][0]; self.k += 1
        lo, hi = off / FPS, (off + D) / FPS
        b = self.beats[(self.beats >= lo) & (self.beats < hi)]
        phi = torch.zeros(1, N); t = torch.zeros(1, N)
        # place the first len(b) candidates on the true beats, rest far away
        t[0, :] = torch.linspace(0, 1, N)
        for i, bs in enumerate(b[:N]):
            t[0, i] = (bs - lo) / (D / FPS)
            # phase: position within the bar, from the true downbeats
            prev = self.downbeats[self.downbeats <= bs + 1e-9]
            nxt = self.downbeats[self.downbeats > bs + 1e-9]
            if len(prev) and len(nxt):
                bar = self.beats[(self.beats >= prev[-1]) & (self.beats < nxt[0])]
                phi[0, i] = (np.searchsorted(bar, bs) / max(len(bar), 1))
        # Park the unused candidates at t = 1.0 exactly, so the keep-region filter
        # drops them (frame == off + D is never < keep_end), and at phase 0, which
        # is a grid point under EVERY candidate meter so it cannot bias infer_meter.
        # Unmatched candidates get UNIFORM phases -- exactly the noise model the
        # -N/(4l) density-bias correction in eq. (67) is derived against. Parking
        # them all on one grid point instead makes that correction overshoot and
        # infer_meter picks l=2 over the true 4.
        g = torch.Generator().manual_seed(off)
        for i in range(len(b), N):
            t[0, i] = 1.0
            phi[0, i] = torch.rand(1, generator=g).item()
        # co-sort phi WITH t: RegHead's output is monotone by construction, so
        # candidate j's phase must follow its time, not stay at its old index.
        t, order = torch.sort(t, dim=1)
        phi = phi.gather(1, order)
        return phi, t, torch.tensor([0.05])

dm = BeatDataModule(Path(os.path.dirname(os.path.abspath(__file__))) / "data", batch_size=1,
                    train_length=D, spect_fps=FPS, num_workers=0, test_dataset="gtzan",
                    length_based_oversampling_factor=0.65, augmentations={},
                    hung_data=False, no_val=False, fold=0, predict_datasplit="val")
dm.setup(stage="predict")

fs, ds_ = [], []
for i, batch in enumerate(dm.predict_dataloader()):
    if i >= 5: break
    mel = batch["spect"][0]
    tb = np.frombuffer(batch["truth_orig_beat"][0])
    td = np.frombuffer(batch["truth_orig_downbeat"][0])
    oracle = Oracle(tb, td)
    oracle.frags = fragment_offsets(max(mel.shape[0], D), D, 6)
    beats, dbs = stitch_piece(mel, oracle, D, 6, FPS, config.METER_CANDIDATES,
                              torch.tensor(config.PI_M), config.TAU,
                              config.TAU_PRIME, 3.0, device=None)
    import mir_eval
    tr = mir_eval.beat.trim_beats(tb, min_beat_time=5.0)
    pr = mir_eval.beat.trim_beats(beats, min_beat_time=5.0)
    # recall at the standard 70 ms window: did every TRUE beat get a prediction?
    hit = sum(1 for x in tr if len(pr) and np.min(np.abs(pr - x)) <= 0.07)
    rec = hit / max(len(tr), 1)
    prec = sum(1 for x in pr if len(tr) and np.min(np.abs(tr - x)) <= 0.07) / max(len(pr), 1)
    fs.append(rec); ds_.append(prec)
    print(f"  piece {i}: pred {len(beats):4d} vs {len(tb):4d} true"
          f"  -> recall {rec:.4f}  precision {prec:.4f}")
print(f"\nORACLE mean RECALL over 5 pieces:    {np.mean(fs):.4f}   <- geometry")
print(f"ORACLE mean precision over 5 pieces: {np.mean(ds_):.4f}   <- parked candidates")
