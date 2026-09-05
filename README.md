# Beat Tracking as Phase and Time Regression

Beat and downbeat detection as a **regression** problem: a fixed number `N` of
candidate predictors independently regress a continuous event time `t̂ⱼ` and a
continuous, circular phase-within-the-bar `φ̂ⱼ ∈ [0,1)`, deliberately
overgenerating more candidates than the `M` true events. A latent
order-preserving alignment `σ` decides which candidate is held responsible for
which annotated event, and is resolved per fragment by an `O(NM)` dynamic
program. Both regression terms are maximum likelihood under a Laplace residual —
one on the line, one on a circle.

This implements the current draft of the accompanying document. It supersedes an
earlier formulation in which each candidate was classified into
`{downbeat, beat, background}`: there is **no classifier here**, no background
class, and no separate meter head. `PhaseHead` emits a point on the unit circle
and `atan2` maps it to `[0,1)`, so its computation needs no meter at all.

## Layout

| file | what |
| --- | --- |
| `heads.py` | `PhaseHead` (eq. 16), `RegHead` + monotone reparameterization (eq. 17), `ScaleHead` (eq. 51) |
| `hybrid_beat_tracker.py` | the model: Beat This!'s encoder + `Downsample` + the candidate heads |
| `train_and_infer.py` | the DP (Alg. 2), E-step (Alg. 4), M-step (Alg. 5), inference and decoding (Alg. 9, 10) |
| `pl_module.py` | `PLPhaseTimeRegression`, the Lightning module |
| `train.py` | launch script (Beat This!'s `BeatDataModule`, augmentation, Trainer) |
| `config.py` | every hyperparameter, each with the source it came from |
| `evaluate.py` | score checkpoints on whole held-out pieces with mir_eval |
| `stitching.py` | whole-piece decoding: fragment offsets, keep regions, reassembly |
| `oracle_check.py` | positive control: a perfect model must score 1.0 through `evaluate.py` |
| `backbone.py`, `roformer.py` | Beat This!'s encoder, vendored (MIT) |
| `downsample.py` | `T → N` candidate downsampling, vendored |

Nothing here imports the earlier draft's code, which is why the vendored files
are copied rather than imported.

## Running

Training needs Beat This!'s preprocessed data — `data/audio/spectrograms/*.npz`
and `data/annotations/` — and its `beat_this` package importable for
`BeatDataModule` alone.

```bash
python train.py --name phase --gpu 0 --fold 0 --num-workers 8
```

```bash
python evaluate.py --checkpoints "checkpoints/phase*.ckpt" --fold 0 --per-dataset
python oracle_check.py     # must print 1.0000 for recall and precision
```

`train_and_infer.py` runs standalone with no data at all: it checks the DP
against brute-force enumeration over 200 seeds, exercises both E-step branches,
trains briefly on synthetic fragments, and decodes. Use it as the correctness
harness; use `train.py` for results.

## What is implemented, and what is not

Implemented: the two heads and the monotone time reparameterization; the
correspondence DP with `L_agree` (eq. 21) as its skip cost; the fully-labeled
(`ind=0`) and beat-only (`ind=1`) training cases; the hard `φ₀` resolution
(eq. 26); the periodicity regularizer (eq. 48); meter inference and
nearest-grid-point decoding with the density-bias correction (eq. 67); and
`MeterConsistencyCorrection` (Alg. 10).

Not implemented, and not stubbed:

- **`MixedMeterTarget` (Alg. 1)** — the forward-backward pass over the
  mixed-meter DBN. Beat-only fragments therefore assume one meter per fragment
  (`--beat-only-meter`, default 4) instead of marginalizing over the candidate
  set.
- **Soft EM and direct marginal SGD** (Algs. 7, 8) — `σ` is hardened, so this is
  Viterbi EM.
- **The joint `(σ, φ₀)` recursion** (eqs. 43, 44). The E-step is the phase-blind
  scheme, whose costs §3.3–3.4 work through in detail.
- **Metrics during training.** Validation logs loss only. Use `evaluate.py`
  on the checkpoints instead: it decodes whole held-out pieces through §5 and
  Algorithm 10 and scores them with mir_eval at the same 70 ms tolerance and 5 s
  trim as the rest of the pipeline, so its numbers are directly comparable.
- **`π̂_L`** (eq. 25) is uniform rather than estimated from the fully-labeled
  corpus.

Three hyperparameters have no inherited value and want their own sweep — they
are flagged in `config.py`: `λ_φ`, `τ′`, and `E₀`.

## Licence

MIT. `backbone.py` and `roformer.py` are derived from
[CPJKU/beat_this](https://github.com/CPJKU/beat_this) (MIT); `roformer.py`
descends from lucidrains' BS-RoFormer (MIT).
