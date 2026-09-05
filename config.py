"""Hyperparameters, taken from the AlignBeat repository rather than reinvented.

Every value below is the one AlignBeat actually runs with -- from
launch_scripts/train.py's own argparse defaults, alignbeat/classes.py,
alignbeat/downsample.py, alignbeat/head.py, and .vscode/launch.json's
in-flight run arguments -- so an A/B against that codebase differs in the
formulation (continuous phase regression) and not in the training setup.

Three values have no AlignBeat counterpart at all, because the earlier draft
AlignBeat implements had no continuous phase and no learned timing scale.
Those are marked NO ALIGNBEAT COUNTERPART below and carry the paper's own
value instead; they are the only knobs here not inherited.
"""

# ---------------------------------------------------------------------------
# Backbone -- launch_scripts/train.py argparse defaults, lines 287-312, and
# the BeatThis(...) construction at train.py:119-126.
# ---------------------------------------------------------------------------
SPECT_DIM = 128
TRANSFORMER_DIM = 512        # --transformer-dim
N_LAYERS = 6                 # --n-layers
HEAD_DIM = 32                # train.py:126
STEM_DIM = 32                # train.py:121
FF_MULT = 4                  # train.py:119
N_HEADS = 16                 # --n-heads; implied by TRANSFORMER_DIM // HEAD_DIM, not a BeatThis kwarg
DROPOUT = {"frontend": 0.1, "transformer": 0.2}   # --dropout-frontend/-transformer

# ---------------------------------------------------------------------------
# Fragment geometry -- --train_length, --fps, --bpm_max, and
# alignbeat/downsample.py's stages_from_tempo().
# ---------------------------------------------------------------------------
FPS = 50                     # --fps
TRAIN_LENGTH = 1500          # --train_length, i.e. T in frames
WINDOW_SECONDS = TRAIN_LENGTH / FPS          # 30.0 s, matching head.py's own default
BPM_MAX = 340.0              # --bpm_max, alignbeat/downsample.py:14

# stages_from_tempo(1500, 50, 340) floors N at ceil(340 * 30 / 60) = 170 and takes
# 3 halvings (750 -> 375 -> 188), since a 4th (94) would drop below that floor.
# Feeding that same floor to this document's eq. (11) as N_min reproduces AlignBeat's
# schedule exactly: S = max{s : ceil(1500/2^s) >= 170} = 3, N = ceil(1500/8) = 188.
TEMPO_FLOOR = 170
N_MIN = TEMPO_FLOOR          # -> S = 3, N = 188, T' = 1504

# Derived by AlignBeat's own stages_from_tempo(), so there is one source of truth
# for N rather than two implementations that must be kept agreeing.
from downsample import stages_from_tempo as _stages_from_tempo   # noqa: E402
DOWNSAMPLE_STAGES, NUM_CANDIDATES, _FLOOR = _stages_from_tempo(
    TRAIN_LENGTH, FPS, BPM_MAX)          # -> (3, 188, 170)
assert _FLOOR == TEMPO_FLOOR, (_FLOOR, TEMPO_FLOOR)
DOWNSAMPLE_MODE = "learned"  # --downsample_mode

# ---------------------------------------------------------------------------
# Head widths -- alignbeat/head.py:23, SubsetSelectionHead's own defaults.
# ---------------------------------------------------------------------------
REDUCED_DIM = 256            # head.py feature_size: the shared trunk's width
HEAD_HIDDEN = 256            # head.py hidden_size, used by every branch

# Candidate self-attention -- --class_attention_layers/-heads and launch.json's
# L2_attn arm. AlignBeat applies it to the class branch; here it is the phase
# branch, which is that branch's replacement.
ATTENTION_LAYERS = 1         # --class_attention_layers
ATTENTION_HEADS = 4          # --class_attention_heads
ATTENTION_FF_MULT = 2        # head.py:47, dim_feedforward = hidden_size * 2
ATTENTION_DROPOUT = 0.0      # head.py:47
ATTENTION_FINAL_NORM = False # --class_attention_final_norm, off by default

# ---------------------------------------------------------------------------
# Optimization -- train.py:100 and the argparse defaults at 306-334.
# AdamW with a cosine schedule after linear warmup (pl_module.py:466, 506,
# CosineWarmupScheduler at 548).
# ---------------------------------------------------------------------------
LR = 3e-4                    # train.py:100, the --head_type subset value
WEIGHT_DECAY = 0.01          # --weight-decay
WARMUP_STEPS = 1000          # --warmup-steps; optimizer STEPS, not epochs
MAX_EPOCHS = 100             # --max-epochs
BATCH_SIZE = 8               # --batch-size
ACCUMULATE_GRAD_BATCHES = 8  # --accumulate-grad-batches

# ---------------------------------------------------------------------------
# Meter -- --meter_candidates from launch.json's latent-meter arm, and
# alignbeat/classes.py:19's METER_PRIOR, measured in docs/METER_DISTRIBUTION.md.
# This is exactly the paper's own pi_M (eq. 4), used at inference in eq. (67).
# ---------------------------------------------------------------------------
METER_CANDIDATES = [2, 3, 4, 6]              # --meter_candidates "2,3,4,6"
METER_PRIOR = {2: 0.0447, 3: 0.0838, 4: 0.8612, 6: 0.0068}
PI_M = [METER_PRIOR[l] for l in METER_CANDIDATES]

# ---------------------------------------------------------------------------
# Decoding and tolerance -- --tau_beat/--tau_downbeat and
# alignbeat/classes.py:13's F_MEASURE_TOLERANCE.
# ---------------------------------------------------------------------------
F_MEASURE_TOLERANCE = 0.07   # seconds; the standard beat-tracking window
TAU = 0.2                    # --tau_beat == --tau_downbeat == 0.2

# NO ALIGNBEAT COUNTERPART. tau' only exists in this document (Algorithm 10's
# gap-filling); AlignBeat has a single threshold. Section 5.2's own suggested
# starting point is tau' ~ 1.5 tau, to be swept on held-out data.
TAU_PRIME = 1.5 * TAU        # 0.3

# ---------------------------------------------------------------------------
# Loss weights.
# ---------------------------------------------------------------------------
# NO ALIGNBEAT COUNTERPART. lambda_phi weights a CIRCULAR DISTANCE against the
# timing term; AlignBeat's nearest analogue, --omega_db 4.0, weights a discrete
# downbeat CLASS against beat/background, which is a different quantity on a
# different scale. Left at the paper's own worked-example value (Sections
# 3.3-3.4 use lambda_phi = 3.0) and flagged as needing its own sweep.
LAMBDA_PHI = 3.0

# AlignBeat's README lists a --lambda_r flag for the periodicity regularizer,
# but no such argument exists anywhere in launch_scripts/train.py or the
# alignbeat package -- the regularizer is documented, not wired up. Off is
# therefore what AlignBeat actually runs, and eq. (53)'s third line vanishes.
LAMBDA_R = 0.0

# ---------------------------------------------------------------------------
# Timing scale b (Section 3.7).
# ---------------------------------------------------------------------------
# NO ALIGNBEAT COUNTERPART for the head itself -- AlignBeat's --predict_precision
# belongs to the earlier draft's per-candidate precision, not this per-fragment
# Laplace scale. The warm-start VALUE, though, is inherited: AlignBeat seeds its
# precision head at F_MEASURE_TOLERANCE, the same 70 ms notion of "acceptable
# timing spread". Converted into this document's units, where t_i is normalized
# to [0, 1] across the fragment rather than measured in seconds.
B_0 = F_MEASURE_TOLERANCE / WINDOW_SECONDS   # 0.07 s / 30 s = 0.002333
B_MIN = 1e-4                                  # alignbeat/criterion.py:24, same units (window fraction)

# eps-insensitive timing (alignbeat/criterion.py:30, eps_l1): residuals inside
# the 70 ms tolerance cost nothing, so they cannot drive b_e toward zero.
EPS = F_MEASURE_TOLERANCE / WINDOW_SECONDS    # == B_0
# AlignBeat's third piece, the Gamma prior on 1/b (_precision_prior), is
# deliberately not carried over: the two terms above already bound the loss.

# Warm-start length for the scale (eq. 51's E_0), as AlignBeat sets it: the
# precision head is held for the first 30% of the run
# (beat_this/model/pl_module.py:347, `current_epoch < max_epochs * 0.3`), so
# E_0 scales with --max-epochs rather than being a fixed epoch count.
# train.py derives E_0 = int(WARMUP_FRACTION * max_epochs) from
# --scale-warmup-fraction; WARMUP_EPOCHS below is only the constructor
# default for HybridBeatTracker when it is built outside train.py.
WARMUP_FRACTION = 0.3
WARMUP_EPOCHS = int(WARMUP_FRACTION * MAX_EPOCHS)   # 30 for the default 100 epochs
