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

# How tau is applied in decode (Algorithm 9). The document tests d_j <= tau
# with d_j the circular distance to the nearest grid point k/L. The grid's
# points are 1/L apart, so the LARGEST d any phase can have is 1/(2L): 0.167
# for L=3, 0.125 for L=4. A fixed tau = 0.2 therefore accepts EVERY candidate
# for L >= 3, and F sat at 0.43 with all 188 candidates emitted whatever the
# model had learned. tau only means anything as a fraction of the grid
# spacing, so the threshold is tau / L: a candidate is kept when it sits
# within tau of a grid point IN BEATS. Measured on a fitted batch: fixed tau
# F = 0.43, tau/L with merge F = 0.97.
TAU_GRID_RELATIVE = True

# Decode merge. With ~3 candidates per beat interval (188 over 30 s against
# ~2 beats/s), the candidates just before and just after a beat interpolate
# to phases within a few hundredths of the beat's grid point and pass any
# usable tau, so decode emitted ~2 candidates per beat. One emission per grid
# slot per bar: among accepted candidates closer in time than half a beat
# period with the same p_hat, keep the one nearest the grid.
DECODE_MERGE = True

# NO ALIGNBEAT COUNTERPART. tau' only exists in this document (Algorithm 10's
# gap-filling); AlignBeat has a single threshold. Section 5.2's own suggested
# starting point is tau' ~ 1.5 tau, to be swept on held-out data.
TAU_PRIME = 1.5 * TAU        # 0.3

# ---------------------------------------------------------------------------
# Loss weights.
# ---------------------------------------------------------------------------
# Phase scale b_phi (appendix "Calibrating the loss weights"). lambda_phi is
# NOT a free weight: it is 1/b_phi, the inverse scale of a wrapped-Laplace
# likelihood on the phase residual d_circ in [0, 0.5], exactly as lambda_L1 =
# 1/b is for timing. The appendix calibrates both against ONE BAR as the
# common unit: a typical well-trained timing error of 50 ms is 0.025 of a 2 s
# bar, a typical phase error is ~0.03 of a bar, so b ~ 0.025, b_phi ~ 0.03 and
# lambda_phi ~ 33 -- not the 3 of the document's worked examples. Phases
# already live in bar fractions, so b_phi carries over unchanged; B_0 below
# stays in window fractions (70 ms / 30 s), which is 0.035 of a 2 s bar.
#
# Like b_e, b_phi is LEARNED per fragment by a second ScaleHead with its own
# log-scale restoration term (train_and_infer.phase_term), warm-started at
# B_PHI_0 and held for the same warm-up as b_e. The appendix's early/late
# argument is why: untrained phase errors are ~0.25 and a fixed b_phi of 0.03
# would charge 8 nats and huge gradients per event.
B_PHI_0 = 0.03
B_PHI_MIN = 1e-3
# Ceiling on the learned b_phi (ScaleHead b_max). At b_phi = 0.15 the wrapped
# Laplace's density ratio between an exact hit and a half-turn miss is only
# e^(0.5/0.15) ~ 28 and Z = 0.29; beyond that the likelihood is nearly flat
# and its gradient 1 / b_phi too small to teach the phase head anything. Set
# so the phase channel can soften while predictions are poor (the appendix's
# early-training argument) but can never switch itself off.
# Measured need (2026-09-06, two uncapped 40-epoch runs): the phase error fell
# from 0.25 to 0.10 during the 12-epoch hold, then at the hand-over the head
# drove b_phi from 0.03 to 0.79 and 5.5 within two epochs, the slope 1/b_phi
# collapsed, and the phase error returned to chance for the rest of the run.
# The first capped run died of an unrelated non-finite step in epoch 0 (now
# caught by on_before_optimizer_step), not of the ceiling.
B_PHI_MAX = 0.15

# The appendix uses the single scale 1/b_phi in the E-step (matching) as well
# as in the loss. None here does exactly that. A float pins the E-step weight
# to that value instead, the escape hatch kept because a matching weight of
# 30 broke the E-step in the sweep of 2026-09-06 (candidates chosen by phase
# agreement, not by time) -- measured before the local time reparam and the
# other fixes, so it may no longer apply; the one-batch overfit test decides.
LAMBDA_PHI_ESTEP = None


# ---------------------------------------------------------------------------
# Candidate time parameterisation (eq. monotone vs a local offset).
# ---------------------------------------------------------------------------
# The document's eq. (monotone) is a GLOBAL cumulative normalisation,
# hat_t_j = cumsum(softplus r)_j / sum(softplus r). Moving one candidate onto
# its beat requires a coordinated change of every earlier softplus term and
# the total, and the per-event gradients through the cumsum cancel: training
# only the timing head on ONE fixed batch of 8 fragments for 300 full steps
# left the candidates on the uniform grid (spacing CV 0.07), and 40 real
# epochs did no better (CV 0.074). "local" anchors candidate j in its own slot
# and lets it move by up to half a slot, hat_t_j = (j + 0.5 + 0.5 tanh r_j)/N,
# which is still strictly increasing by construction (each hat_t_j lies in
# (j/N, (j+1)/N)) and fits the same batch in 60 steps (CV 0.29, every beat
# inside tolerance). "monotone" keeps the document's construction.
TIME_REPARAM = "local"

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

# Periodicity regulariser, eq. (48) in the appendix's normalised form
# (eq. periodicitynormalized): the deviation of each downbeat spacing from
# L * Delta_bar is divided by L * Delta_bar BEFORE squaring, so R is a
# dimensionless fraction of a bar and one sigma_R serves every tempo (a fixed
# absolute scale calibrated at 180 BPM would penalise the same relative
# deviation 9x more at 60 BPM). Then lambda_R := 1 / (2 sigma_R^2), eq.
# (lambdaR), i.e. R is a Gaussian likelihood on relative bar-length error.
#
# sigma_R is the empirical std of relative downbeat-spacing deviations over
# the fully-labeled training pieces (scratch script sigma_r.py), with
# Delta_bar taken over a 30 s window around each bar as the loss sees it.
# The distribution is heavy-tailed (pieces with tempo or meter changes), so
# the plain std is set by the tails; the value below is stated with both
# figures so the choice is visible. lambda_R is gated per fragment as before:
# annotated L -> eq. (48); latent L -> marginal over the candidate meters.
# Measured 2026-09-06 over 3744 pieces / 306584 bars: std 0.131, robust
# (1.4826 MAD) 0.007, median |dev| 0.005, 90th percentile 0.097. The std is
# the appendix's definition and is used; the robust figure would give
# lambda_R ~ 10^4, which the 2026-09-06 sweep showed destroys the phase head.
SIGMA_R = 0.131
LAMBDA_R = 1.0 / (2.0 * SIGMA_R ** 2)

# The tolerance in the units hat_t lives in (== B_0). Used in two places
# that AlignBeat ties together but this model must NOT:
EPS = F_MEASURE_TOLERANCE / WINDOW_SECONDS    # == B_0
# (1) The normaliser log(2 EPS_NORM + 2 b) of the timing likelihood
#     (alignbeat/criterion.py _per_candidate_time_term). Bounded below by
#     log(2 EPS_NORM), which is what keeps b_e from collapsing to 0 --
#     the document's own M log(2 b) is unbounded and did collapse.
EPS_NORM = EPS
# (2) The eps-insensitive DEAD ZONE on the residual (alignbeat eps_l1):
#     |dt| inside EPS_RESIDUAL costs nothing. AlignBeat can afford it because
#     its candidates carry a beat/non-beat CLASS; the timing channel only
#     refines. Here there is no class: the ONLY thing that can separate a
#     beat candidate from a non-beat one is where hat_t puts it. With a dead
#     zone of 70 ms, a UNIFORM grid of 188 candidates over 30 s (160 ms apart)
#     already has a candidate within tolerance of every beat, so the timing
#     term is ~0 without the model learning anything -- and the cheapest way
#     for the trunk to emit a uniform grid is identical features for every
#     candidate, which then cannot carry a phase either. Measured: with the
#     dead zone the model could not fit ONE batch of 8 fragments in 300 full
#     steps (phase error stuck at chance, spacing CV 0.004); without it the
#     phase starts moving. 0 restores the document's |dt| / b residual.
EPS_RESIDUAL = 0.0
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
