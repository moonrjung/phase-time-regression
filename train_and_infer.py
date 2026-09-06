"""
Training and inference for the hybrid beat tracker (heads.py / hybrid_beat_tracker.py).

Scope, stated honestly: implements the fully-labeled (ind=0) and the
single-known-meter beat-only (ind=1, phase-blind) training cases, plus
inference decoding with MeterConsistencyCorrection. Does NOT implement the
full mixed-meter DBN / MixedMeterTarget forward-backward (restart
mechanism, per-event meter changes) -- that is a substantial further
extension, deliberately out of scope here to keep this bounded and
verifiable rather than attempting every case in one pass.

Batch size of 1 throughout (one fragment at a time) -- correspondence and
decoding are inherently per-fragment operations (M and N differ per
fragment), so batching them naively is its own, separate complication,
not attempted here.
"""

import math
from typing import NamedTuple
import os

import numpy as np
import torch

# Optional compiled kernel for the correspondence DP. The Python loop below is
# the readable reference and stays the fallback; measured at M=60, N=188 -- a
# typical fragment -- it costs 42 ms, so ~337 ms of every batch of 8 is spent
# here, against a step time of ~2.3 s with the GPU at 12-25%. The E-step, not
# the network, is what training waits on.
#
# AlignBeat's own dp.py has a kernel for the same problem, but its recursion
# has NO skip cost: it is min over j' <= j of D[i-1,j'-1] + cost, which
# np.minimum.accumulate solves in one vectorised pass. Eq. (23) charges
# L_agree(j) on the skip branch, so that running-minimum trick does not apply
# and the loop below is scalar rather than a copy of theirs.
#
# ALIGNBEAT_NO_NUMBA=1 forces the Python path, so the two can be A/B'd; they
# must agree exactly.
try:
    from numba import njit as _njit

    @_njit(cache=True, fastmath=False)
    def _dp_kernel(cost, agree, choice):
        M, N = cost.shape
        INF = np.inf
        previous = np.empty(N + 1, dtype=np.float64)   # row i-1
        current = np.empty(N + 1, dtype=np.float64)    # row i

        # Row 0: no event matched yet, so the first j candidates were all
        # skipped and each owes its own L_agree(j) -- a prefix sum.
        previous[0] = 0.0
        for j in range(1, N + 1):
            previous[j] = previous[j - 1] + agree[j - 1]

        for i in range(1, M + 1):
            current[0] = INF
            for j in range(1, N + 1):
                c_skip = current[j - 1] + agree[j - 1]
                c_match = previous[j - 1] + cost[i - 1, j - 1]
                if c_match < c_skip:
                    current[j] = c_match
                    choice[i, j] = 1          # match
                else:
                    current[j] = c_skip
                    choice[i, j] = 0          # skip
            for j in range(N + 1):
                previous[j] = current[j]
        return previous[N]

    _HAVE_NUMBA = not os.environ.get("ALIGNBEAT_NO_NUMBA")
except ImportError:                                          # pragma: no cover
    _HAVE_NUMBA = False

import config


def circ_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """d_circ(a,b) := min(|a-b| mod 1, 1-|a-b| mod 1), the unit-circle distance."""
    d = torch.abs(a - b) % 1.0
    return torch.minimum(d, 1.0 - d)


def match_cost_matrix(t_true: torch.Tensor, phi_true: torch.Tensor,
                       hat_t: torch.Tensor, hat_phi: torch.Tensor,
                       b_e: torch.Tensor, lambda_phi: float,
                       phase_blind: bool) -> torch.Tensor:
    """eq. (matchcostneg) / eq. (matchcostBstar): L'_match(y_i, hat_y_j),
    for every (i,j) pair, as an (M, N) cost matrix.
    phase_blind=True: timing only (t_true, phi_true unused for phase term).
    phase_blind=False: timing + circular phase (phi_true must be given)."""
    M, N = t_true.shape[0], hat_t.shape[0]
    timing = eps_l1(hat_t[None, :], t_true[:, None], b_e)  # (M, N)
    if phase_blind:
        return timing
    phase = lambda_phi * circ_dist(phi_true[:, None], hat_phi[None, :])  # (M, N)
    return timing + phase


def eps_l1(t_hat: torch.Tensor, t_target: torch.Tensor, b: torch.Tensor,
           eps: float = config.EPS_RESIDUAL) -> torch.Tensor:
    """alignbeat/criterion.py:96 eps_l1 -- eps-insensitive L1, exactly zero
    within eps of the annotation, in the window-fraction units t_hat lives in.
    eps = 0 (config.EPS_RESIDUAL's default) is the document's plain |dt| / b;
    see config.py for why the dead zone is off for this model."""
    return (t_hat - t_target).abs().sub(eps).clamp(min=0.0) / b


def time_term(residual: torch.Tensor, b: torch.Tensor,
              eps: float = config.EPS_NORM) -> torch.Tensor:
    """alignbeat/criterion.py:501 _per_candidate_time_term -- -log p(r | b) for
    the uniform-core / Laplace-tail density, gradient split so the timing
    channel moves hat_t and the precision channel moves b. residual is already
    eps-insensitive; log(2 eps + 2 b) is bounded below by log(2 eps), which
    the previous M log(2 b) was not."""
    localisation = residual / b.detach()
    precision = residual.detach() / b + torch.log(2.0 * eps + 2.0 * b)
    return localisation + precision


def wrapped_laplace_log_norm(b_phi: torch.Tensor) -> torch.Tensor:
    """log Z(b_phi) for the Laplace density exp(-d / b_phi) / Z wrapped on the
    unit circle, d in [0, 1/2]: Z = int_{-1/2}^{1/2} exp(-|x| / b) dx
    = 2 b (1 - exp(-1 / (2 b))). Tends to 2b for small b (the line's Laplace
    normaliser) and to 1 for large b (uniform on the circle), so it is
    bounded: log Z in [log 2 b_min, 0]. This is eq. (scalerestore) for phase."""
    return torch.log(2.0 * b_phi * (1.0 - torch.exp(-1.0 / (2.0 * b_phi))))


def phase_term(dist: torch.Tensor, b_phi: torch.Tensor) -> torch.Tensor:
    """-log p(d | b_phi) for the wrapped Laplace, per event, with the same
    gradient split as time_term: the localisation channel moves hat_phi with
    slope 1 / b_phi (the appendix's lambda_phi := 1 / b_phi), the precision
    channel moves b_phi toward the MLE, where b_phi equals the typical phase
    residual. dist is a circular distance in [0, 1/2]."""
    localisation = dist / b_phi.detach()
    precision = dist.detach() / b_phi + wrapped_laplace_log_norm(b_phi)
    return localisation + precision


def phase_weight(b_phi: torch.Tensor, lambda_phi_estep: float | None = config.LAMBDA_PHI_ESTEP) -> float:
    """The E-step's phase weight for one fragment: 1 / b_phi (the appendix's
    single scale) unless config.LAMBDA_PHI_ESTEP pins it."""
    if lambda_phi_estep is not None:
        return float(lambda_phi_estep)
    return float(1.0 / b_phi.detach())


def l_agree(hat_phi: torch.Tensor, t_true: torch.Tensor, phi_true: torch.Tensor,
            hat_t: torch.Tensor) -> torch.Tensor:
    """eq. (agree): for every candidate j, its agreement cost against the
    nearest bracketing ground-truth pair's own interpolated target phase.
    Only meaningful when phi_true is fully known (ind=0) -- see the
    document's own resolution of the ind=1 circularity concern.
    Returns an (N,) tensor, one value per candidate."""
    M = t_true.shape[0]

    # Vectorised over candidates. The scalar loop this replaces called
    # torch.searchsorted and .item() once per candidate, which cost 7.3 ms per
    # fragment -- ~59 ms of every batch of 8, on top of the DP's own 337 ms.
    # Same arithmetic, same clamping into a valid bracket, same StopGradient on
    # the target: gradient still reaches hat_phi alone, never hat_t.
    idx = torch.searchsorted(t_true, hat_t.detach().contiguous())
    idx = idx.clamp(1, max(M - 1, 1))

    t_i = t_true[idx - 1]
    t_i1 = t_true[idx]
    phi_i = phi_true[idx - 1]
    phi_i1 = phi_true[idx]

    span = (t_i1 - t_i).clamp_min(1e-8)
    w = (hat_t.detach() - t_i) / span
    # Interpolate along the SHORTEST ARC of the circle, not along the raw
    # numbers. Phase advances by +1/L from beat to beat, so across the bar
    # boundary it goes 0.75 -> 0.0 by wrapping UP through 0.83, 0.92; the raw
    # difference (-0.75) would send the target DOWN through 0.5 and 0.25 --
    # exactly onto grid points -- and decode then emitted those unmatched
    # candidates as beats (measured: 1 extra beat per bar on a fitted batch).
    delta = ((phi_i1 - phi_i + 0.5) % 1.0) - 0.5          # signed, in (-0.5, 0.5]
    target = (phi_i + w * delta) % 1.0
    return circ_dist(hat_phi, target.detach())


def subset_select_dp(cost: torch.Tensor, agree: torch.Tensor | None = None):
    """Algorithm 2: SubsetSelectDP. cost: (M, N) match costs.
    agree: optional (N,) skip cost, folded in per eq. (dprecursion) -- only
    valid when phi_i is already known for every i (ind=0); pass None for
    the phase-blind (ind=1) search, per the document's own resolution.
    Returns hat_sigma as a length-M LongTensor of 1-indexed candidate
    positions (hat_sigma[i] = j means event i matched to candidate j)."""
    M, N = cost.shape
    # A NaN anywhere makes every comparison false, the DP never records a
    # match, and the backtrack walks j below 0 (seen as an IndexError deep in
    # the kernel on 2026-09-06 after a batch produced non-finite outputs under
    # 16-bit mixed precision). Fail here, with the cause named.
    if not torch.isfinite(cost).all() or (agree is not None and not torch.isfinite(agree).all()):
        raise ValueError("non-finite matching cost: the model emitted NaN/Inf (hat_t, hat_phi "
                         "or b_e); callers should skip such fragments (pl_module does)")

    if _HAVE_NUMBA:
        cost_np = np.ascontiguousarray(cost.detach().cpu().numpy(), dtype=np.float64)
        agree_np = (np.ascontiguousarray(agree.detach().cpu().numpy(), dtype=np.float64)
                    if agree is not None else np.zeros(N, dtype=np.float64))
        choice = np.zeros((M + 1, N + 1), dtype=np.int8)
        _dp_kernel(cost_np, agree_np, choice)
        hat_sigma = [0] * M
        i, j = M, N
        while i > 0:
            if choice[i, j] == 1:
                hat_sigma[i - 1] = j
                i, j = i - 1, j - 1
            else:
                j -= 1
        return torch.tensor(hat_sigma, dtype=torch.long, device=cost.device)

    INF = float("inf")
    D = [[0.0] * (N + 1) for _ in range(M + 1)]
    choice = [[None] * (N + 1) for _ in range(M + 1)]
    for i in range(1, M + 1):
        D[i][0] = INF
    # Row 0: no event matched yet, so every one of the first j candidates was
    # skipped and each owes its own L_agree(j). A running prefix sum. Without
    # this the row stays at 0 and candidates skipped BEFORE the first match are
    # free, while interior and trailing skips are charged -- so the search
    # minimizes something other than eq. (22), its own loss. (Note eq. (23)
    # literally writes D[0, j] = 0; eq. (22) sums L_agree over every unmatched
    # j. The two disagree, and this follows eq. (22), the actual objective.)
    # When agree is None -- the ind=1 phase-blind search, where L_agree is not
    # yet computable -- the row correctly stays all zeros.
    for j in range(1, N + 1):
        D[0][j] = D[0][j - 1] + (agree[j - 1].item() if agree is not None else 0.0)
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            skip_cost = (agree[j - 1].item() if agree is not None else 0.0)
            c_skip = D[i][j - 1] + skip_cost
            c_match = D[i - 1][j - 1] + cost[i - 1, j - 1].item()
            if c_match < c_skip:
                D[i][j], choice[i][j] = c_match, "match"
            else:
                D[i][j], choice[i][j] = c_skip, "skip"
    # backtrack
    hat_sigma = [0] * M
    i, j = M, N
    while i > 0:
        if choice[i][j] == "match":
            hat_sigma[i - 1] = j
            i, j = i - 1, j - 1
        else:
            j -= 1
    return torch.tensor(hat_sigma, dtype=torch.long, device=cost.device)


def hard_phi0(hat_sigma: torch.Tensor, hat_phi: torch.Tensor, L: int,
              lambda_phi: float, pi_L: torch.Tensor) -> tuple[int, torch.Tensor]:
    """Section 2.4's hard phi_0 construction for ONE meter hypothesis L: for
    each of the L offsets p, score sum_i lambda_phi * d_circ((p+i-1)/L,
    hat_phi_sigma(i)) - log pi_L[p]. Returns (best p, per-event phi_i under
    that p, that best cost) -- phi_i for event i is ((p + i - 1) % L) / L.

    The cost is returned so that a caller trying several L can compare the
    hypotheses: it is the negative log of the (unnormalised) joint of the
    matched phases and phi_0 under L. No L-dependent bias correction is
    needed here, unlike infer_meter's -N/(4L): the implied phases are a
    fixed progression, so under a random hat_phi the expected circular
    distance is 1/4 for every L, not 1/(4L)."""
    M = hat_sigma.shape[0]
    matched_phi = hat_phi[hat_sigma - 1]  # (M,), hat_phi_{sigma(i)} for each i
    best_cost, best_p = float("inf"), 0
    device, dtype = matched_phi.device, matched_phi.dtype
    for p in range(L):
        implied = torch.tensor([((p + i) % L) / L for i in range(M)],
                               device=device, dtype=dtype)
        cost = lambda_phi * circ_dist(implied, matched_phi).sum() - torch.log(pi_L[p] + 1e-12)
        if cost.item() < best_cost:
            best_cost, best_p = cost.item(), p
    phi_i = torch.tensor([((best_p + i) % L) / L for i in range(M)],
                         device=device, dtype=dtype)
    return best_p, phi_i, best_cost


class MeterHypothesis(NamedTuple):
    """One candidate meter L for a beat-only fragment, after hard phi_0
    resolution under that L. Produced by e_step (ind=1), consumed by
    m_step_loss's marginal periodicity term."""
    L: int
    log_prior: float        # log pi_M(L), the corpus meter prior (eq. 4)
    phi_i: torch.Tensor     # (M,) resolved phases of the matched events under L
    score: float            # hard_phi0's cost - log pi_M(L): lower is better


def e_step(t_true: torch.Tensor, hat_t: torch.Tensor, hat_phi: torch.Tensor,
           b_e: torch.Tensor, lambda_phi: float, ind: int,
           phi_true: torch.Tensor | None = None,
           meter_candidates: list[int] | None = None,
           pi_M: torch.Tensor | list[float] | None = None,
           pi_L: dict[int, torch.Tensor] | None = None):
    """Algorithm 4: EStep.

    ind=0 (fully-labeled): phi_true given directly, L_agree computable
    upfront and folded into the DP's skip cost.

    ind=1 (beat-only): the meter L is LATENT. Section 2.5 says a beat-only
    track carries no meter information of its own, and the eq. (48)
    discussion forbids "assigning an incorrect default meter". So instead
    of one assumed L, every candidate meter in `meter_candidates` (the
    document's own set, eq. 4's pi_M) is tried:
      1. phase-blind search on timing alone -- the DP does not depend on L,
         so hat_sigma is shared by every hypothesis;
      2. for EACH L, the hard phi_0 resolution of Section 2.4 (hard_phi0),
         giving per-L phases phi_i^L and a cost;
      3. the meter with the lowest cost - log pi_M(L) is the hard-EM choice
         hat_L. Its phi_i^L feeds the phase term and L_agree, exactly as the
         single-L version did;
      4. ALL hypotheses are returned as MeterHypothesis records so that the
         M-step can marginalise the periodicity regulariser over L rather
         than trusting hat_L alone (see m_step_loss).
    pi_L maps each L to its phi_0 prior (eq. 25); None means uniform.

    Returns (hat_sigma, phi_i_for_matched_events, agree_for_unmatched,
    meter_hypotheses_or_None)."""
    if ind == 0:
        assert phi_true is not None
        agree = l_agree(hat_phi, t_true, phi_true, hat_t)
        cost = match_cost_matrix(t_true, phi_true, hat_t, hat_phi, b_e, lambda_phi, phase_blind=False)
        hat_sigma = subset_select_dp(cost, agree).detach()
        # eq. (22) sums L_agree over every unmatched candidate for BOTH annotation
        # types, so the ind=0 branch returns it too rather than dropping it. No
        # recomputation: phi_i is observed here, so the same tensor the DP already
        # used is simply restricted to the candidates sigma-hat left unmatched.
        # It still carries gradient into hat_phi (l_agree detaches its own target),
        # which is the direction Section 2.4 requires.
        matched = set(hat_sigma.tolist())          # built once, not per candidate
        unmatched = [j for j in range(1, hat_t.shape[0] + 1) if j not in matched]
        if unmatched:
            unmatched_idx = torch.tensor(unmatched, dtype=torch.long) - 1
            agree_unmatched = agree[unmatched_idx]
        else:
            agree_unmatched = torch.zeros(0)
        return hat_sigma, phi_true.detach(), (unmatched, agree_unmatched), None
    else:
        assert meter_candidates is not None and pi_M is not None, \
            "beat-only fragments need the candidate meter set and its prior"
        cost = match_cost_matrix(t_true, None, hat_t, hat_phi, b_e, lambda_phi, phase_blind=True)
        hat_sigma = subset_select_dp(cost)  # no skip cost yet -- phi unknown
        hat_sigma = hat_sigma.detach()

        # One phi_0 resolution per candidate meter. hat_phi is detached: the
        # E-step decides, the M-step differentiates (Algorithm 3).
        hyps = []
        for idx, L in enumerate(meter_candidates):
            if pi_L is not None and L in pi_L:
                pi_L_of_L = pi_L[L].to(hat_phi.device)
            else:
                pi_L_of_L = torch.full((L,), 1.0 / L, device=hat_phi.device)
            _, phi_L, cost_L = hard_phi0(hat_sigma, hat_phi.detach(), L, lambda_phi, pi_L_of_L)
            log_prior = math.log(float(pi_M[idx]) + 1e-12)
            hyps.append(MeterHypothesis(L=L, log_prior=log_prior, phi_i=phi_L,
                                        score=cost_L - log_prior))
        best = min(hyps, key=lambda h: h.score)     # hard-EM over L
        phi_i = best.phi_i

        matched = set(hat_sigma.tolist())          # built once, not per candidate
        unmatched = [j for j in range(1, hat_t.shape[0] + 1) if j not in matched]
        if unmatched:
            unmatched_idx = torch.tensor(unmatched) - 1
            agree = l_agree(hat_phi[unmatched_idx], t_true, phi_i, hat_t[unmatched_idx])
        else:
            agree = torch.zeros(0)
        return hat_sigma, phi_i.detach(), (unmatched, agree), hyps


def periodicity_R(matched_t: torch.Tensor, downbeat_idx: torch.Tensor,
                  bar_lengths: torch.Tensor) -> torch.Tensor:
    """eq. (48), normalised (appendix eq. periodicitynormalized):
    R = sum_k ( (t_sigma(i_{k+1}) - t_sigma(i_k) - L_k Delta_bar) / (L_k Delta_bar) )^2.

    matched_t:    (M,) predicted times of the matched events, in time order.
    downbeat_idx: (K,) event indices i_1 < ... < i_K of the downbeats.
    bar_lengths:  (K-1,) beats per bar between consecutive downbeats, L_k.
                  The document writes a single per-track L; passing it per
                  bar is the same thing for a constant meter and stays
                  correct when the meter changes inside the fragment
                  (Section 3.1), which phases_from_downbeats already handles
                  the same way.
    Delta_bar is the model's own mean beat period over the matched events,
    the model-dependent quantity the document writes as Delta_bar(theta, x).
    Gradient flows through the predicted times only; L_k is data."""
    db_times = matched_t[downbeat_idx]
    spacings = db_times[1:] - db_times[:-1]
    M = matched_t.shape[0]
    delta_bar = (matched_t[-1] - matched_t[0]) / max(M - 1, 1)
    expected = bar_lengths.to(spacings) * delta_bar
    # Appendix eq. (periodicitynormalized): relative to the expected bar
    # duration, so R is a dimensionless fraction of a bar and one sigma_R
    # (config.SIGMA_R) serves every tempo. Gradient flows through both the
    # spacing and the expected duration, as the appendix writes it.
    return (((spacings - expected) / expected.clamp_min(1e-6)) ** 2).sum()


def marginal_periodicity(matched_t: torch.Tensor, meter_hyps: list,
                         lambda_R: float) -> torch.Tensor | None:
    """The periodicity term for a beat-only fragment, where L is unknown.

    The document's eq. (48) discussion excludes such fragments from R rather
    than assign a default meter. This goes one step further and MARGINALISES
    over the candidate meters instead of excluding them: treat
    exp(-lambda_R * R_L) as the likelihood of the predicted downbeat spacing
    under meter L, weight by the corpus prior pi_M(L), and take the
    negative log of the sum:

        R_marg = -log sum_L pi_M(L) exp(-lambda_R * R_L)

    Why this form and not the average of R_L over L: the average is
    minimised at a spacing of E[L] * Delta_bar, about 3.7 beats under the
    corpus prior, which is the bar length of no meter at all. The soft-min
    above is minimised at L * Delta_bar for whichever L fits best, so it
    still pushes toward SOME periodic bar structure while letting the data
    overrule the prior -- e.g. a clear 3/4 piece wins against the 0.86 prior
    mass on 4/4 once its R_3 is small enough.

    Each R_L is evaluated on that L's OWN downbeat set, i.e. the phi_i^L that
    hard_phi0 resolved under L, so hypothesis and penalty are consistent. A
    hypothesis with fewer than two downbeats has no spacing to penalise and
    is dropped, with the prior renormalised over the survivors, so that R
    identically zero gives a term of exactly zero. Returns None when no
    hypothesis survives."""
    log_priors, log_liks = [], []
    for h in meter_hyps:
        db_idx = torch.nonzero(h.phi_i == 0.0).flatten()
        if db_idx.numel() < 2:
            continue
        bar_lengths = torch.full((db_idx.numel() - 1,), float(h.L), device=matched_t.device)
        R_L = periodicity_R(matched_t, db_idx, bar_lengths)
        log_priors.append(torch.tensor(h.log_prior, device=matched_t.device))
        log_liks.append(-lambda_R * R_L)
    if not log_liks:
        return None
    log_priors = torch.stack(log_priors)
    log_joint = log_priors + torch.stack(log_liks)
    # -log sum_L pi(L) e^{-lambda R_L}, with pi renormalised over the survivors
    return -(torch.logsumexp(log_joint, dim=0) - torch.logsumexp(log_priors, dim=0))


def m_step_loss(hat_sigma: torch.Tensor, phi_i: torch.Tensor,
                 t_true: torch.Tensor, hat_t: torch.Tensor, hat_phi: torch.Tensor,
                 b_e: torch.Tensor, b_phi: torch.Tensor,
                 unmatched_agree=None, lambda_R: float = 0.0,
                 downbeat_idx: torch.Tensor | None = None,
                 bar_lengths: torch.Tensor | None = None,
                 meter_hyps: list | None = None) -> torch.Tensor:
    """Algorithm 5: MStep. Assembles eq. (totalloss)'s per-fragment summand:
    matched timing+phase terms (eps-insensitive, with AlignBeat's bounded
    log(2 eps + 2 b_e) normaliser; AlignBeat's Gamma precision prior is
    deliberately NOT carried over), L_agree over unmatched candidates (if
    given), and the periodicity regulariser of eq. (48).

    The regulariser is gated PER FRAGMENT by what the annotation provides,
    which is how the document specifies lambda_R ("for meter-unlabeled
    tracks, lambda_R is set to 0 for that fragment"):
      * fully-labeled fragment: pass downbeat_idx and bar_lengths, both read
        off the annotation (fragment_targets). R uses the annotated L.
      * beat-only fragment: pass meter_hyps from e_step. R is marginalised
        over the candidate meters (marginal_periodicity) -- the document
        would set lambda_R = 0 here; see that function for why this is a
        strictly milder choice than assuming a default meter.
      * neither given, or lambda_R == 0: no regulariser, eq. (22) exactly."""
    matched_t = hat_t[hat_sigma - 1]
    matched_phi = hat_phi[hat_sigma - 1]

    # AlignBeat's timing channel (criterion.py:213-225, _per_candidate_time_term)
    # with the per-fragment b_e standing in for its per-candidate b_j:
    # eps-insensitive residual and the bounded normaliser log(2 eps + 2 b_e)
    # per event. The M log(2 b_e) of eq. (totalloss) is the eps=0 case.
    residual = (t_true - matched_t).abs().sub(config.EPS_RESIDUAL).clamp(min=0.0)
    timing_term = time_term(residual, b_e).sum()
    # Wrapped-Laplace phase likelihood with the learned per-fragment scale
    # b_phi (appendix: lambda_phi := 1 / b_phi), same structure as the timing
    # term including its log-scale restoration; see phase_term.
    phase_nll = phase_term(circ_dist(phi_i, matched_phi), b_phi).sum()

    loss = timing_term + phase_nll

    if unmatched_agree is not None:
        _, agree_vals = unmatched_agree
        if agree_vals.numel() > 0:
            # L_agree is a bare circular distance (l_agree). It is the ONLY
            # training signal the unmatched candidates' phases get, and decode
            # relies on those phases sitting OFF the grid (interpolated between
            # the bracketing beats) to reject them. Weighted 1 against a phase
            # term at 1 / b_phi they were not trained at all: on a fitted batch
            # every unmatched candidate copied the neighbouring beat's phase and
            # decode emitted 2 candidates per beat. There is one phase
            # likelihood, so the unmatched residuals go through the same
            # wrapped-Laplace term, and b_phi is the MLE over all of them.
            loss = loss + phase_term(agree_vals, b_phi).sum()

    if lambda_R > 0:
        if downbeat_idx is not None and bar_lengths is not None and downbeat_idx.numel() >= 2:
            # Annotated meter available: eq. (48) as written.
            loss = loss + lambda_R * periodicity_R(matched_t, downbeat_idx, bar_lengths)
        elif meter_hyps is not None:
            # Meter latent: marginal over the candidate meters.
            R_marg = marginal_periodicity(matched_t, meter_hyps, lambda_R)
            if R_marg is not None:
                loss = loss + R_marg

    return loss


def train_step(model, x: torch.Tensor, t_true: torch.Tensor, ind: int,
                phi_true: torch.Tensor | None,
                downbeat_idx: torch.Tensor | None, bar_lengths: torch.Tensor | None,
                epoch: int, lambda_R: float = config.LAMBDA_R,
                meter_candidates: list[int] = config.METER_CANDIDATES,
                pi_M: list[float] = config.PI_M):
    """Algorithm 3: one fragment's training step. Runs the model forward,
    resolves (hat_sigma, phi_i) via the E-step, and returns the M-step loss
    -- ready for loss.backward() and an optimizer step by the caller.
    downbeat_idx / bar_lengths are the annotation's downbeats (ind=0 only);
    for ind=1 the meter is latent and comes from meter_candidates / pi_M."""
    hat_phi, hat_t, b_e, b_phi = model(x, epoch=epoch)
    hat_phi, hat_t, b_e, b_phi = hat_phi[0], hat_t[0], b_e[0], b_phi[0]  # batch=1 only

    hat_sigma, phi_i, unmatched, hyps = e_step(
        t_true, hat_t, hat_phi, b_e, phase_weight(b_phi), ind, phi_true=phi_true,
        meter_candidates=meter_candidates, pi_M=pi_M)
    loss = m_step_loss(hat_sigma, phi_i, t_true, hat_t, hat_phi, b_e, b_phi,
                        unmatched_agree=unmatched, lambda_R=lambda_R,
                        downbeat_idx=downbeat_idx, bar_lengths=bar_lengths,
                        meter_hyps=hyps)
    return loss


def training_loop(model, dataset, n_epochs: int = config.MAX_EPOCHS,
                   lr: float = config.LR, lambda_R: float = config.LAMBDA_R,
                   weight_decay: float = config.WEIGHT_DECAY,
                   warmup_steps: int = config.WARMUP_STEPS):
    """Algorithm 6: minibatch training loop, batch size 1 (see module
    docstring). dataset: list of (x, t_true, ind, phi_true_or_None,
    downbeat_idx_or_None, bar_lengths_or_None) tuples, one per fragment."""
    # AdamW + linear warmup then cosine decay, exactly AlignBeat's own optimizer
    # and CosineWarmupScheduler (pl_module.py:466, 506, 548).
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(n_epochs * len(dataset), 1)

    def lr_factor(step):
        step = step + 1
        warm = min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0
        return warm * 0.5 * (1 + math.cos(math.pi * min(step / total_steps, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    history = []
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        for x, t_true, ind, phi_true, downbeat_idx, bar_lengths in dataset:
            optimizer.zero_grad()
            loss = train_step(model, x, t_true, ind, phi_true, downbeat_idx, bar_lengths,
                              epoch, lambda_R)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(dataset)
        history.append(avg)
        print(f"epoch {epoch}: avg loss = {avg:.4f}")
    return history


def fit_ell(hat_phi: torch.Tensor, ell: int) -> torch.Tensor:
    """eq. (fitell): fit(ell) = sum_j min_k d_circ(hat_phi_j, k/ell)."""
    grid = torch.arange(ell, dtype=hat_phi.dtype, device=hat_phi.device) / ell  # (ell,)
    d = circ_dist(hat_phi[:, None], grid[None, :])         # (N, ell)
    return d.min(dim=1).values.sum()


def infer_meter(hat_phi: torch.Tensor, candidate_meters: list[int],
                 pi_M: torch.Tensor, lambda_phi: float) -> int:
    """Algorithm 9, lines 17-20: bias-corrected, prior-weighted meter
    inference. Returns hat_L."""
    N = hat_phi.shape[0]
    best_score, best_ell = float("inf"), candidate_meters[0]
    for idx, ell in enumerate(candidate_meters):
        score = (fit_ell(hat_phi, ell).item() - N / (4 * ell)
                 - (1.0 / lambda_phi) * math.log(float(pi_M[idx]) + 1e-12))
        if score < best_score:
            best_score, best_ell = score, ell
    return best_ell


def decode_threshold(tau: float, hat_L: int,
                     grid_relative: bool = config.TAU_GRID_RELATIVE) -> float:
    """The d_j threshold actually applied: tau / L when grid-relative (tau in
    beats), else the document's bare tau. See config.TAU_GRID_RELATIVE."""
    return tau / hat_L if grid_relative else tau


def merge_duplicates(p_hat: list, d: list, t_hat: list, keep: list) -> list:
    """One emission per grid slot per bar (config.DECODE_MERGE).

    keep: per-candidate acceptance flags from the tau test. Walk the accepted
    candidates in time order; when the next one carries the SAME p_hat as the
    last kept and follows it by less than half a beat period, the two are the
    same beat seen by neighbouring candidates -- keep whichever is nearer its
    grid point. The beat period is estimated from the accepted candidates
    themselves: the median time between consecutive accepted candidates whose
    p_hat advances by exactly one slot, which is what genuine consecutive
    beats do; if no such pair exists, the median gap of all accepted ones."""
    idx = [j for j in range(len(keep)) if keep[j]]
    if len(idx) < 3:
        return keep
    L = max(p_hat) + 1 if p_hat else 1
    advancing = [t_hat[b] - t_hat[a] for a, b in zip(idx, idx[1:])
                 if (p_hat[b] - p_hat[a]) % L == 1 and t_hat[b] > t_hat[a]]
    gaps = advancing if advancing else [t_hat[b] - t_hat[a] for a, b in zip(idx, idx[1:])]
    period = sorted(gaps)[len(gaps) // 2]
    out = [False] * len(keep)
    last = None
    for j in idx:
        if last is not None and p_hat[j] == p_hat[last] and t_hat[j] - t_hat[last] < 0.5 * period:
            if d[j] < d[last]:
                out[last], out[j], last = False, True, j
            continue
        out[j], last = True, j
    return out


def decode(hat_phi: torch.Tensor, hat_t: torch.Tensor, hat_L: int, tau: float,
           grid_relative: bool = config.TAU_GRID_RELATIVE,
           merge: bool = config.DECODE_MERGE):
    """Algorithm 9, lines 21-28: nearest-grid-point decoding given hat_L.
    Returns lists (p_hat, d, t_hat) for every candidate, and B = the
    accepted (p_hat, t_hat) pairs, in time order (guaranteed by hat_t's
    own strict monotonicity). Two departures from the document, both
    off-switchable and explained in config.py: the threshold is tau / L
    (TAU_GRID_RELATIVE) and near-duplicate emissions are merged
    (DECODE_MERGE)."""
    N = hat_phi.shape[0]
    p_hat = torch.round(hat_phi * hat_L).long() % hat_L
    grid_vals = p_hat.float() / hat_L
    d = circ_dist(hat_phi, grid_vals)
    thr = decode_threshold(tau, hat_L, grid_relative)
    p_l, d_l, t_l = p_hat.tolist(), d.tolist(), hat_t.tolist()
    keep = [d_l[j] <= thr for j in range(N)]
    if merge:
        keep = merge_duplicates(p_l, d_l, t_l, keep)
    B = [(p_l[j], t_l[j]) for j in range(N) if keep[j]]
    return p_l, d_l, t_l, B


def meter_consistency_correction(B: list, hat_L: int, p_hat: list, d: list, t_hat: list,
                                  tau: float, tau_prime: float):
    """Algorithm 10: gap-filling and pruning. tau and tau_prime are scaled the
    same way decode scales them (decode_threshold), so a downbeat that decode
    accepted is a downbeat here too."""
    tau = decode_threshold(tau, hat_L)
    tau_prime = decode_threshold(tau_prime, hat_L)
    downbeats = sorted([j for j in range(len(p_hat)) if p_hat[j] == 0 and d[j] <= tau], key=lambda j: t_hat[j])
    if len(B) < 2 or len(downbeats) < 2:
        return B
    t_min, t_max = min(t for _, t in B), max(t for _, t in B)
    mean_spacing = (t_max - t_min) / max(len(B) - 1, 1)
    s = hat_L * mean_spacing
    B_set = {(p_hat[j], t_hat[j]) for j in downbeats}
    corrected = list(B)
    for k in range(len(downbeats) - 1):
        j_k, j_k1 = downbeats[k], downbeats[k + 1]
        gap = t_hat[j_k1] - t_hat[j_k]
        if gap > 1.5 * s:
            candidates = [j for j in range(len(p_hat)) if t_hat[j_k] < t_hat[j] < t_hat[j_k1]
                          and (p_hat[j], t_hat[j]) not in B_set]
            if candidates:
                j_star = min(candidates, key=lambda j: abs(t_hat[j] - (t_hat[j_k] + s)))
                if p_hat[j_star] == 0 and d[j_star] <= tau_prime:
                    corrected.append((0, t_hat[j_star]))
        elif gap < 0.5 * s:
            j_weak = j_k if d[j_k] >= d[j_k1] else j_k1
            entry = (p_hat[j_weak], t_hat[j_weak])
            if entry in corrected and d[j_weak] > tau:
                corrected.remove(entry)
    return sorted(set(corrected), key=lambda pt: pt[1])


def inference(model, x: torch.Tensor,
              candidate_meters: list[int] = None, pi_M: torch.Tensor = None,
              tau: float = config.TAU, tau_prime: float = config.TAU_PRIME,
              lambda_phi: float | None = None):
    """Algorithm 9 + 10, full inference pipeline."""
    if candidate_meters is None:
        candidate_meters = config.METER_CANDIDATES
    if pi_M is None:
        pi_M = torch.tensor(config.PI_M)
    model.eval()
    with torch.no_grad():
        hat_phi, hat_t, _, b_phi = model(x)  # epoch=None -> the trained ScaleHeads
        hat_phi, hat_t = hat_phi[0], hat_t[0]
        # lambda_phi None -> the fragment's own 1 / b_phi, as in training
        hat_L = infer_meter(hat_phi, candidate_meters, pi_M,
                            phase_weight(b_phi[0], lambda_phi))
        p_hat, d, t_hat, B = decode(hat_phi, hat_t, hat_L, tau)
        B = meter_consistency_correction(B, hat_L, p_hat, d, t_hat, tau, tau_prime)
    return B, hat_L


if __name__ == "__main__":
    # Verify subset_select_dp against brute-force enumeration on a small case.
    torch.manual_seed(0)
    M, N = 3, 6
    t_true = torch.sort(torch.rand(M))[0]
    hat_t = torch.sort(torch.rand(N))[0]
    phi_true = torch.rand(M)
    hat_phi = torch.rand(N)
    b_e = torch.tensor(0.05)

    lam = 1.0 / config.B_PHI_0
    cost = match_cost_matrix(t_true, phi_true, hat_t, hat_phi, b_e, lambda_phi=lam, phase_blind=False)
    agree = l_agree(hat_phi, t_true, phi_true, hat_t)

    hat_sigma = subset_select_dp(cost, agree)
    print("DP result sigma:", hat_sigma.tolist())

    # Brute force: try every order-preserving injection sigma: {1..M} -> {1..N},
    # scoring each by eq. (22) -- matched costs PLUS L_agree for every candidate
    # left unmatched, leading ones included.
    from itertools import combinations

    def brute_force(cost, agree, M, N):
        best_cost, best_sigma = float("inf"), None
        for combo in combinations(range(1, N + 1), M):
            total = sum(cost[i, combo[i] - 1].item() for i in range(M))
            skipped = set(range(1, N + 1)) - set(combo)
            total += sum(agree[j - 1].item() for j in skipped)
            if total < best_cost:
                best_cost, best_sigma = total, combo
        return best_sigma

    print("Brute-force best sigma:", brute_force(cost, agree, M, N))
    print("Match:", tuple(hat_sigma.tolist()) == brute_force(cost, agree, M, N))

    # One seed is not a test: the leading-skip bug this checks for is invisible
    # whenever the optimal sigma happens to start at candidate 1, which seed 0
    # does. Sweep instead, and require EVERY case to agree.
    disagreements = 0
    for seed in range(200):
        torch.manual_seed(seed)
        m_, n_ = 3, 7
        tt = torch.sort(torch.rand(m_))[0]
        ht = torch.sort(torch.rand(n_))[0]
        pt, hp = torch.rand(m_), torch.rand(n_)
        c = match_cost_matrix(tt, pt, ht, hp, torch.tensor(0.05),
                              lambda_phi=lam, phase_blind=False)
        a = l_agree(hp, tt, pt, ht)
        if tuple(subset_select_dp(c, a).tolist()) != brute_force(c, a, m_, n_):
            disagreements += 1
    print(f"200-seed sweep vs brute force: {disagreements} disagreement(s)"
          f" {'-- OK' if disagreements == 0 else '-- FAIL'}")

    torch.manual_seed(0)   # the sweep above consumed the RNG; restore seed 0
    t_true = torch.sort(torch.rand(M))[0]
    hat_t = torch.sort(torch.rand(N))[0]
    phi_true = torch.rand(M)
    hat_phi = torch.rand(N)

    print("\n=== E-step / M-step, ind=0 (fully-labeled) ===")
    hat_t2 = hat_t.clone().requires_grad_(True)
    hat_phi2 = hat_phi.clone().requires_grad_(True)
    b_e2 = torch.tensor(0.05, requires_grad=True)
    b_phi2 = torch.tensor(config.B_PHI_0, requires_grad=True)
    hs, phi_i, unmatched, hyps0 = e_step(t_true, hat_t2, hat_phi2, b_e2, lambda_phi=lam, ind=0, phi_true=phi_true)
    print("hat_sigma:", hs.tolist(), " requires_grad:", hs.requires_grad if hasattr(hs, 'requires_grad') else False)
    print("meter hypotheses for ind=0 (should be None):", hyps0)
    loss = m_step_loss(hs, phi_i, t_true, hat_t2, hat_phi2, b_e2, b_phi2, unmatched_agree=unmatched)
    print("loss:", loss.item())
    loss.backward()
    print("grad flows to hat_t:", hat_t2.grad is not None and hat_t2.grad.abs().sum() > 0)
    print("grad flows to hat_phi:", hat_phi2.grad is not None and hat_phi2.grad.abs().sum() > 0)
    print("grad flows to b_e:", b_e2.grad is not None and b_e2.grad.abs().sum() > 0)
    print("grad flows to b_phi:", b_phi2.grad is not None and b_phi2.grad.abs().sum() > 0)

    print("\n=== E-step / M-step, ind=1 (beat-only, latent meter) ===")
    hat_t3 = torch.sort(torch.rand(N))[0].requires_grad_(True)
    hat_phi3 = torch.rand(N, requires_grad=True)
    b_e3 = torch.tensor(0.05, requires_grad=True)
    b_phi3 = torch.tensor(config.B_PHI_0, requires_grad=True)
    hs2, phi_i2, unmatched2, hyps = e_step(t_true, hat_t3, hat_phi3, b_e3, lambda_phi=lam, ind=1,
                                           meter_candidates=config.METER_CANDIDATES, pi_M=config.PI_M)
    print("hat_sigma:", hs2.tolist(), "phi_i (resolved, detached):", phi_i2.tolist())
    print("phi_i requires_grad (should be False):", phi_i2.requires_grad)
    print("meter hypotheses (L, score):", [(h.L, round(h.score, 3)) for h in hyps],
          "-> hat_L =", min(hyps, key=lambda h: h.score).L)
    loss2 = m_step_loss(hs2, phi_i2, t_true, hat_t3, hat_phi3, b_e3, b_phi3,
                         unmatched_agree=unmatched2, lambda_R=config.LAMBDA_R, meter_hyps=hyps)
    print("loss:", loss2.item())
    loss2.backward()
    print("grad flows to hat_t:", hat_t3.grad is not None and hat_t3.grad.abs().sum() > 0)
    print("grad flows to hat_phi:", hat_phi3.grad is not None and hat_phi3.grad.abs().sum() > 0)

    print("\n=== Wrapped-Laplace phase term: finite minimum in b_phi ===")
    dist = torch.full((48,), 0.03)
    for b in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        bt = torch.tensor(b, requires_grad=True); Lp = phase_term(dist, bt).sum(); Lp.backward()
        print(f"  b_phi={b:5.3f}  loss={Lp.item():8.2f}  dL/db={bt.grad.item():+9.1f}")
    print("  (residuals all 0.03: minimum must sit at b_phi ~ 0.03, gradient changes sign there)")

    print("\n=== Periodicity regulariser, eq. (48), on a synthetic 4/4 fragment ===")
    # 12 beats at a constant period; downbeats every 4 -> R must be 0 with the
    # annotated L=4 and grow as one downbeat is displaced.
    period = 1.0 / 13
    beats = torch.arange(1, 13, dtype=torch.float32) * period
    db_idx = torch.tensor([0, 4, 8])
    bars = torch.tensor([4.0, 4.0])
    print("R at the truth:", round(periodicity_R(beats, db_idx, bars).item(), 8), "(should be 0)")
    shifted = beats.clone(); shifted[4] += 0.5 * period
    print("R with one downbeat late by half a beat:", round(periodicity_R(shifted, db_idx, bars).item(), 8))
    fake_hyps = [MeterHypothesis(L=L, log_prior=math.log(p), phi_i=torch.tensor([((i) % L) / L for i in range(12)]),
                                 score=0.0) for L, p in zip(config.METER_CANDIDATES, config.PI_M)]
    Rm = marginal_periodicity(beats, fake_hyps, config.LAMBDA_R)
    print("marginal R on the same beats (every L fits a constant period equally, so ~0):", round(Rm.item(), 8))

    print("\n=== Full training loop, real hybrid model, synthetic data ===")
    from hybrid_beat_tracker import HybridBeatTracker

    torch.manual_seed(1)
    # AlignBeat's own configuration throughout (config.py): T = 1500 frames at
    # 50 fps, N_min = 170 -> S = 3, N = 188 candidates.
    spect_dim, T = config.SPECT_DIM, config.TRAIN_LENGTH
    model = HybridBeatTracker()

    def make_fragment(T, M, ind):
        x = torch.randn(1, T, spect_dim)
        t_true = torch.sort(torch.rand(M))[0]
        if ind == 0:
            L = 4
            phi_true = torch.tensor([((i) % L) / L for i in range(M)])
            downbeat_idx = torch.arange(0, M, L)
            bar_lengths = torch.full((downbeat_idx.numel() - 1,), float(L))
            return x, t_true, ind, phi_true, downbeat_idx, bar_lengths
        else:
            # beat-only: no meter of its own; e_step tries config.METER_CANDIDATES
            return x, t_true, ind, None, None, None

    # M = 60 events in 30 s is an ordinary 120 BPM; N = 188 leaves ample slack,
    # exactly the overgeneration the formulation expects.
    dataset = [make_fragment(T, 60, ind=0), make_fragment(T, 60, ind=1)]
    history = training_loop(model, dataset, n_epochs=3)
    print("Loss history:", [round(h, 4) for h in history])

    print("\n=== Inference, on the (briefly) trained model ===")
    x_test = torch.randn(1, T, spect_dim)
    B, hat_L = inference(model, x_test)   # meters, prior, tau, tau' all from config
    print(f"candidate meters {config.METER_CANDIDATES}, prior {config.PI_M}")
    print(f"hat_L = {hat_L}")
    print(f"B ({len(B)} events): {[(p, round(t, 4)) for p, t in B[:10]]}{'...' if len(B) > 10 else ''}")
    n_downbeats = sum(1 for p, _ in B if p == 0)
    print(f"downbeats decoded: {n_downbeats}")
