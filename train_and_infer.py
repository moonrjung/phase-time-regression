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
import torch

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
    timing = torch.abs(t_true[:, None] - hat_t[None, :]) / b_e  # (M, N)
    if phase_blind:
        return timing
    phase = lambda_phi * circ_dist(phi_true[:, None], hat_phi[None, :])  # (M, N)
    return timing + phase


def l_agree(hat_phi: torch.Tensor, t_true: torch.Tensor, phi_true: torch.Tensor,
            hat_t: torch.Tensor) -> torch.Tensor:
    """eq. (agree): for every candidate j, its agreement cost against the
    nearest bracketing ground-truth pair's own interpolated target phase.
    Only meaningful when phi_true is fully known (ind=0) -- see the
    document's own resolution of the ind=1 circularity concern.
    Returns an (N,) tensor, one value per candidate."""
    N = hat_t.shape[0]
    M = t_true.shape[0]
    out = torch.zeros(N, device=hat_phi.device, dtype=hat_phi.dtype)
    for j in range(N):
        tj = hat_t[j].item()
        # find the bracketing ground-truth pair (i, i+1) with t_i <= tj <= t_{i+1}
        i = torch.searchsorted(t_true, hat_t[j].detach()).item()
        i = max(1, min(i, M - 1))  # clamp into a valid bracket
        t_i, t_i1 = t_true[i - 1], t_true[i]
        phi_i, phi_i1 = phi_true[i - 1], phi_true[i]
        w = (tj - t_i.item()) / max(t_i1.item() - t_i.item(), 1e-8)
        target = (phi_i + w * (phi_i1 - phi_i)) % 1.0
        out[j] = circ_dist(hat_phi[j], target.detach())
    return out


def subset_select_dp(cost: torch.Tensor, agree: torch.Tensor | None = None):
    """Algorithm 2: SubsetSelectDP. cost: (M, N) match costs.
    agree: optional (N,) skip cost, folded in per eq. (dprecursion) -- only
    valid when phi_i is already known for every i (ind=0); pass None for
    the phase-blind (ind=1) search, per the document's own resolution.
    Returns hat_sigma as a length-M LongTensor of 1-indexed candidate
    positions (hat_sigma[i] = j means event i matched to candidate j)."""
    M, N = cost.shape
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
    """Section 2.4's hard phi_0 construction, single known L: for each of the
    L hypotheses p, score sum_i lambda_phi * d_circ((p+i-1)/L, hat_phi_sigma(i)),
    weighted by the learned prior pi_L. Returns (best p, per-event phi_i under
    that p) -- phi_i for event i is ((p + i - 1) % L) / L."""
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
    return best_p, phi_i


def e_step(t_true: torch.Tensor, hat_t: torch.Tensor, hat_phi: torch.Tensor,
           b_e: torch.Tensor, lambda_phi: float, ind: int,
           phi_true: torch.Tensor | None = None, L: int | None = None,
           pi_L: torch.Tensor | None = None):
    """Algorithm 4: EStep. ind=0 (fully-labeled): phi_true given directly,
    L_agree computable upfront and folded into the DP's skip cost. ind=1
    (beat-only, single known L): phase-blind search (timing alone), then
    hard phi_0 resolution given the now-fixed hat_sigma, then L_agree
    computed for the leftover unmatched candidates.
    Returns (hat_sigma, phi_i_for_matched_events, agree_for_unmatched)."""
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
        unmatched = [j for j in range(1, hat_t.shape[0] + 1)
                     if j not in hat_sigma.tolist()]
        if unmatched:
            unmatched_idx = torch.tensor(unmatched, dtype=torch.long) - 1
            agree_unmatched = agree[unmatched_idx]
        else:
            agree_unmatched = torch.zeros(0)
        return hat_sigma, phi_true.detach(), (unmatched, agree_unmatched)
    else:
        assert L is not None and pi_L is not None
        cost = match_cost_matrix(t_true, None, hat_t, hat_phi, b_e, lambda_phi, phase_blind=True)
        hat_sigma = subset_select_dp(cost)  # no skip cost yet -- phi unknown
        hat_sigma = hat_sigma.detach()
        _, phi_i = hard_phi0(hat_sigma, hat_phi.detach(), L, lambda_phi,
                             pi_L.to(hat_phi.device))
        unmatched = [j for j in range(1, hat_t.shape[0] + 1) if j not in hat_sigma.tolist()]
        if unmatched:
            unmatched_idx = torch.tensor(unmatched) - 1
            agree = l_agree(hat_phi[unmatched_idx], t_true, phi_i, hat_t[unmatched_idx])
        else:
            agree = torch.zeros(0)
        return hat_sigma, phi_i.detach(), (unmatched, agree)


def m_step_loss(hat_sigma: torch.Tensor, phi_i: torch.Tensor,
                 t_true: torch.Tensor, hat_t: torch.Tensor, hat_phi: torch.Tensor,
                 b_e: torch.Tensor, lambda_phi: float,
                 unmatched_agree=None, lambda_R: float = 0.0,
                 downbeat_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Algorithm 5: MStep. Assembles eq. (totalloss)'s per-fragment summand:
    matched timing+phase terms, the M log(2 b_e) scale-restoration term,
    L_agree over unmatched candidates (if given), and the periodicity
    regularizer (if lambda_R > 0 and >=2 downbeats matched)."""
    matched_t = hat_t[hat_sigma - 1]
    matched_phi = hat_phi[hat_sigma - 1]
    M = t_true.shape[0]

    timing_term = (torch.abs(t_true - matched_t) / b_e).sum()
    phase_term = lambda_phi * circ_dist(phi_i, matched_phi).sum()
    scale_restore = M * torch.log(2 * b_e)

    loss = timing_term + phase_term + scale_restore

    if unmatched_agree is not None:
        _, agree_vals = unmatched_agree
        if agree_vals.numel() > 0:
            loss = loss + agree_vals.sum()

    if lambda_R > 0 and downbeat_mask is not None and downbeat_mask.sum() >= 2:
        db_times = matched_t[downbeat_mask]
        spacings = db_times[1:] - db_times[:-1]
        mean_spacing = (matched_t[-1] - matched_t[0]) / max(matched_t.shape[0] - 1, 1)
        L_est = spacings.mean() / mean_spacing if mean_spacing > 0 else torch.tensor(1.0)
        expected = L_est * mean_spacing
        R = ((spacings - expected) ** 2).sum()
        loss = loss + lambda_R * R

    return loss


def train_step(model, x: torch.Tensor, t_true: torch.Tensor, ind: int,
                phi_true: torch.Tensor | None, L: int | None, pi_L: torch.Tensor | None,
                epoch: int, lambda_phi: float = config.LAMBDA_PHI,
                lambda_R: float = config.LAMBDA_R):
    """Algorithm 3: one fragment's training step. Runs the model forward,
    resolves (hat_sigma, phi_i) via the E-step, and returns the M-step loss
    -- ready for loss.backward() and an optimizer step by the caller."""
    hat_phi, hat_t, b_e = model(x, epoch=epoch)
    hat_phi, hat_t, b_e = hat_phi[0], hat_t[0], b_e[0]  # drop the batch dim (batch=1 only)

    hat_sigma, phi_i, unmatched = e_step(t_true, hat_t, hat_phi, b_e, lambda_phi, ind,
                                          phi_true=phi_true, L=L, pi_L=pi_L)
    downbeat_mask = (phi_i == 0.0) if ind == 1 else (phi_true == 0.0)
    loss = m_step_loss(hat_sigma, phi_i, t_true, hat_t, hat_phi, b_e, lambda_phi,
                        unmatched_agree=unmatched, lambda_R=lambda_R, downbeat_mask=downbeat_mask)
    return loss


def training_loop(model, dataset, n_epochs: int = config.MAX_EPOCHS,
                   lr: float = config.LR, lambda_phi: float = config.LAMBDA_PHI,
                   lambda_R: float = config.LAMBDA_R,
                   weight_decay: float = config.WEIGHT_DECAY,
                   warmup_steps: int = config.WARMUP_STEPS):
    """Algorithm 6: minibatch training loop, batch size 1 (see module
    docstring). dataset: list of (x, t_true, ind, phi_true_or_None, L_or_None,
    pi_L_or_None) tuples, one per fragment."""
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
        for x, t_true, ind, phi_true, L, pi_L in dataset:
            optimizer.zero_grad()
            loss = train_step(model, x, t_true, ind, phi_true, L, pi_L, epoch, lambda_phi, lambda_R)
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
    grid = torch.arange(ell, dtype=torch.float32) / ell  # (ell,)
    d = circ_dist(hat_phi[:, None], grid[None, :])         # (N, ell)
    return d.min(dim=1).values.sum()


def infer_meter(hat_phi: torch.Tensor, candidate_meters: list[int],
                 pi_M: torch.Tensor, lambda_phi: float) -> int:
    """Algorithm 9, lines 17-20: bias-corrected, prior-weighted meter
    inference. Returns hat_L."""
    N = hat_phi.shape[0]
    best_score, best_ell = float("inf"), candidate_meters[0]
    for idx, ell in enumerate(candidate_meters):
        score = fit_ell(hat_phi, ell).item() - N / (4 * ell) - (1.0 / lambda_phi) * math.log(pi_M[idx].item() + 1e-12)
        if score < best_score:
            best_score, best_ell = score, ell
    return best_ell


def decode(hat_phi: torch.Tensor, hat_t: torch.Tensor, hat_L: int, tau: float):
    """Algorithm 9, lines 21-28: nearest-grid-point decoding given hat_L.
    Returns lists (p_hat, d, t_hat) for every candidate, and B = the
    accepted (p_hat, t_hat) pairs, in time order (guaranteed by hat_t's
    own strict monotonicity)."""
    N = hat_phi.shape[0]
    p_hat = torch.round(hat_phi * hat_L).long() % hat_L
    grid_vals = p_hat.float() / hat_L
    d = circ_dist(hat_phi, grid_vals)
    B = [(p_hat[j].item(), hat_t[j].item()) for j in range(N) if d[j].item() <= tau]
    return p_hat.tolist(), d.tolist(), hat_t.tolist(), B


def meter_consistency_correction(B: list, hat_L: int, p_hat: list, d: list, t_hat: list,
                                  tau: float, tau_prime: float):
    """Algorithm 10: gap-filling and pruning."""
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
              lambda_phi: float = config.LAMBDA_PHI):
    """Algorithm 9 + 10, full inference pipeline."""
    if candidate_meters is None:
        candidate_meters = config.METER_CANDIDATES
    if pi_M is None:
        pi_M = torch.tensor(config.PI_M)
    model.eval()
    with torch.no_grad():
        hat_phi, hat_t, _ = model(x)  # epoch=None -> always the trained ScaleHead
        hat_phi, hat_t = hat_phi[0], hat_t[0]
        hat_L = infer_meter(hat_phi, candidate_meters, pi_M, lambda_phi)
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

    cost = match_cost_matrix(t_true, phi_true, hat_t, hat_phi, b_e, lambda_phi=config.LAMBDA_PHI, phase_blind=False)
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
                              lambda_phi=config.LAMBDA_PHI, phase_blind=False)
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
    hs, phi_i, unmatched = e_step(t_true, hat_t2, hat_phi2, b_e2, lambda_phi=config.LAMBDA_PHI, ind=0, phi_true=phi_true)
    print("hat_sigma:", hs.tolist(), " requires_grad:", hs.requires_grad if hasattr(hs, 'requires_grad') else False)
    loss = m_step_loss(hs, phi_i, t_true, hat_t2, hat_phi2, b_e2, lambda_phi=config.LAMBDA_PHI, unmatched_agree=unmatched)
    print("loss:", loss.item())
    loss.backward()
    print("grad flows to hat_t:", hat_t2.grad is not None and hat_t2.grad.abs().sum() > 0)
    print("grad flows to hat_phi:", hat_phi2.grad is not None and hat_phi2.grad.abs().sum() > 0)
    print("grad flows to b_e:", b_e2.grad is not None and b_e2.grad.abs().sum() > 0)

    print("\n=== E-step / M-step, ind=1 (beat-only, single known L) ===")
    L = 4
    pi_L = torch.full((L,), 1.0 / L)
    hat_t3 = torch.sort(torch.rand(N))[0].requires_grad_(True)
    hat_phi3 = torch.rand(N, requires_grad=True)
    b_e3 = torch.tensor(0.05, requires_grad=True)
    hs2, phi_i2, unmatched2 = e_step(t_true, hat_t3, hat_phi3, b_e3, lambda_phi=config.LAMBDA_PHI, ind=1, L=L, pi_L=pi_L)
    print("hat_sigma:", hs2.tolist(), "phi_i (resolved, detached):", phi_i2.tolist())
    print("phi_i requires_grad (should be False):", phi_i2.requires_grad)
    downbeat_mask = (phi_i2 == 0.0)
    loss2 = m_step_loss(hs2, phi_i2, t_true, hat_t3, hat_phi3, b_e3, lambda_phi=config.LAMBDA_PHI,
                         unmatched_agree=unmatched2, lambda_R=0.1, downbeat_mask=downbeat_mask)
    print("loss:", loss2.item())
    loss2.backward()
    print("grad flows to hat_t:", hat_t3.grad is not None and hat_t3.grad.abs().sum() > 0)
    print("grad flows to hat_phi:", hat_phi3.grad is not None and hat_phi3.grad.abs().sum() > 0)

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
            return x, t_true, ind, phi_true, None, None
        else:
            L = 4
            pi_L = torch.full((L,), 1.0 / L)
            return x, t_true, ind, None, L, pi_L

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
