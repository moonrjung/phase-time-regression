"""
PhaseHead and RegHead: the two per-candidate heads sitting on top of the
transformer encoder's downsampled candidate features z_1, ..., z_N.

Faithful to the document's own definitions:
  - PhaseHead:  eq. (phasehead) -- atan2 construction, no meter needed at all.
  - RegHead:    eq. (monotone)  -- cumulative-softplus reparameterization,
                a *global* normalization over all N candidates together,
                not an independent per-candidate output.
  - CandidateSelfAttention: eq. (candattn) -- the phase branch alone passes
    through self-attention before its head; RegHead reads z_j directly.
  - DownsampleStep / schedule: eq. (scheduleS), eq. (zeropad),
    eq. (gradualdownsample) -- reduces the encoder's raw output, length T,
    down to N candidate slots via S progressive halving steps.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class SpectrogramEncoder(nn.Module):
    """The part that was missing: h_t^(0) = W_emb x_t + e_t (linear embedding
    + positional encoding), then h^(l) = Block_l(h^(l-1); theta) for
    l=1,...,L_enc (Algorithm 3, lines 1-6) -- a standard transformer encoder
    stack, producing h^(L_enc), the RAW encoder output CandidateHeads
    actually starts from."""

    def __init__(self, n_mel: int, d_model: int, L_enc: int, n_heads: int = 8,
                 dim_feedforward: int = 2048, max_len: int = 4096):
        super().__init__()
        self.W_emb = nn.Linear(n_mel, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_enc, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            activation="gelu", batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=L_enc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, n_mel) -- raw log-mel spectrogram frames
        T = x.shape[1]
        h0 = self.W_emb(x) + self.pos_enc[:, :T, :]  # linear embed + positional encoding
        return self.blocks(h0)                        # h^(L_enc): (batch, T, d_model)


def downsample_schedule(T: int, N_min: int):
    """eq. (scheduleS): S = max{s : ceil(T/2^s) >= N_min}; N = ceil(T/2^S);
    T' = N * 2^S (the zero-padded length DownsampleStep operates on)."""
    s = 0
    while math.ceil(T / 2 ** (s + 1)) >= N_min:
        s += 1
    S = s
    N = math.ceil(T / 2 ** S)
    T_prime = N * 2 ** S
    return S, N, T_prime


class DownsampleStep(nn.Module):
    """One halving step: pairs adjacent positions and projects back to
    d_model -- a learned merge, not lossy average-pooling, so the model
    decides how to combine each pair rather than always averaging them."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        # g: (batch, T_prev, d_model), T_prev even -> (batch, T_prev/2, d_model)
        batch, T_prev, d = g.shape
        paired = g.reshape(batch, T_prev // 2, 2 * d)
        return self.norm(self.proj(paired))


class EncoderToCandidates(nn.Module):
    """Reduces the encoder's raw output h^(L_enc), length T, down to N
    candidate features z_1,...,z_N via S progressive DownsampleStep calls,
    with zero-padding to T' first (eq. zeropad) whenever T isn't already
    a multiple of 2^S."""

    def __init__(self, d_model: int, N_min: int):
        super().__init__()
        self.d_model = d_model
        self.N_min = N_min
        # One DownsampleStep per halving level; S depends on T at call time,
        # so we lazily build/cache steps keyed by S the first time we see it.
        self.steps = nn.ModuleDict()

    def _get_steps(self, S: int) -> nn.ModuleList:
        key = str(S)
        if key not in self.steps:
            self.steps[key] = nn.ModuleList(
                [DownsampleStep(self.d_model) for _ in range(S)]
            )
        return self.steps[key]

    def forward(self, h: torch.Tensor):
        # h: (batch, T, d_model) -- the encoder block's own raw output
        batch, T, d = h.shape
        S, N, T_prime = downsample_schedule(T, self.N_min)

        pad_len = T_prime - T
        g = F.pad(h, (0, 0, 0, pad_len)) if pad_len > 0 else h  # eq. (zeropad)

        for step in self._get_steps(S):
            g = step(g)  # halves length each call

        assert g.shape[1] == N, f"expected N={N} candidates, got {g.shape[1]}"
        return g, N  # z_1, ..., z_N and the resulting candidate count



class CandidateSelfAttention(nn.Module):
    """Phase-branch-only self-attention over the N candidate features.
    RegHead does NOT go through this -- it reads z_j directly (Section 2.2).

    Block configuration follows AlignBeat's own candidate attention
    (alignbeat/head.py:44-57): pre-LN encoder layers, feed-forward width
    2*d_model, no dropout, and the optional final LayerNorm a pre-LN stack
    otherwise lacks -- off by default there, so off by default here."""

    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 1,
                 ff_mult: int = 2, dropout: float = 0.0,
                 final_norm: bool = False):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ff_mult, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            layer, n_layers,
            norm=nn.LayerNorm(d_model) if final_norm else None)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, N, d_model) -> tilde_z: (batch, N, d_model)
        return self.encoder(z)


class PhaseHead(nn.Module):
    """Outputs (u_j, v_j) in R^2, then hat_phi_j via atan2 -- always in [0,1),
    no meter L needed at any point (this is the whole point of the [0,1)
    convention: PhaseHead's own construction is meter-independent)."""

    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),  # raw (u, v)
        )

    def forward(self, tilde_z: torch.Tensor) -> torch.Tensor:
        # tilde_z: (batch, N, d_model) -> hat_phi: (batch, N), in [0, 1)
        uv = self.net(tilde_z)                      # (batch, N, 2)
        u, v = uv[..., 0], uv[..., 1]
        angle = torch.atan2(v, u)                    # in (-pi, pi]
        hat_phi = (angle % (2 * torch.pi)) / (2 * torch.pi)  # eq. (phasehead)
        return hat_phi


class RegHead(nn.Module):
    """Outputs a raw per-candidate score r_j; the actual monotonic time
    estimate hat_t_j is a SEPARATE, global reparameterization over all N
    scores together (eq. monotone) -- not produced by this module alone."""

    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, N, d_model) -> r: (batch, N)
        return self.net(z).squeeze(-1)


class ScaleHead(nn.Module):
    """eq. (scalehead): b_e = softplus(ScaleHead(z_bar; theta)) + b_min --
    the LEARNED Laplace timing scale, warm-started at a fixed b_0 for the
    first E_0 epochs (Algorithm 3, lines 13-17), so training the scale
    itself does not begin until b_0's own warm-start phase is over.

    Genuinely per-FRAGMENT, not per-candidate: reads z_bar, the mean-pooled
    candidate features (eq. meterpool), a single (batch, d_model) summary --
    NOT the per-candidate z_j -- since timing spread is a property of the
    whole fragment, not of any one candidate."""

    def __init__(self, d_model: int, hidden: int = 128, b_min: float = 1e-3,
                 b_init: float = config.B_0):
        super().__init__()
        self.b_min = b_min
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Start the head AT b_0, so eq. (51)'s hand-over from the fixed b_0 to
        # the learned b_e is continuous. Without this the untrained head emits
        # softplus(random) + b_min, measured at 18-23 SECONDS on real data
        # (the sweep's lambda_phi=30 runs never recovered from that jump; the
        # loss rose from ~200 to ~450 at the hand-over epoch and stayed there).
        # Zero last-layer weights and a bias of softplus^-1(b_0 - b_min) give
        # exactly b_0 for every input; the weights then learn deviations.
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        target = max(b_init - b_min, 1e-6)
        nn.init.constant_(last.bias, math.log(math.expm1(target)))

    def forward(self, z_bar: torch.Tensor) -> torch.Tensor:
        # z_bar: (batch, d_model) -> b_e: (batch,), always > b_min
        raw = self.net(z_bar).squeeze(-1)
        return F.softplus(raw) + self.b_min


def mean_pool_candidates(z: torch.Tensor) -> torch.Tensor:
    """eq. (meterpool): z_bar = (1/N) sum_j z_j -- pooled once, shared by
    ScaleHead (this module) and, in the document's earlier meter-classifier
    draft, MeterHead -- now unused for meter (Section 5.1's own point:
    meter is inferred from fit(ell) directly, no pooled summary needed
    there any more), kept here solely for ScaleHead's own use."""
    return z.mean(dim=1)  # (batch, N, d_model) -> (batch, d_model)


def scale_with_warmup(z_bar: torch.Tensor, scale_head: ScaleHead,
                       epoch: int, warmup_epochs: int, b_0: float) -> torch.Tensor:
    """Algorithm 3, lines 13-17: b_e = b_0 (fixed, no gradient path) for
    epoch <= E_0; b_e = ScaleHead(z_bar) once epoch > E_0. Returns a
    (batch,) tensor either way, so callers never need to branch on shape."""
    if epoch <= warmup_epochs:
        return torch.full((z_bar.shape[0],), b_0, device=z_bar.device, dtype=z_bar.dtype)
    return scale_head(z_bar)


def monotonic_time_reparam(r: torch.Tensor) -> torch.Tensor:
    """eq. (monotone): hat_t_j = (sum_{k<=j} softplus(r_k)) / (sum_k softplus(r_k)).
    A global cumulative normalization over the WHOLE candidate sequence --
    guarantees hat_t_1 < ... < hat_t_N strictly, by construction."""
    # r: (batch, N) -> hat_t: (batch, N), strictly increasing along dim=-1
    sp = F.softplus(r)                    # (batch, N)
    cum = torch.cumsum(sp, dim=-1)        # (batch, N)
    Z = cum[..., -1:]                     # (batch, 1), total sum
    return cum / Z


def local_time_reparam(r: torch.Tensor) -> torch.Tensor:
    """config.TIME_REPARAM = "local": hat_t_j = (j + 0.5 + 0.5 tanh(r_j)) / N.
    Candidate j owns the slot (j/N, (j+1)/N) and moves within it, so the
    sequence is strictly increasing without any cumulative coupling between
    candidates -- see config.py for why eq. (monotone)'s cumsum did not train."""
    N = r.shape[-1]
    j = torch.arange(N, device=r.device, dtype=r.dtype)
    return (j + 0.5 + 0.5 * torch.tanh(r)) / N


def time_reparam(r: torch.Tensor, kind: str = config.TIME_REPARAM) -> torch.Tensor:
    if kind == "monotone":
        return monotonic_time_reparam(r)
    if kind == "local":
        return local_time_reparam(r)
    raise ValueError(f"unknown time reparameterisation {kind!r}; 'monotone' or 'local'")


class SharedProjection(nn.Module):
    """One shared bottleneck, feeding BOTH heads -- not a per-head reduction.
    Sits right after downsampling, before the branch splits into phase-attention
    and regression, so encoder + downsampling keep their full representational
    capacity (d_model=512), and only the two heads' shared input is reduced.
    An explicit design choice beyond what the document specifies, not a fix
    to something the document required."""

    def __init__(self, d_model: int, reduced_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_model, reduced_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(reduced_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, N, d_model) -> (batch, N, reduced_dim)
        return self.norm(self.act(self.proj(z)))


class CandidateHeads(nn.Module):
    """Top-level module: takes the encoder block's RAW output h (batch, T,
    d_model) -- e.g. (B, 1504, 512) -- and returns (hat_phi_j, hat_t_j, b_e)
    for every one of the resulting N candidates (b_e is per-fragment, shared
    across all N). Wires together downsampling, ONE shared bottleneck
    projection, the phase-branch self-attention, and all three heads --
    everything Algorithm 3 (training) and Algorithm 9 (inference) need,
    identical in both, per Proposition 1 (no sigma dependence anywhere in
    this module)."""

    def __init__(self, d_model: int, N_min: int = config.N_MIN,
                 reduced_dim: int = config.REDUCED_DIM,
                 phase_hidden: int = config.HEAD_HIDDEN,
                 reg_hidden: int = config.HEAD_HIDDEN,
                 scale_hidden: int = config.HEAD_HIDDEN,
                 b_0: float = config.B_0,
                 warmup_epochs: int = config.WARMUP_EPOCHS):
        super().__init__()
        self.downsample = EncoderToCandidates(d_model, N_min)
        self.shared_proj = SharedProjection(d_model, reduced_dim)
        self.candidate_attn = CandidateSelfAttention(reduced_dim)
        self.phase_head = PhaseHead(reduced_dim, phase_hidden)
        self.reg_head = RegHead(reduced_dim, reg_hidden)
        self.scale_head = ScaleHead(reduced_dim, scale_hidden)
        self.b_0 = b_0
        self.warmup_epochs = warmup_epochs

    def forward(self, h: torch.Tensor, epoch: int | None = None):
        # h: (batch, T, d_model) -- the encoder's own raw output, e.g. (B, 1504, 512)
        # epoch: current training epoch, for warm-start (Algorithm 3, lines 13-17);
        #        pass None at inference, where the trained ScaleHead is always used.
        z, N = self.downsample(h)                 # (batch, N, d_model)
        z = self.shared_proj(z)                    # (batch, N, reduced_dim) -- shared bottleneck
        tilde_z = self.candidate_attn(z)           # phase branch only
        hat_phi = self.phase_head(tilde_z)         # (batch, N), in [0, 1)
        r = self.reg_head(z)                       # (batch, N), raw scores
        hat_t = monotonic_time_reparam(r)          # (batch, N), strictly increasing

        z_bar = mean_pool_candidates(z)             # (batch, reduced_dim), eq. (meterpool)
        if epoch is None:
            b_e = self.scale_head(z_bar)            # inference: always the trained head
        else:
            b_e = scale_with_warmup(z_bar, self.scale_head, epoch, self.warmup_epochs, self.b_0)

        return hat_phi, hat_t, b_e


class BeatTracker(nn.Module):
    """The full pipeline, spectrogram in: SpectrogramEncoder -> CandidateHeads.
    This is what actually corresponds to f_theta(x) throughout the document --
    the previous CandidateHeads alone started from an already-given encoder
    output, never from x itself."""

    def __init__(self, n_mel: int = config.SPECT_DIM,
                 d_model: int = config.TRANSFORMER_DIM,
                 L_enc: int = config.N_LAYERS,
                 N_min: int = config.N_MIN,
                 reduced_dim: int = config.REDUCED_DIM):
        super().__init__()
        self.encoder = SpectrogramEncoder(n_mel, d_model, L_enc)
        self.heads = CandidateHeads(d_model, N_min, reduced_dim=reduced_dim)

    def forward(self, x: torch.Tensor, epoch: int | None = None):
        # x: (batch, T, n_mel) -- raw log-mel spectrogram
        h = self.encoder(x)          # (batch, T, d_model)
        return self.heads(h, epoch)  # (hat_phi, hat_t, b_e)


if __name__ == "__main__":
    # End-to-end sanity check, starting from the actual spectrogram input --
    # not from an already-given encoder output. batch and L_enc kept modest
    # here purely because this sandbox has limited memory; T=1504, d_model=512
    # (the shapes actually asked for) are exact and unchanged -- verified
    # separately that batch=1, L_enc=1 succeeds at this full T, d_model; a
    # real training machine should handle batch=4+, L_enc=6+ without issue.
    torch.manual_seed(0)
    batch, T, n_mel, d_model, L_enc = 1, 1504, 80, 512, 2
    N_min = 32

    x = torch.randn(batch, T, n_mel)

    model = BeatTracker(n_mel, d_model, L_enc, N_min)

    # Training-time call, epoch <= warmup: b_e should be exactly b_0, no grad path to ScaleHead.
    hat_phi, hat_t, b_e_warmup = model(x, epoch=1)
    print(f"warm-start (epoch=1): b_e == b_0? {torch.allclose(b_e_warmup, torch.full_like(b_e_warmup, model.heads.b_0))}")

    # Training-time call, epoch > warmup: b_e comes from the trained ScaleHead.
    hat_phi, hat_t, b_e_trained = model(x, epoch=10)
    print(f"post-warmup (epoch=10): b_e = {b_e_trained.item():.4f} (from ScaleHead, not b_0)")

    # Inference-time call: no epoch argument, always the trained head.
    hat_phi, hat_t, b_e_infer = model(x)
    print(f"inference (no epoch): b_e = {b_e_infer.item():.4f}")
    print(f"inference matches post-warmup training call? {torch.allclose(b_e_infer, b_e_trained)}")

    S, N, T_prime = downsample_schedule(T, N_min)
    print(f"T={T}, N_min={N_min} -> S={S}, N={N}, T'={T_prime} (padding={T_prime - T})")
    print("hat_phi shape:", hat_phi.shape)
    print("hat_t shape:  ", hat_t.shape)
    print("hat_phi range: [{:.4f}, {:.4f}]".format(hat_phi.min().item(), hat_phi.max().item()))
    print("hat_t range:   [{:.4f}, {:.4f}]".format(hat_t.min().item(), hat_t.max().item()))

    diffs = hat_t[:, 1:] - hat_t[:, :-1]
    print("hat_t strictly increasing everywhere:", bool((diffs > 0).all()))

    loss = hat_phi.sum() + hat_t.sum() + b_e_infer.sum()  # include b_e to confirm ScaleHead trains too
    loss.backward()
    n_total = sum(1 for _ in model.parameters())
    n_grad = sum(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"Parameters receiving gradient: {n_grad} / {n_total}")