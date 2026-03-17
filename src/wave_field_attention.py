"""
Wave Field Attention V4.7
=========================

Physics-based O(n log n) attention via damped wave kernels on continuous fields.

Each head has 3 learnable wave parameters (frequency, damping, phase).
Tokens scatter onto a 1D field, wave kernels convolve via FFT, results gather back.

Key components:
  - LearnedFeatureMap: identity-init MLP + ReLU (Hedgehog, ICLR 2024)
  - SpectralGate: MLP conditioned on token 0 modulates kernel FFT per-sample
    (SPECTRE, arXiv:2502.18394). Makes kernel input-dependent.
  - HiPPO init: harmonic frequencies ω_n = π(2n+1)/2 (S4D, arXiv:2206.11893)

Pipeline:
  1. DEPOSIT: φ_k(K) ⊙ V → scatter onto field
  2. PROPAGATE: FFT convolution with (optionally content-adaptive) wave kernel
  3. COUPLE: cross-head field coupling
  4. GATHER: read field at token positions
  5. READ: φ_q(Q) ⊙ gathered → cross-dim mixing → gating → output

Complexity: O(n log n) per layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os


def _next_fast_size(n):
    """Find smallest m >= n whose prime factors are all <= 7 (cuFFT sweet spot).

    cuFFT has optimized radix-2/3/5/7 kernels. Sizes with larger prime factors
    fall back to slower Bluestein/Rader algorithms. Padding to a "fast" size
    avoids 2-5x slowdowns for unlucky field sizes.
    """
    while True:
        m = n
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


class LearnedFeatureMap(nn.Module):
    """Learned positive feature map for linear attention (Hedgehog, ICLR 2024).

    MLP of `depth` identity-initialized Linear(d,d) + activation layers.
    At init: phi(x) ≈ ReLU(Ix) + eps — 50% sparse, rank d/2.
    During training: learns spiky, dot-product-monotonic maps that mimic softmax.

    V4.3.8: Intermediate layers use GELU (no dead neurons, smooth gradients).
    Final layer keeps ReLU for positivity guarantee (linear attention needs φ>0).
    V4.3.7 had 59% dead Q neurons from ReLU killing neurons permanently.
    GELU intermediates allow gradient flow even for negative inputs, while
    final ReLU still ensures positivity. Dead neurons can recover because
    upstream GELU gradients update the first Linear, shifting inputs positive.

    depth=1: original V4.3 (single linear).
    depth=2: Hedgehog-style (closes 68.6% of linear-vs-softmax gap at 125M scale).
    """

    def __init__(self, dim, eps=1e-6, depth=2):
        super().__init__()
        self.eps = eps
        layers = []
        for i in range(depth):
            lin = nn.Linear(dim, dim, bias=True)
            with torch.no_grad():
                nn.init.eye_(lin.weight)
                nn.init.zeros_(lin.bias)
            layers.append(lin)
            if i < depth - 1:
                layers.append(nn.GELU())   # intermediate: no dead neurons
            else:
                layers.append(nn.ReLU())   # final: ensures positivity for linear attn
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) + self.eps


class SpectralGate(nn.Module):
    """Content-adaptive spectral gate (SPECTRE, arXiv:2502.18394).

    Conditioned on token 0's query vector (the only token visible to all
    positions in a causal model). Produces per-sample spectral modulation
    of the wave kernel FFT, making the attention pattern input-dependent
    while staying O(n log n).

    Output is 3D (B, H, freq_bins) — requires causal enforcement via
    IFFT→zero→FFT since modulation can break Kramers-Kronig.

    At init, MLP output ≈ 0, so modulated ≈ base kernel (safe start).
    """

    def __init__(self, num_heads, head_dim, freq_bins, n_control=32):
        super().__init__()
        self.num_heads = num_heads
        self.freq_bins = freq_bins
        self.n_control = n_control

        input_dim = num_heads * head_dim

        self.norm = nn.LayerNorm(head_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, num_heads * n_control),
        )

        # V4.3.4: 5x larger init (was 0.01). Old scale + weight decay caused
        # gate to decay to near-zero (range < 0.07 after 20M tokens).
        with torch.no_grad():
            self.net[-1].weight.mul_(0.05)
            self.net[-1].bias.zero_()

    def forward(self, q, base_kernel_fft):
        """
        q: (B, H, N, head_dim)
        base_kernel_fft: (H, freq_bins) complex
        Returns: (B, H, freq_bins) complex — modulated kernel per batch element
        """
        B, H, N, d = q.shape

        # V4.3.4 FIX: Use first token only (position 0). mean(Q) over all positions
        # LEAKS FUTURE — kernel shape depends on future Q vectors, giving ~6 logit
        # difference when changing last token. _enforce_causal_kernel only makes the
        # impulse response causal, NOT the kernel construction. With FFT convolution,
        # one kernel applies to all positions, so it can only depend on tokens all
        # positions can see = token 0.
        q_bar = self.norm(q[:, :, 0, :])           # (B, H, d)
        q_flat = q_bar.reshape(B, H * d)          # (B, H*d)

        # MLP → spectral control points
        ctrl = self.net(q_flat)                    # (B, H*n_control)
        ctrl = ctrl.view(B, H, self.n_control)     # (B, H, n_control)

        # Interpolate to full frequency resolution (smooth, low-rank gate)
        gate = F.interpolate(
            ctrl, size=self.freq_bins, mode='linear', align_corners=True
        ).float()                                  # (B, H, freq_bins), float32 for complex mul

        # Modulate: at init gate≈0, so output ≈ base_kernel_fft
        return base_kernel_fft.unsqueeze(0) * (1.0 + gate)


class DeltaCorrection(nn.Module):
    """Additive delta-rule correction for wave field attention (RLA-inspired).

    Tracks prediction errors in a (d, d) recurrent state per head and outputs
    a correction signal. The wave path stays untouched — this ADDS to it.

    Delta rule: before writing k→v, erase old v associated with k.
      error_t = v_t - S_{t-1} @ k_t       (what we got wrong)
      S_t = α_t * S_{t-1} + β_t * v_t @ k_t^T - β_t * (S_{t-1} @ k_t) @ k_t^T
      correction_t = S_t @ q_t

    Cost: O(n * d²) per layer. At d=48, N=512: ~1.2M FLOPs (0.1% of FFN).
    Params: ~200 per layer (2 gate projections).

    References:
      - GatedDeltaNet (ICLR 2025): 0.97x gap at 1.3B
      - RLA (arXiv:2509.25223): matches transformer at 1.5B
    """

    def __init__(self, head_dim, num_heads):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        # Per-token gates from key content (shared across heads)
        # Decay: how much of old state to keep. Init sigmoid(-2)=0.12 → mostly forget
        self.decay_proj = nn.Linear(head_dim, 1, bias=True)
        nn.init.zeros_(self.decay_proj.weight)
        nn.init.constant_(self.decay_proj.bias, -2.0)
        # Write strength. Init sigmoid(-1)=0.27 → gentle writes
        self.write_proj = nn.Linear(head_dim, 1, bias=True)
        nn.init.zeros_(self.write_proj.weight)
        nn.init.constant_(self.write_proj.bias, -1.0)
        # Output scale — start near zero so correction doesn't disrupt wave path at init
        self.out_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, q, k, v):
        """
        q, k, v: (B, H, N, d) — reuses projections from wave path
        Returns: (B, H, N, d) — additive correction

        Chunk-parallel: within each chunk (C=64 tokens), use causal linear
        attention (fully parallel, O(C²d)). Between chunks (N/C=8 steps),
        propagate (d,d) state sequentially. Total: 8 sequential steps
        instead of 512.
        """
        B, H, N, d = q.shape
        C = 64  # chunk size

        # Content-dependent gates
        beta = torch.sigmoid(self.write_proj(k))    # (B, H, N, 1) write strength
        alpha = torch.sigmoid(self.decay_proj(k))   # (B, H, N, 1) decay

        # Normalize k for stable state
        k_norm = F.normalize(k, dim=-1)

        # Pad to chunk boundary
        pad = (C - N % C) % C
        if pad > 0:
            q = F.pad(q, (0, 0, 0, pad))
            k_norm = F.pad(k_norm, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
            beta = F.pad(beta, (0, 0, 0, pad))
            alpha = F.pad(alpha, (0, 0, 0, pad), value=1.0)
        Np = N + pad
        n_chunks = Np // C

        # Reshape to chunks: (B, H, n_chunks, C, d/1)
        q_c = q.reshape(B, H, n_chunks, C, d)
        k_c = k_norm.reshape(B, H, n_chunks, C, d)
        v_c = v.reshape(B, H, n_chunks, C, d)
        b_c = beta.reshape(B, H, n_chunks, C, 1)

        # Per-chunk causal linear attention (intra-chunk, parallel)
        # scores[i,j] = q[i] · k[j] for j <= i, weighted by beta[j]
        # This is the "what this chunk contributes" part
        scores = torch.matmul(q_c, k_c.transpose(-1, -2))  # (B,H,nc,C,C)
        causal = torch.tril(torch.ones(C, C, device=q.device, dtype=q.dtype))
        scores = scores * causal.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # Weight by write strength
        scores = scores * b_c.transpose(-1, -2)  # broadcast beta over rows
        intra = torch.matmul(scores, v_c)  # (B, H, nc, C, d)

        # Inter-chunk state propagation (sequential over n_chunks, not N)
        # S: (B, H, d, d) accumulated key-value state
        S = q.new_zeros(B, H, d, d)
        # Compute mean decay per chunk for state propagation
        a_c = alpha.reshape(B, H, n_chunks, C, 1)
        # Chunk decay ≈ product of per-token decays = mean^C (log-space approx)
        chunk_decay = a_c.mean(dim=3) ** C  # (B, H, nc, 1)

        inter_list = []
        for ci in range(n_chunks):
            # State contribution: each token in chunk reads from S
            # inter[t] = q[t] @ S for all t in chunk (parallel)
            inter = torch.einsum('bhij,bhcj->bhci', S, q_c[:, :, ci])  # (B,H,C,d)
            inter_list.append(inter)

            # Update state: S = decay * S + sum(beta * v @ k^T) over chunk
            kv = torch.einsum('bhci,bhcj->bhij',
                              b_c[:, :, ci] * v_c[:, :, ci],
                              k_c[:, :, ci])  # (B, H, d, d)
            S = chunk_decay[:, :, ci].unsqueeze(-1) * S + kv

        inter = torch.stack(inter_list, dim=2)  # (B, H, nc, C, d)

        # Combine intra-chunk + inter-chunk corrections
        corrections = (intra + inter).reshape(B, H, Np, d)
        if pad > 0:
            corrections = corrections[:, :, :N]

        return corrections * self.out_scale


class WaveFieldAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads, field_size=512, max_seq_len=128,
                 n_components=1, use_analytic_kernel=True,
                 feature_map_depth=2,
                 layer_idx=0, num_layers=1,
                 skip_causal_enforce=False,
                 n_frozen_heads=0,
                 use_monarch_fft=None,
                 use_spectral_gate=None,
                 n_attn_heads=0,
                 use_delta_correction=None,
                 device='cuda'):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.field_size = field_size
        self.max_seq_len = max_seq_len
        self.n_components = n_components
        self.use_analytic_kernel = use_analytic_kernel
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.skip_causal_enforce = skip_causal_enforce
        self.device = device

        # Hymba-style mixed heads (optional, default off)
        self.n_attn_heads = n_attn_heads
        self.n_wave_heads = num_heads - n_attn_heads

        assert embedding_dim % num_heads == 0
        assert n_attn_heads < num_heads, "Need at least 1 wave head"

        # Fused QKV + Gate projection: 4D instead of separate 3D + 1D
        self.qkvg_proj = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        # Initialize gate portion (last D columns) to start open (bias=2.0, weight=0)
        with torch.no_grad():
            self.qkvg_proj.weight[3 * embedding_dim:].zero_()
            self.qkvg_proj.bias[3 * embedding_dim:].fill_(2.0)

        # Learned feature maps (Hedgehog-style, identity-init)
        self.q_feature_map = LearnedFeatureMap(self.head_dim, depth=feature_map_depth)
        self.k_feature_map = LearnedFeatureMap(self.head_dim, depth=feature_map_depth)

        # Pad to cuFFT-friendly size (prime factors <= 7).
        # 2x padding for base (analytic kernel is causal by construction).
        # SpectralGate needs 4x (modulation can break Kramers-Kronig).
        if use_spectral_gate is None:
            use_spectral_gate = os.environ.get('SPECTRAL_GATE', '0') == '1'
        pad_mult = 4 if use_spectral_gate else 2
        self._fast_pad_size = _next_fast_size(pad_mult * field_size)
        self.freq_bins = self._fast_pad_size // 2 + 1

        # Optional Monarch FFT: tensor-core-friendly FFT via matmul decomposition
        if use_monarch_fft is None:
            use_monarch_fft = os.environ.get('MONARCH_FFT', '0') == '1'
        self.use_monarch_fft = use_monarch_fft
        self._monarch = None
        if use_monarch_fft:
            from src.fft_optimizer import MonarchFFT
            self._monarch = MonarchFFT(n=self._fast_pad_size, dtype=torch.float32)
            if layer_idx == 0:
                print(f"  [MonarchFFT] Enabled: pad={self._fast_pad_size}, "
                      f"factors=({self._monarch.P},{self._monarch.Q})")

        # SpectralGate: content-adaptive kernel modulation (optional).
        # Only applies to wave heads (attn heads don't use kernels).
        if use_spectral_gate:
            self.spectral_gate = SpectralGate(
                num_heads=self.n_wave_heads,
                head_dim=self.head_dim,
                freq_bins=self.freq_bins,
                n_control=32,
            )
        else:
            self.spectral_gate = None

        # ---- WAVE KERNEL PARAMETERS (wave heads only) ----
        H = self.n_wave_heads

        if n_components == 1:
            # V4.3.2: Per-layer HiPPO init with diversity (S4D, arXiv:2206.11893)
            # Early layers: low freq (broad patterns), low damping (long reach)
            # Later layers: high freq (sharp local), high damping (fast decay)
            # This forces layers to specialize from step 0 instead of all starting identical.
            layer_frac = layer_idx / max(num_layers - 1, 1)  # 0.0 to 1.0

            # Frequency: 0.5x HiPPO (L0) to 2.0x HiPPO (last layer)
            freq_scale = 0.5 * (4.0 ** layer_frac) if num_layers > 1 else 1.0
            hippo_freq = torch.tensor(
                [math.pi * (2 * n + 1) / 2 for n in range(H)]
            ) * freq_scale

            # Damping: per-HEAD diversity within each layer (V4.3.8)
            # V4.3.7 had all heads at same damping → all collapsed to high damping
            # (reach ~2 tokens). Now spread heads from low damping (long range) to
            # high damping (short range) within each layer.
            # Layer center: softplus(-1.4)=0.22 (L0) to softplus(0.0)=0.69 (last)
            damp_center = -1.4 + 1.4 * layer_frac if num_layers > 1 else -0.69
            damp_spread = 0.7  # half-range for per-head diversity
            hippo_damp = torch.linspace(damp_center - damp_spread,
                                        damp_center + damp_spread, H)

            # Phase: offset per layer for inter-layer diversity
            phase_offset = (math.pi / max(num_layers, 1)) * layer_idx
            hippo_phase = torch.linspace(0, math.pi, H) + phase_offset

            self.wave_frequency = nn.Parameter(hippo_freq)
            self.wave_damping = nn.Parameter(hippo_damp)
            self.wave_phase = nn.Parameter(hippo_phase)
            self.component_weights = None

            # V4.3.9: Frozen damping heads — prevent training from collapsing
            # all heads to high damping (bigram shortcut). First n_frozen_heads
            # get log-spaced damping from α=0.05 (reach 184 tokens) to α=0.35
            # (reach 26 tokens). These values are buffers, not parameters, so
            # gradients don't flow and training can't move them.
            if n_frozen_heads > 0 and n_frozen_heads < H:
                frozen_mask = torch.zeros(H, dtype=torch.bool)
                frozen_mask[:n_frozen_heads] = True
                frozen_alphas = torch.zeros(H)
                frozen_alphas[:n_frozen_heads] = torch.exp(torch.linspace(
                    math.log(0.05), math.log(0.35), n_frozen_heads
                ))
                self.register_buffer('_frozen_damping_mask', frozen_mask)
                self.register_buffer('_frozen_damping_values', frozen_alphas)
            else:
                self.register_buffer('_frozen_damping_mask', None)
                self.register_buffer('_frozen_damping_values', None)
        else:
            # Multi-component wavelet: (H, C) params with multi-resolution init
            # Each component covers a different scale — from broad to sharp
            freq_ranges = [
                (0.3, 4.0),    # broad (same as V3.5 single component)
                (2.0, 8.0),    # medium
                (5.0, 15.0),   # sharp
                (10.0, 30.0),  # spike
            ]
            damp_ranges = [
                (-3.0, 0.5),   # broad (low damping = far reach)
                (-1.0, 2.0),   # medium
                (1.0, 4.0),    # sharp (high damping = fast decay)
                (3.0, 6.0),    # spike (very local)
            ]

            freq_init = torch.zeros(H, C)
            damp_init = torch.zeros(H, C)
            phase_init = torch.zeros(H, C)

            for c in range(C):
                ci = min(c, len(freq_ranges) - 1)
                freq_init[:, c] = torch.linspace(freq_ranges[ci][0], freq_ranges[ci][1], H)
                damp_init[:, c] = torch.linspace(damp_ranges[ci][0], damp_ranges[ci][1], H)
                phase_init[:, c] = torch.linspace(0, math.pi, H)

            self.wave_frequency = nn.Parameter(freq_init)     # (H, C)
            self.wave_damping = nn.Parameter(damp_init)        # (H, C)
            self.wave_phase = nn.Parameter(phase_init)         # (H, C)
            self.component_weights = nn.Parameter(torch.zeros(H, C))  # uniform after softmax

            # Multi-component: frozen damping not supported (would need per-C values)
            self.register_buffer('_frozen_damping_mask', None)
            self.register_buffer('_frozen_damping_values', None)

        # Static multi-field coupling
        self.field_coupling = nn.Parameter(
            torch.eye(H) + torch.randn(H, H) * 0.01
        )

        # V4.5.0: Cross-dimension mixing after gather.
        # The Hadamard product q_feat ⊙ gathered keeps all head_dim dimensions
        # independent (48 scalar products, not 1 dot product). This Linear layer
        # lets dimensions interact: "if dims 3 and 17 are both high, output high
        # in dim 5". Identity init = no change at start.
        self.cross_dim = nn.Linear(self.head_dim, self.head_dim, bias=False)
        nn.init.eye_(self.cross_dim.weight)

        # Hymba branch scales (only when n_attn_heads > 0)
        if n_attn_heads > 0:
            self.wave_branch_scale = nn.Parameter(torch.tensor(1.0))
            self.attn_branch_scale = nn.Parameter(torch.tensor(1.0))

        # Delta-rule correction: additive fix for associative recall failures.
        # Runs alongside wave path, doesn't modify it. Starts near-zero (out_scale=0.01).
        if use_delta_correction is None:
            use_delta_correction = os.environ.get('DELTA_CORRECTION', '0') == '1'
        if use_delta_correction:
            self.delta_correction = DeltaCorrection(self.head_dim, self.n_wave_heads)
        else:
            self.delta_correction = None

        # Fixed stride for absolute position mapping
        # CAUSALITY CRITICAL: stride < 1 causes bilinear interpolation to share
        # field cells between adjacent tokens, leaking future info through the
        # gather step. Force stride >= 1.0 so each token maps to its own cell.
        stride_val = (field_size - 1) / max(max_seq_len - 1, 1)
        if stride_val < 1.0:
            # Clamp to 1.0: tokens beyond field_size will map to the last cell.
            # This is safe because seq_len <= field_size in practice.
            stride_val = 1.0
        self.register_buffer(
            'field_stride',
            torch.tensor(stride_val, dtype=torch.float32)
        )

        # Precompute scatter/gather indices (same every forward pass)
        seq_pos = torch.arange(max_seq_len, dtype=torch.float32)
        field_pos = (seq_pos * stride_val).clamp(0, field_size - 1)
        # Integer mapping: each token gets its own cell, no interpolation
        idx_lo = field_pos.long().clamp(0, field_size - 1)
        idx_hi = (idx_lo + 1).clamp(0, field_size - 1)
        frac = (field_pos - idx_lo.float()).clamp(0, 1)
        self.register_buffer('_cached_field_pos', field_pos)
        self.register_buffer('_cached_idx_lo', idx_lo)
        self.register_buffer('_cached_idx_hi', idx_hi)
        self.register_buffer('_cached_frac', frac)

        self.scale = math.sqrt(self.head_dim)

        # Kernel FFT cache
        self._kernel_fft_cache = None
        self._kernel_param_snapshot = None

    def _get_damping(self):
        """Compute effective damping with frozen-head support and clamping.

        Returns: (H,) tensor of damping values in [0, 1.5].
        """
        alpha = F.softplus(self.wave_damping)
        if self._frozen_damping_mask is not None:
            trainable_alpha = alpha.clone().clamp(max=1.5)
            return torch.where(
                self._frozen_damping_mask,
                self._frozen_damping_values,
                trainable_alpha
            )
        return alpha.clamp(max=1.5)

    def _check_eval_cache(self):
        """Return cached kernel FFT if params haven't changed (eval only)."""
        if not self.training and self._kernel_fft_cache is not None and self._kernel_param_snapshot is not None:
            cache_keys = [self.wave_frequency.data, self.wave_damping.data, self.wave_phase.data]
            if self.component_weights is not None:
                cache_keys.append(self.component_weights.data)
            if all(s.equal(c) for s, c in zip(self._kernel_param_snapshot, cache_keys)):
                return self._kernel_fft_cache
        return None

    def _save_eval_cache(self, result):
        """Store kernel FFT in cache (eval only)."""
        if not self.training:
            cache_keys = [self.wave_frequency.data, self.wave_damping.data, self.wave_phase.data]
            if self.component_weights is not None:
                cache_keys.append(self.component_weights.data)
            self._kernel_fft_cache = result
            self._kernel_param_snapshot = [p.clone() for p in cache_keys]

    def _build_wave_kernels(self, device):
        """Build causal wave kernels via time-domain sampling + FFT.

        Single-component (C=1) or multi-component (C>1) damped cosine kernels.
        """
        cached = self._check_eval_cache()
        if cached is not None:
            return cached

        G = self.field_size
        t = torch.arange(G, dtype=torch.float32, device=device)

        if self.n_components == 1:
            alpha = self._get_damping().unsqueeze(1)              # (H, 1)
            omega = self.wave_frequency.unsqueeze(1)              # (H, 1)
            phi = self.wave_phase.unsqueeze(1)                    # (H, 1)
            kernels = torch.exp(-alpha * t.unsqueeze(0)) * torch.cos(omega * t.unsqueeze(0) + phi)
        else:
            alpha = F.softplus(self.wave_damping).clamp(max=1.5).unsqueeze(2)   # (H, C, 1)
            omega = self.wave_frequency.unsqueeze(2)              # (H, C, 1)
            phi = self.wave_phase.unsqueeze(2)                    # (H, C, 1)
            t_exp = t.unsqueeze(0).unsqueeze(0)                   # (1, 1, G)
            components = torch.exp(-alpha * t_exp) * torch.cos(omega * t_exp + phi)
            weights = F.softmax(self.component_weights, dim=-1).unsqueeze(2)  # (H, C, 1)
            kernels = (weights * components).sum(dim=1)  # (H, G)

        # Normalize and FFT (always fp32 for numerical stability)
        kernels = kernels.float()
        kernel_sum = kernels.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
        kernels = kernels / kernel_sum

        result = torch.fft.rfft(kernels, n=self._fast_pad_size)
        self._save_eval_cache(result)
        return result

    def _build_analytic_kernel_fft(self, device):
        """Analytic FFT via Z-transform (S4D-style). Single-component only.

        Computes DFT directly from complex poles — causal by construction,
        no time-domain materialization needed.
        """
        cached = self._check_eval_cache()
        if cached is not None:
            return cached

        G = self.field_size
        pad_size = self._fast_pad_size
        freq_bins = self.freq_bins

        alpha = self._get_damping()
        omega = self.wave_frequency
        phi = self.wave_phase

        # Build complex quantities (all in float32)
        lam = torch.complex(-alpha.float(), omega.float())                  # (H,)
        c = torch.complex(torch.cos(phi.float()), torch.sin(phi.float()))   # (H,) = e^{i*phi}

        # DFT frequency bins: z_k = exp(i * 2*pi*k / pad_size)
        k = torch.arange(freq_bins, device=device, dtype=torch.float32)
        z_angles = 2.0 * math.pi * k / pad_size                             # (freq_bins,)
        z = torch.complex(torch.cos(z_angles), torch.sin(z_angles))         # (freq_bins,)

        # Geometric sum: H(z) = (1 - exp(lam*G)*z^{-G}) / (1 - exp(lam)*z^{-1})
        # Reshape for broadcasting: (H, 1) x (1, freq_bins)
        exp_lam = torch.exp(lam).unsqueeze(1)      # (H, 1) per-step decay
        exp_lam_G = torch.exp(lam * G).unsqueeze(1) # (H, 1) total decay over G
        c_bc = c.unsqueeze(1)                        # (H, 1)
        z_bc = z.unsqueeze(0)                        # (1, freq_bins)

        z_inv = 1.0 / z_bc                           # z^{-1}
        z_inv_G = z_inv ** G                          # z^{-G}

        numerator = 1.0 - exp_lam_G * z_inv_G        # (H, freq_bins)
        denominator = 1.0 - exp_lam * z_inv           # (H, freq_bins)

        # Numerical safety: always add small constant to prevent division by zero.
        # (pole on unit circle = exp(lam)*z^{-1} = 1, impossible when alpha > 0,
        # but guards against numerical edge cases)
        denom_safe = denominator + 1e-10

        H_z = numerator / denom_safe                  # (H, freq_bins) complex

        # Conjugate pole: lam* = -alpha - i*omega
        # H_{lam*}(z_k) uses conj(exp(lam)) but SAME z_k^{-1} (not conj(z_k^{-1}))
        # This is NOT the same as conj(H_lam(z_k)) since conj(z_k^{-1}) = z_k
        exp_lam_conj = exp_lam.conj()                  # (H, 1)
        exp_lam_G_conj = exp_lam_G.conj()              # (H, 1)

        numer_conj = 1.0 - exp_lam_G_conj * z_inv_G   # (H, freq_bins)
        denom_conj = 1.0 - exp_lam_conj * z_inv        # (H, freq_bins)
        denom_conj_safe = denom_conj + 1e-10

        H_z_conj = numer_conj / denom_conj_safe        # (H, freq_bins) complex

        # Real kernel DFT = c * H_lam(z) + conj(c) * H_{lam*}(z)
        # This is the DFT of k(t) = Re[c * exp(lam*t)] (the real cosine kernel)
        kernel_fft = c_bc * H_z + c_bc.conj() * H_z_conj  # (H, freq_bins) complex

        # Normalize by DC component (kernel_fft[:, 0] = sum of time-domain kernel).
        # NOTE: DC norm != L1 norm for oscillatory kernels. High-frequency heads
        # get amplified because positive/negative lobes cancel in DC but not L1.
        # This is intentional — it gives high-freq heads more influence, which
        # all best results (V4.3.3=234, V4.6.0=228.5) relied on.
        dc = kernel_fft[:, 0:1].real.abs().clamp(min=1e-8)
        kernel_fft = kernel_fft / dc

        self._save_eval_cache(kernel_fft)
        return kernel_fft

    def _enforce_causal_kernel(self, kernel_fft, G):
        """Project a frequency-domain kernel back to causal (zero for t >= G).

        Spectral gate modulation can break the Kramers-Kronig relation,
        introducing anti-causal components (non-zero at positions G..2G-1).
        This enforces causality by round-tripping through time domain:
          IFFT → zero anti-causal half → FFT.

        kernel_fft: (..., freq_bins) complex — static (H,F) or adaptive (B,H,F)
        Returns: same shape, causal-projected.
        """
        pad_size = self._fast_pad_size
        if self._monarch is not None:
            kernel_td = self._monarch.irfft(kernel_fft, n=pad_size)
            kernel_td[..., G:] = 0
            return self._monarch.rfft(kernel_td)
        else:
            kernel_td = torch.fft.irfft(kernel_fft, n=pad_size)  # (..., pad_size)
            kernel_td[..., G:] = 0  # zero anti-causal half
            return torch.fft.rfft(kernel_td, n=pad_size)

    @staticmethod
    def _complex_mul_real(a_real, a_imag, b_real, b_imag):
        """Complex multiply decomposed into real ops (Inductor-fusible).

        (a + bi)(c + di) = (ac - bd) + (ad + bc)i

        Inductor can't fuse complex*complex (opaque struct-of-arrays layout).
        Decomposing into 4 real multiplies lets it fuse with surrounding ops.
        """
        return (
            a_real * b_real - a_imag * b_imag,
            a_real * b_imag + a_imag * b_real,
        )

    def _wave_convolve(self, field, kernel_fft, needs_causal_enforce=False):
        """Per-head wave convolution via zero-padded FFT (linear convolution).

        Args:
            field: (B, H, G, D)
            kernel_fft: (H, freq_bins) static or (B, H, freq_bins) adaptive
            needs_causal_enforce: if True, project kernel to causal via
                IFFT→zero→FFT. Required when SpectralGate modulates the kernel
                (can break Kramers-Kronig). Not needed for analytic kernels.
        """
        B, H, G, D = field.shape
        pad_size = self._fast_pad_size

        if needs_causal_enforce and not self.skip_causal_enforce:
            kernel_fft = self._enforce_causal_kernel(kernel_fft, G)

        # Keep 4D: (B, H, G, D) → (B, D, H, G) — FFT along last dim (G)
        field_t = field.permute(0, 3, 1, 2).contiguous()

        # FFT in fp32 for numerical stability (bf16 twiddle factors lose precision)
        input_dtype = field_t.dtype
        if self._monarch is not None:
            field_fft = self._monarch.rfft(field_t.float())    # (B, D, H, freq)
        else:
            field_fft = torch.fft.rfft(field_t.float(), n=pad_size)  # (B, D, H, freq)

        # Decompose complex multiply into real ops for Inductor fusion
        f_real, f_imag = field_fft.real, field_fft.imag

        if kernel_fft.dim() == 3:
            # Content-adaptive: (B, H, freq) → (B, 1, H, freq) broadcasts over D
            k_real = kernel_fft.real.unsqueeze(1)
            k_imag = kernel_fft.imag.unsqueeze(1)
        else:
            # Static: (H, freq) broadcasts naturally over (B, D)
            k_real, k_imag = kernel_fft.real, kernel_fft.imag

        out_real, out_imag = self._complex_mul_real(f_real, f_imag, k_real, k_imag)
        convolved_fft = torch.complex(out_real, out_imag)

        if self._monarch is not None:
            convolved = self._monarch.irfft(convolved_fft, n=pad_size)[..., :G]
        else:
            convolved = torch.fft.irfft(convolved_fft, n=pad_size)[..., :G]  # (B, D, H, G)
        convolved = convolved.to(input_dtype)

        return convolved.permute(0, 2, 3, 1)  # back to (B, H, G, D)

    def _bilinear_scatter(self, values, idx_lo, idx_hi, frac, B, H, G, head_dim, device):
        """Deposit values onto field using integer or bilinear interpolation.

        V4.4.0: When stride >= 1.0, frac is always 0 — skip the expensive
        bilinear path (2 scatter_add_ + 2 weight multiplies) and use direct
        single scatter_add_ instead. ~2x faster on scatter step.
        """
        N = idx_lo.shape[0]
        field = torch.zeros(B, H, G, head_dim, device=device, dtype=values.dtype)
        idx_lo_exp = idx_lo.view(1, 1, N, 1).expand(B, H, -1, head_dim)

        if frac.abs().max() < 1e-6:
            # Integer stride: no interpolation needed
            field.scatter_add_(2, idx_lo_exp, values)
        else:
            # Fractional stride: bilinear interpolation
            w_lo = (1.0 - frac).to(values.dtype).view(1, 1, N, 1)
            w_hi = frac.to(values.dtype).view(1, 1, N, 1)
            idx_hi_exp = idx_hi.view(1, 1, N, 1).expand(B, H, -1, head_dim)
            field.scatter_add_(2, idx_lo_exp, values * w_lo)
            field.scatter_add_(2, idx_hi_exp, values * w_hi)

        return field

    def _bilinear_gather(self, field, idx_lo, idx_hi, frac):
        """Read from field using integer or bilinear interpolation.

        V4.4.0: Fast path when frac=0 (integer stride) — single gather.
        """
        B, H, G, D = field.shape
        N = idx_lo.shape[0]
        idx_lo_exp = idx_lo.view(1, 1, N, 1).expand(B, H, -1, D)

        if frac.abs().max() < 1e-6:
            # Integer stride: direct gather
            return torch.gather(field, 2, idx_lo_exp)
        else:
            # Fractional stride: bilinear interpolation
            w_lo = (1.0 - frac).to(field.dtype).view(1, 1, N, 1)
            w_hi = frac.to(field.dtype).view(1, 1, N, 1)
            idx_hi_exp = idx_hi.view(1, 1, N, 1).expand(B, H, -1, D)
            val_lo = torch.gather(field, 2, idx_lo_exp)
            val_hi = torch.gather(field, 2, idx_hi_exp)
            return val_lo * w_lo + val_hi * w_hi

    def _apply_field_coupling(self, field):
        """Static multi-field coupling via einsum (avoids flatten + bmm + reshape)."""
        coupling = F.softmax(self.field_coupling, dim=-1)
        return torch.einsum('ij,bjgd->bigd', coupling, field)

    def forward(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        B, N, D = x.shape
        G = self.field_size
        H = self.num_heads
        Hw = self.n_wave_heads
        Ha = self.n_attn_heads
        head_dim = self.head_dim

        # Fused QKV + Gate projection (single matmul)
        qkvg = self.qkvg_proj(x)
        q, k, v, gate_raw = qkvg.chunk(4, dim=-1)

        q = q.view(B, N, H, head_dim).transpose(1, 2)
        k = k.view(B, N, H, head_dim).transpose(1, 2)
        v = v.view(B, N, H, head_dim).transpose(1, 2)

        # ================= SPLIT HEADS: wave [:Hw] + attention [Hw:] =================
        q_wave, q_attn = q[:, :Hw], q[:, Hw:]
        k_wave, k_attn = k[:, :Hw], k[:, Hw:]
        v_wave, v_attn = v[:, :Hw], v[:, Hw:]

        # ================= WAVE FIELD PATH (O(n log n)) =================
        # ABSOLUTE POSITION MAPPING — use precomputed indices (slice to actual N)
        if N <= self._cached_idx_lo.shape[0]:
            idx_lo = self._cached_idx_lo[:N]
            idx_hi = self._cached_idx_hi[:N]
            frac = self._cached_frac[:N]
        else:
            seq_pos = torch.arange(N, device=x.device, dtype=torch.float32)
            field_pos_float = (seq_pos * self.field_stride).clamp(0, G - 2)
            idx_lo = field_pos_float.long().clamp(0, G - 2)
            idx_hi = idx_lo + 1
            frac = (field_pos_float - idx_lo.float()).clamp(0, 1)

        # Learned feature maps (Hedgehog-style)
        q_feat = self.q_feature_map(q_wave)  # (B, Hw, N, head_dim)
        k_feat = self.k_feature_map(k_wave)  # (B, Hw, N, head_dim)

        # K-weighted deposit → scatter → convolve → couple → gather
        deposit = k_feat * v_wave  # (B, Hw, N, head_dim)
        field = self._bilinear_scatter(deposit, idx_lo, idx_hi, frac, B, Hw, G, head_dim, x.device)

        if self.use_analytic_kernel and self.n_components == 1:
            base_kernel_fft = self._build_analytic_kernel_fft(x.device)
        else:
            base_kernel_fft = self._build_wave_kernels(x.device)

        has_spectral_gate = self.spectral_gate is not None
        if has_spectral_gate:
            kernel_fft = self.spectral_gate(q_wave, base_kernel_fft)
        else:
            kernel_fft = base_kernel_fft
        field = self._wave_convolve(field, kernel_fft,
                                    needs_causal_enforce=has_spectral_gate)

        field = self._apply_field_coupling(field)
        gathered = self._bilinear_gather(field, idx_lo, idx_hi, frac)  # (B, Hw, N, head_dim)

        # Q-weighted reading + cross-dimension mixing
        wave_output = self.cross_dim(q_feat * gathered)  # (B, Hw, N, head_dim)

        # ================= DELTA CORRECTION (additive, O(n·d²)) =================
        if self.delta_correction is not None:
            wave_output = wave_output + self.delta_correction(q_wave, k_wave, v_wave)

        # ================= ATTENTION PATH (O(n²), Hymba-style) =================
        if Ha > 0:
            attn_output = F.scaled_dot_product_attention(
                q_attn, k_attn, v_attn, is_causal=True
            )  # (B, Ha, N, head_dim)
            # Learnable scalar balance (2 params) — gate handles per-head weighting
            combined = torch.cat([
                wave_output * self.wave_branch_scale,
                attn_output * self.attn_branch_scale
            ], dim=1)  # (B, H, N, head_dim)
        else:
            combined = wave_output

        # Content-dependent gating (applies to all heads)
        gate = torch.sigmoid(gate_raw).view(B, N, H, head_dim).transpose(1, 2)
        output = combined * gate

        output = output.transpose(1, 2).reshape(B, N, D)
        output = self.out_proj(output)

        if squeeze_output:
            output = output.squeeze(0)

        return output
