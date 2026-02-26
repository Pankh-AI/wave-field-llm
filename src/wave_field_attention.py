"""
Wave Field Attention V4.3 - SPECTRE-Wave Hybrid
================================================

V4.1 hit PPL 543 at 15M tokens but needed 3x more data than Standard
Transformer (PPL 473 at 5M). V4.2 init fixes gave PPL 997 at 5M — still 2x gap.

V4.3 FIX — Three architectural changes grounded in literature:

  1. LEARNED FEATURE MAPS (Hedgehog, ICLR 2024):
     Replace elu(x)+1 with Linear(head_dim) initialized as identity + ReLU.
     At init, φ(q) ≈ ReLU(q) — every token is distinct from step 1.
     Cost: +49K params (0.6%)

  2. HiPPO KERNEL INIT (S4D, arXiv:2206.11893):
     Uniform damping + harmonic frequencies ω_n = π(2n+1)/2.
     Cost: Zero (just initialization)

  3. CONTENT-ADAPTIVE SPECTRAL GATE (SPECTRE, arXiv:2502.18394):
     A small MLP conditioned on mean(Q) produces per-head spectral
     modulation of the wave kernel FFT. This makes the effective
     attention kernel INPUT-DEPENDENT — different inputs get different
     receptive fields. SPECTRE proved this beats standard transformers
     (PPL 39.0 vs 39.4 on PG-19).

     The MLP produces 32 control points per head, interpolated to full
     frequency resolution (smooth, low-rank spectral gate). Initialized
     near zero so model starts identical to base V4.3.
     Cost: +131K params (1.6%)

Pipeline:
  1. DEPOSIT: φ_k(K) ⊙ V  — learned K feature map modulates V
  2. PROPAGATE: Wave convolution with CONTENT-ADAPTIVE kernel
     (base HiPPO kernel × spectral gate conditioned on mean(Q))
  3. GATHER: Read field at token positions
  4. READ: φ_q(Q) ⊙ gathered  — learned Q feature map selects dims

Complexity: O(n log n) — unchanged from V4.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class ELUPlus1(nn.Module):
    """elu(x) + 1: always positive, no dead neurons, smooth gradient everywhere.

    Standard activation for linear attention feature maps (Katharopoulos et al., 2020).
    Unlike ReLU which permanently kills ~50% of neurons, ELU+1 has non-zero gradient
    for all inputs: f'(x) = 1 for x>0, f'(x) = exp(x) for x<0.
    """

    def forward(self, x):
        return F.elu(x) + 1.0


class LearnedFeatureMap(nn.Module):
    """Learned positive feature map for linear attention (Hedgehog, ICLR 2024).

    MLP of `depth` identity-initialized Linear(d,d) + ReLU layers.
    At init: phi(x) = ReLU(Ix) + eps — 50% sparse, rank d/2.
    During training: learns spiky, dot-product-monotonic maps that mimic softmax.

    V4.3.4: Reverted to ReLU from ELU+1. ELU+1 collapsed effective rank to 2.3
    because ELU(0)+1=1 maps all outputs ≈ [1,1,...,1]. ReLU's 50% zeroed dims
    give rank d/2 = 24 >> 2.3. The eps floor ensures positivity for linear attn.

    depth=1: original V4.3 (single linear).
    depth=2: Hedgehog-style (closes 68.6% of linear-vs-softmax gap at 125M scale).
    """

    def __init__(self, dim, eps=1e-6, depth=2):
        super().__init__()
        self.eps = eps
        layers = []
        for _ in range(depth):
            lin = nn.Linear(dim, dim, bias=True)
            with torch.no_grad():
                nn.init.eye_(lin.weight)
                nn.init.zeros_(lin.bias)
            layers.append(lin)
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) + self.eps


class SpectralGate(nn.Module):
    """Content-adaptive spectral gate (SPECTRE, arXiv:2502.18394).

    A small MLP conditioned on the first query token modulates the base wave
    kernel in frequency domain. This makes the effective attention pattern
    input-dependent while staying O(n log n).

    Architecture:
      q_bar = LayerNorm(q[:, :, 0, :])       # (B, H, head_dim) — token 0 only
      ctrl = MLP(flatten(q_bar))             # (B, H, n_control)
      gate = interpolate(ctrl, freq_bins)    # (B, H, freq_bins) — smooth
      modulated = base_fft * (1 + gate)      # content-adaptive kernel

    CAUSALITY NOTE: Must NOT use mean(Q) over all positions — that leaks future
    tokens into the kernel shape. Token 0 is visible to all positions, so safe.

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


class WaveFieldAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads, field_size=512, max_seq_len=128,
                 n_components=1, local_window=0, use_analytic_kernel=True,
                 feature_map_depth=2, use_write_gate=True,
                 use_3d_interference=False,
                 use_kernel_mixture=False, num_basis_kernels=4,
                 layer_idx=0, num_layers=1,
                 skip_causal_enforce=False,
                 device='cuda'):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.field_size = field_size
        self.max_seq_len = max_seq_len
        self.n_components = n_components
        self.local_window = local_window
        self.use_analytic_kernel = use_analytic_kernel
        self.use_write_gate = use_write_gate
        self.use_3d_interference = use_3d_interference
        self.use_kernel_mixture = use_kernel_mixture
        self.num_basis_kernels = num_basis_kernels
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.skip_causal_enforce = skip_causal_enforce
        self.device = device

        assert embedding_dim % num_heads == 0

        # Fused QKV + Gate projection: 4D instead of separate 3D + 1D
        self.qkvg_proj = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        # Initialize gate portion (last D columns) to start open (bias=2.0, weight=0)
        with torch.no_grad():
            self.qkvg_proj.weight[3 * embedding_dim:].zero_()
            self.qkvg_proj.bias[3 * embedding_dim:].fill_(2.0)

        # V4.3: Learned feature maps (Hedgehog-style, identity-init)
        # depth=1: single Linear+ReLU (original). depth=2: 2-layer MLP (Hedgehog).
        self.q_feature_map = LearnedFeatureMap(self.head_dim, depth=feature_map_depth)
        self.k_feature_map = LearnedFeatureMap(self.head_dim, depth=feature_map_depth)

        # V4.4: Selective write gate (GLA-inspired)
        # Controls per-token, per-head write strength to wave field.
        # Input is raw K (mean-zero at init) → sigmoid(0) = 0.5 (neutral).
        if use_write_gate:
            self.write_gate_proj = nn.Linear(self.head_dim, 1, bias=True)

        # Pad to cuFFT-friendly size (prime factors <= 7).
        # 4x padding (not 2x) to prevent FFT circular wraparound leaking future
        # tokens when field_size ≈ seq_len. With 2x, fp32 precision errors in the
        # IFFT→zero→FFT causal enforcement chain create ~1e-4 leakage at init that
        # training amplifies to ~10 logit diffs. 4x padding gives enough zero-buffer
        # to keep leakage at ~1e-7 (harmless). Cost: ~2x larger FFT.
        self._fast_pad_size = _next_fast_size(4 * field_size)
        # rfft(n=pad) produces pad//2 + 1 complex frequency bins
        self.freq_bins = self._fast_pad_size // 2 + 1

        if use_kernel_mixture:
            # Content-Adaptive Kernel Mixture: K basis kernels, per-token mixing
            # Replaces SpectralGate with per-token (not per-sample) adaptivity
            K = num_basis_kernels
            H = num_heads

            # Multi-scale HiPPO init: spread frequencies and dampings across K scales
            hippo_base = torch.tensor([math.pi * (2 * n + 1) / 2 for n in range(H)])
            freq_scales = torch.tensor([0.5, 1.0, 2.0, 4.0][:K])
            damp_raw = torch.tensor([-1.5, -0.69, 0.0, 0.69][:K])
            phase_base = torch.linspace(0, math.pi, H)

            self.basis_frequency = nn.Parameter(
                hippo_base.unsqueeze(1) * freq_scales.unsqueeze(0)  # (H, K)
            )
            self.basis_damping = nn.Parameter(
                damp_raw.unsqueeze(0).expand(H, K).clone()  # (H, K)
            )
            self.basis_phase = nn.Parameter(
                phase_base.unsqueeze(1).expand(H, K).clone()  # (H, K)
            )

            # Per-token mixing: q @ mix_proj -> (B, H, N, K) weights
            # Zero init for projection (content-dependent deviation)
            self.kernel_mix_proj = nn.Parameter(torch.zeros(H, self.head_dim, K))

            # Bias: strongly prefer basis 1 (standard HiPPO) at init
            # softmax([0, 5, 0, 0]) ≈ [0.5%, 97.9%, 0.5%, 0.5%]
            # This gives a coherent starting point (like V4.3 single kernel)
            # Training gradually shifts weight to other kernels per-token
            bias_init = torch.zeros(K)
            if K > 1:
                bias_init[1] = 5.0  # basis 1 = 1.0x HiPPO (standard)
            self.kernel_mix_bias = nn.Parameter(
                bias_init.unsqueeze(0).expand(H, K).clone()  # (H, K)
            )

            self.spectral_gate = None
        else:
            # V4.3: Content-adaptive spectral gate (SPECTRE-style)
            self.spectral_gate = SpectralGate(
                num_heads=num_heads,
                head_dim=self.head_dim,
                freq_bins=self.freq_bins,
                n_control=32,
            )

        # ---- WAVE KERNEL PARAMETERS ----
        H = num_heads
        C = n_components

        if C == 1:
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

            # Damping: softplus(-1.4)=0.22 (L0, long reach) to softplus(0.0)=0.69 (last, local)
            damp_raw = -1.4 + 1.4 * layer_frac if num_layers > 1 else -0.69
            hippo_damp = torch.full((H,), damp_raw)

            # Phase: offset per layer for inter-layer diversity
            phase_offset = (math.pi / max(num_layers, 1)) * layer_idx
            hippo_phase = torch.linspace(0, math.pi, H) + phase_offset

            self.wave_frequency = nn.Parameter(hippo_freq)
            self.wave_damping = nn.Parameter(hippo_damp)
            self.wave_phase = nn.Parameter(hippo_phase)
            self.component_weights = None
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

        # ---- LOCAL ATTENTION (near-field) ----
        if local_window > 0:
            # Per-head blend: sigmoid(1.0) = 0.73 — biased toward local at init
            # BASED (Hazy Research 2024) showed local patterns dominate short-range
            self.local_blend = nn.Parameter(torch.full((H,), 1.0))

            # Precompute causal + window mask (registered buffer → moves with .to(device))
            rows = torch.arange(max_seq_len).unsqueeze(1)
            cols = torch.arange(max_seq_len).unsqueeze(0)
            mask = (cols <= rows) & (rows - cols < local_window)
            self.register_buffer('_window_mask', mask)
        else:
            self.local_blend = None

        # Static multi-field coupling (used when 3D interference is off)
        self.field_coupling = nn.Parameter(
            torch.eye(H) + torch.randn(H, H) * 0.01
        )

        # V4.4: 3D wave interference — content-dependent cross-head mixing
        # Replaces static field_coupling with physics-based interference
        if use_3d_interference:
            self.head_positions = nn.Parameter(
                self._init_head_positions(H)
            )
            self.token_pos_proj = nn.Linear(embedding_dim, 3)

        # Fixed stride for absolute position mapping
        stride_val = (field_size - 1) / max(max_seq_len - 1, 1)
        self.register_buffer(
            'field_stride',
            torch.tensor(stride_val, dtype=torch.float32)
        )

        # Precompute scatter/gather indices (same every forward pass)
        seq_pos = torch.arange(max_seq_len, dtype=torch.float32)
        field_pos = (seq_pos * stride_val).clamp(0, field_size - 2)
        idx_lo = field_pos.long().clamp(0, field_size - 2)
        idx_hi = idx_lo + 1
        frac = (field_pos - idx_lo.float()).clamp(0, 1)
        self.register_buffer('_cached_field_pos', field_pos)
        self.register_buffer('_cached_idx_lo', idx_lo)
        self.register_buffer('_cached_idx_hi', idx_hi)
        self.register_buffer('_cached_frac', frac)

        self.scale = math.sqrt(self.head_dim)

        # Kernel FFT cache
        self._kernel_fft_cache = None
        self._kernel_param_snapshot = None

    def _build_wave_kernels(self, device):
        """
        Build LEFT-ALIGNED causal wave kernels with caching.

        Single-component (C=1): exact V3.5 behavior.
        Multi-component (C>1): weighted sum of C damped waves per head,
        enabling multi-resolution attention (wavelet decomposition).
        """
        # Cache only during eval — during training, params change every step AND
        # gradient checkpointing re-runs forward with different tensor counts if cached
        if not self.training:
            cache_keys = [self.wave_frequency.data, self.wave_damping.data, self.wave_phase.data]
            if self.component_weights is not None:
                cache_keys.append(self.component_weights.data)

            if self._kernel_fft_cache is not None and self._kernel_param_snapshot is not None:
                if all(s.equal(c) for s, c in zip(self._kernel_param_snapshot, cache_keys)):
                    return self._kernel_fft_cache

        G = self.field_size

        t = torch.arange(G, dtype=torch.float32, device=device)

        if self.n_components == 1:
            # Single component: params are (H,) — exact V3.5 path
            alpha = F.softplus(self.wave_damping).unsqueeze(1)   # (H, 1)
            omega = self.wave_frequency.unsqueeze(1)              # (H, 1)
            phi = self.wave_phase.unsqueeze(1)                    # (H, 1)

            kernels = torch.exp(-alpha * t.unsqueeze(0)) * torch.cos(omega * t.unsqueeze(0) + phi)
        else:
            # Multi-component: params are (H, C), sum weighted components
            alpha = F.softplus(self.wave_damping).unsqueeze(2)   # (H, C, 1)
            omega = self.wave_frequency.unsqueeze(2)              # (H, C, 1)
            phi = self.wave_phase.unsqueeze(2)                    # (H, C, 1)
            t_exp = t.unsqueeze(0).unsqueeze(0)                   # (1, 1, G)

            # Per-component kernels: (H, C, G)
            components = torch.exp(-alpha * t_exp) * torch.cos(omega * t_exp + phi)

            # Weighted sum across components
            weights = F.softmax(self.component_weights, dim=-1).unsqueeze(2)  # (H, C, 1)
            kernels = (weights * components).sum(dim=1)  # (H, G)

        # Normalize and FFT (always fp32 for numerical stability)
        kernels = kernels.float()
        kernel_sum = kernels.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
        kernels = kernels / kernel_sum

        result = torch.fft.rfft(kernels, n=self._fast_pad_size)

        # Cache (eval only — training invalidates every step + breaks checkpointing)
        if not self.training:
            cache_keys = [self.wave_frequency.data, self.wave_damping.data, self.wave_phase.data]
            if self.component_weights is not None:
                cache_keys.append(self.component_weights.data)
            self._kernel_fft_cache = result
            self._kernel_param_snapshot = [p.clone() for p in cache_keys]

        return result

    def _build_analytic_kernel_fft(self, device):
        """Analytic FFT of complex exponential kernel via Z-transform (S4D-style).

        Instead of materializing G time-domain samples and FFTing, computes the
        DFT directly from complex poles using the geometric series closed form:

          H(z_k) = (1 - exp(lambda*G) * z_k^{-G}) / (1 - exp(lambda) * z_k^{-1})

        where lambda = -alpha + i*omega is the complex pole per head, and
        z_k = exp(i*2*pi*k / 2G) are the DFT frequency bins.

        The real kernel (conjugate pair) is: c*H(z) + conj(c)*H(conj(z))
        where c = exp(i*phi) applies the phase offset.

        Benefits over time-domain approach:
          - Automatic causality (sums only t >= 0, no _enforce_causal_kernel needed)
          - Direct gradient flow through pole params (alpha, omega) to loss
          - Kramers-Kronig compliant by construction
          - Minimum-phase property when alpha > 0

        Returns: (H, freq_bins) complex — same format as _build_wave_kernels.
        Only supports single-component (C=1) kernels.
        """
        # Eval caching (same logic as _build_wave_kernels)
        if not self.training:
            cache_keys = [self.wave_frequency.data, self.wave_damping.data, self.wave_phase.data]
            if self._kernel_fft_cache is not None and self._kernel_param_snapshot is not None:
                if all(s.equal(c) for s, c in zip(self._kernel_param_snapshot, cache_keys)):
                    return self._kernel_fft_cache

        G = self.field_size
        pad_size = self._fast_pad_size
        freq_bins = self.freq_bins  # rfft output size for n=pad_size

        # Complex pole: lambda = -alpha + i*omega
        alpha = F.softplus(self.wave_damping)  # (H,) positive damping
        omega = self.wave_frequency             # (H,) frequency
        phi = self.wave_phase                   # (H,) phase

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

        # Numerical safety: clamp denominator away from zero
        # (pole on unit circle = exp(lam)*z^{-1} = 1, impossible when alpha > 0)
        denom_safe = denominator + 1e-10 * torch.sgn(denominator) * (denominator.abs() < 1e-10)

        H_z = numerator / denom_safe                  # (H, freq_bins) complex

        # Conjugate pole: lam* = -alpha - i*omega
        # H_{lam*}(z_k) uses conj(exp(lam)) but SAME z_k^{-1} (not conj(z_k^{-1}))
        # This is NOT the same as conj(H_lam(z_k)) since conj(z_k^{-1}) = z_k
        exp_lam_conj = exp_lam.conj()                  # (H, 1)
        exp_lam_G_conj = exp_lam_G.conj()              # (H, 1)

        numer_conj = 1.0 - exp_lam_G_conj * z_inv_G   # (H, freq_bins)
        denom_conj = 1.0 - exp_lam_conj * z_inv        # (H, freq_bins)
        denom_conj_safe = denom_conj + 1e-10 * torch.sgn(denom_conj) * (denom_conj.abs() < 1e-10)

        H_z_conj = numer_conj / denom_conj_safe        # (H, freq_bins) complex

        # Real kernel DFT = c * H_lam(z) + conj(c) * H_{lam*}(z)
        # This is the DFT of k(t) = Re[c * exp(lam*t)] (the real cosine kernel)
        kernel_fft = c_bc * H_z + c_bc.conj() * H_z_conj  # (H, freq_bins) complex

        # Normalize by DC component (same effect as L1 normalization in time domain)
        dc = kernel_fft[:, 0:1].real.abs().clamp(min=1e-8)
        kernel_fft = kernel_fft / dc

        # Cache (eval only)
        if not self.training:
            cache_keys = [self.wave_frequency.data, self.wave_damping.data, self.wave_phase.data]
            self._kernel_fft_cache = kernel_fft
            self._kernel_param_snapshot = [p.clone() for p in cache_keys]

        return kernel_fft

    def _build_basis_kernel_ffts(self, device):
        """Build K analytic kernel FFTs for Content-Adaptive Kernel Mixture.

        Vectorized Z-transform: same math as _build_analytic_kernel_fft but
        with (H, K) shaped poles. Returns (K, H, freq_bins) complex.
        Causal by construction — no _enforce_causal_kernel needed.
        """
        G = self.field_size
        pad_size = self._fast_pad_size
        freq_bins = self.freq_bins

        alpha = F.softplus(self.basis_damping)   # (H, K)
        omega = self.basis_frequency              # (H, K)
        phi = self.basis_phase                    # (H, K)

        # Complex poles: lambda = -alpha + i*omega  (H, K)
        lam = torch.complex(-alpha.float(), omega.float())
        c = torch.complex(torch.cos(phi.float()), torch.sin(phi.float()))

        # DFT frequency bins: z_k = exp(i * 2*pi*k / pad_size)
        k_freq = torch.arange(freq_bins, device=device, dtype=torch.float32)
        z = torch.exp(torch.complex(
            torch.zeros_like(k_freq),
            2.0 * math.pi * k_freq / pad_size
        ))  # (freq_bins,)

        # Broadcasting: (H, K, 1) x (1, 1, freq_bins)
        exp_lam = torch.exp(lam).unsqueeze(2)        # (H, K, 1)
        exp_lam_G = torch.exp(lam * G).unsqueeze(2)  # (H, K, 1)
        c_bc = c.unsqueeze(2)                          # (H, K, 1)
        z_bc = z.unsqueeze(0).unsqueeze(0)             # (1, 1, freq_bins)

        z_inv = 1.0 / z_bc
        z_inv_G = z_inv ** G

        numerator = 1.0 - exp_lam_G * z_inv_G
        denominator = 1.0 - exp_lam * z_inv
        denom_safe = denominator + 1e-10 * (denominator.abs() < 1e-10).float()

        H_z = numerator / denom_safe  # (H, K, freq_bins)

        # Conjugate pole
        H_z_conj = (1.0 - exp_lam_G.conj() * z_inv_G) / (
            (1.0 - exp_lam.conj() * z_inv) + 1e-10 * ((1.0 - exp_lam.conj() * z_inv).abs() < 1e-10).float()
        )

        # Real kernel DFT
        kernel_fft = c_bc * H_z + c_bc.conj() * H_z_conj  # (H, K, freq_bins)

        # Normalize by DC
        dc = kernel_fft[:, :, 0:1].real.abs().clamp(min=1e-8)
        kernel_fft = kernel_fft / dc

        return kernel_fft.permute(1, 0, 2)  # (K, H, freq_bins)

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

    def _wave_convolve(self, field, kernel_fft):
        """Per-head wave convolution via zero-padded FFT (linear convolution).

        kernel_fft: (H, freq_bins) for static kernel, or
                    (B, H, freq_bins) for content-adaptive (spectral gate).

        Uses 4D layout (B, D, H, G) to avoid expanding kernel across D —
        CUDA broadcasting handles it, saving 16 MB allocation per layer.
        """
        B, H, G, D = field.shape
        pad_size = self._fast_pad_size

        # Enforce causality: spectral gate modulation can introduce anti-causal
        # components by breaking the Hilbert transform relationship. Project
        # back to causal space before convolving.
        # Skip when using analytic kernel (causal by construction) + real-valued
        # spectral gate (preserves causality). Verify with test_causality.py.
        if kernel_fft.dim() == 3 and not self.skip_causal_enforce:
            kernel_fft = self._enforce_causal_kernel(kernel_fft, G)

        # Keep 4D: (B, H, G, D) → (B, D, H, G) — FFT along last dim (G)
        field_t = field.permute(0, 3, 1, 2).contiguous()

        # FFT in fp32 for numerical stability (bf16 twiddle factors lose precision)
        input_dtype = field_t.dtype
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

        convolved = torch.fft.irfft(convolved_fft, n=pad_size)[..., :G]  # (B, D, H, G)
        convolved = convolved.to(input_dtype)

        return convolved.permute(0, 2, 3, 1)  # back to (B, H, G, D)

    def _bilinear_scatter(self, values, idx_lo, idx_hi, frac, B, H, G, head_dim, device):
        """Deposit values onto field using bilinear interpolation.

        Uses precomputed indices (cached as buffers) to avoid recomputation.
        """
        N = idx_lo.shape[0]

        # Cast weights to match values dtype (AMP produces float16 values but float32 positions)
        w_lo = (1.0 - frac).to(values.dtype).view(1, 1, N, 1)
        w_hi = frac.to(values.dtype).view(1, 1, N, 1)

        field = torch.zeros(B, H, G, head_dim, device=device, dtype=values.dtype)

        idx_lo_exp = idx_lo.view(1, 1, N, 1).expand(B, H, -1, head_dim)
        idx_hi_exp = idx_hi.view(1, 1, N, 1).expand(B, H, -1, head_dim)

        field.scatter_add_(2, idx_lo_exp, values * w_lo)
        field.scatter_add_(2, idx_hi_exp, values * w_hi)

        return field

    def _bilinear_gather(self, field, idx_lo, idx_hi, frac):
        """Read from field using bilinear interpolation.

        Uses precomputed indices (cached as buffers) to avoid recomputation.
        """
        B, H, G, D = field.shape
        N = idx_lo.shape[0]

        # Cast weights to match field dtype (AMP produces float16 field but float32 positions)
        w_lo = (1.0 - frac).to(field.dtype).view(1, 1, N, 1)
        w_hi = frac.to(field.dtype).view(1, 1, N, 1)

        idx_lo_exp = idx_lo.view(1, 1, N, 1).expand(B, H, -1, D)
        idx_hi_exp = idx_hi.view(1, 1, N, 1).expand(B, H, -1, D)

        val_lo = torch.gather(field, 2, idx_lo_exp)
        val_hi = torch.gather(field, 2, idx_hi_exp)

        return val_lo * w_lo + val_hi * w_hi

    def _apply_field_coupling(self, field):
        """Static multi-field coupling via einsum (avoids flatten + bmm + reshape)."""
        coupling = F.softmax(self.field_coupling, dim=-1)
        return torch.einsum('ij,bjgd->bigd', coupling, field)

    def _kernel_mixture_forward(self, field, q, idx_lo, idx_hi, frac, x):
        """Content-Adaptive Kernel Mixture: K basis convolutions, per-token mixing.

        Each query token selects its own weighted mixture of K convolved fields.
        Sequential loop over K kernels to save memory (reuses field_fft).

        field: (B, H, G, D) — scattered deposit
        q: (B, H, N, head_dim) — raw queries (pre-feature-map)
        idx_lo, idx_hi, frac: precomputed scatter/gather indices
        x: (B, N, D) — original input (for 3D interference if enabled)
        Returns: (B, H, N, D) — gathered output
        """
        B, H, G, D = field.shape
        K = self.num_basis_kernels
        N = q.shape[2]
        pad_size = self._fast_pad_size

        # 1. Build K basis kernel FFTs: (K, H, freq_bins)
        basis_ffts = self._build_basis_kernel_ffts(field.device)

        # 2. Per-token mixing weights: q @ mix_proj + bias -> softmax over K
        # fp32 for numerical stability under AMP
        # bias provides strong prior (98% on standard HiPPO at init)
        # q @ mix_proj adds content-dependent deviation
        alpha = torch.einsum('bhnd,hdk->bhnk', q.float(), self.kernel_mix_proj.float())
        alpha = alpha + self.kernel_mix_bias.float().unsqueeze(0).unsqueeze(2)  # (1, H, 1, K)
        alpha = F.softmax(alpha, dim=-1).to(field.dtype)  # (B, H, N, K)

        # 3. Compute field FFT once (reused for all K kernels)
        field_t = field.permute(0, 3, 1, 2).contiguous()  # (B, D, H, G)
        input_dtype = field_t.dtype
        field_fft = torch.fft.rfft(field_t.float(), n=pad_size)  # (B, D, H, freq)

        # 4. Sequential convolution loop: convolve → couple → gather → accumulate
        output = torch.zeros(B, H, N, D, device=field.device, dtype=input_dtype)

        for m in range(K):
            # Convolve with m-th basis kernel: (H, freq) static — broadcasts over (B, D)
            convolved_fft = field_fft * basis_ffts[m]  # (B, D, H, freq)
            convolved = torch.fft.irfft(convolved_fft, n=pad_size)[..., :G]
            field_m = convolved.to(input_dtype).permute(0, 2, 3, 1)  # (B, H, G, D)

            # Field coupling (shared across K)
            if self.use_3d_interference:
                field_m = self._apply_3d_interference(field_m, x)
            else:
                field_m = self._apply_field_coupling(field_m)

            # Gather at token positions
            gathered_m = self._bilinear_gather(field_m, idx_lo, idx_hi, frac)  # (B, H, N, D)

            # Weight by per-token alpha and accumulate
            output = output + alpha[:, :, :, m:m+1] * gathered_m

        return output

    @staticmethod
    def _init_head_positions(H):
        """Initialize head positions for maximum spatial diversity."""
        if H == 8:
            # Cube vertices — perfect spacing for 8 heads
            return torch.tensor([
                [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5],
                [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5],
                [0.5, -0.5, -0.5], [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5], [0.5, 0.5, 0.5],
            ], dtype=torch.float32)
        else:
            # Fibonacci sphere for arbitrary H
            golden = (1 + 5 ** 0.5) / 2
            i = torch.arange(H, dtype=torch.float32)
            theta = 2 * math.pi * i / golden
            phi = torch.acos(1 - 2 * (i + 0.5) / H)
            return torch.stack([
                torch.sin(phi) * torch.cos(theta),
                torch.sin(phi) * torch.sin(theta),
                torch.cos(phi),
            ], dim=-1) * 0.5

    def _apply_3d_interference(self, field, x):
        """Content-dependent cross-head mixing via 3D wave interference.

        Each head is a wave source at a learned 3D position. Token embeddings
        are projected to 3D positions. The interference pattern between heads
        (based on path differences to the reference point) determines coupling.

        field: (B, H, G, D)
        x: (B, N, D) — original input for content-dependent positions
        Returns: (B, H, G, D)
        """
        B, H, G, D_field = field.shape

        # Content-dependent 3D position (first token as reference, like SpectralGate)
        tok_pos = self.token_pos_proj(x)     # (B, N, 3)
        ref_pos = tok_pos[:, 0, :]           # (B, 3)
        head_pos = self.head_positions       # (H, 3)

        # Distance from reference point to each head source
        dist = torch.norm(
            ref_pos.unsqueeze(1) - head_pos.unsqueeze(0), dim=-1
        )  # (B, H)

        # Path difference between head pairs → interference
        dist_i = dist.unsqueeze(2)  # (B, H, 1)
        dist_j = dist.unsqueeze(1)  # (B, 1, H)
        path_diff = (dist_i - dist_j).abs()  # (B, H, H)

        # Per-head-pair wave params (pairwise average preserves head diversity)
        alpha_h = F.softplus(self.wave_damping)    # (H,)
        omega_h = self.wave_frequency.abs()        # (H,)
        alpha_ij = (alpha_h.unsqueeze(0) + alpha_h.unsqueeze(1)) / 2  # (H, H)
        omega_ij = (omega_h.unsqueeze(0) + omega_h.unsqueeze(1)) / 2  # (H, H)

        # Interference: constructive when path_diff ≈ n*lambda, destructive otherwise
        interference = (
            torch.exp(-alpha_ij * path_diff)
            * torch.cos(omega_ij * path_diff)
        )  # (B, H, H)
        coupling = F.softmax(interference, dim=-1)  # (B, H, H)

        return torch.einsum('bij,bjgd->bigd', coupling, field)

    def _local_attention(self, q, k, v, N):
        """
        Sliding window causal dot-product attention (near-field).

        q, k, v: (B, H, N, head_dim)
        Returns: (B, H, N, head_dim)

        Uses precomputed causal+window mask. Masked positions get -inf
        before softmax, so they contribute zero weight.
        """
        # Scaled dot-product scores: (B, H, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply causal + window mask (slice to actual seq length)
        mask = self._window_mask[:N, :N]  # (N, N) bool
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)

    def forward(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        B, N, D = x.shape
        G = self.field_size
        H = self.num_heads
        head_dim = self.head_dim

        # Fused QKV + Gate projection (single matmul)
        qkvg = self.qkvg_proj(x)
        q, k, v, gate_raw = qkvg.chunk(4, dim=-1)

        q = q.view(B, N, H, head_dim).transpose(1, 2)
        k = k.view(B, N, H, head_dim).transpose(1, 2)
        v = v.view(B, N, H, head_dim).transpose(1, 2)

        # ================= WAVE FIELD PATH (linear-wave attention) =================
        # ABSOLUTE POSITION MAPPING — use precomputed indices (slice to actual N)
        if N <= self._cached_idx_lo.shape[0]:
            idx_lo = self._cached_idx_lo[:N]
            idx_hi = self._cached_idx_hi[:N]
            frac = self._cached_frac[:N]
        else:
            # N exceeds max_seq_len — recompute on the fly (rare, e.g. tests)
            seq_pos = torch.arange(N, device=x.device, dtype=torch.float32)
            field_pos_float = (seq_pos * self.field_stride).clamp(0, G - 2)
            idx_lo = field_pos_float.long().clamp(0, G - 2)
            idx_hi = idx_lo + 1
            frac = (field_pos_float - idx_lo.float()).clamp(0, 1)

        # V4.3: LEARNED FEATURE MAPS (Hedgehog-style)
        # At init (identity weights): φ(x) = ReLU(x) + eps — tokens are distinct
        # During training: learns spiky, softmax-mimicking maps
        q_feat = self.q_feature_map(q)  # (B, H, N, head_dim)
        k_feat = self.k_feature_map(k)  # (B, H, N, head_dim)

        # K-WEIGHTED DEPOSIT: K modulates V per dimension (D-dim routing, not scalar!)
        deposit = k_feat * v  # (B, H, N, head_dim)

        # V4.4: Selective write gate — per-token control of field contribution
        # Gate input is raw K (mean-zero), not k_feat (post-ReLU, all positive)
        if self.use_write_gate:
            write_strength = torch.sigmoid(self.write_gate_proj(k))  # (B, H, N, 1)
            deposit = deposit * write_strength

        # SCATTER → CONVOLVE → COUPLE → GATHER
        field = self._bilinear_scatter(deposit, idx_lo, idx_hi, frac, B, H, G, head_dim, x.device)

        if self.use_kernel_mixture:
            # Content-Adaptive Kernel Mixture: K basis convolutions, per-token mixing
            gathered = self._kernel_mixture_forward(field, q, idx_lo, idx_hi, frac, x)
        else:
            # V4.3 path: single kernel with SpectralGate modulation
            if self.use_analytic_kernel and self.n_components == 1:
                base_kernel_fft = self._build_analytic_kernel_fft(x.device)
            else:
                base_kernel_fft = self._build_wave_kernels(x.device)

            kernel_fft = self.spectral_gate(q, base_kernel_fft)  # (B, H, freq_bins)
            field = self._wave_convolve(field, kernel_fft)

            if self.use_3d_interference:
                field = self._apply_3d_interference(field, x)
            else:
                field = self._apply_field_coupling(field)

            gathered = self._bilinear_gather(field, idx_lo, idx_hi, frac)  # (B, H, N, head_dim)

        # Q-WEIGHTED READING: Q selects which dimensions to use
        wave_output = q_feat * gathered  # (B, H, N, head_dim)

        # ================= LOCAL ATTENTION PATH (near-field) =================
        if self.local_window > 0:
            local_out = self._local_attention(q, k, v, N)  # (B, H, N, head_dim)

            # Learned per-head blend: wave ←→ local
            blend = torch.sigmoid(self.local_blend).view(1, H, 1, 1)
            combined = (1.0 - blend) * wave_output + blend * local_out
        else:
            combined = wave_output

        # ================= GATING + OUTPUT =================
        # CONTENT-DEPENDENT GATING (from fused projection)
        gate = torch.sigmoid(gate_raw)
        gate = gate.view(B, N, H, head_dim).transpose(1, 2)

        output = combined * gate

        output = output.transpose(1, 2).reshape(B, N, D)
        output = self.out_proj(output)

        if squeeze_output:
            output = output.squeeze(0)

        return output
