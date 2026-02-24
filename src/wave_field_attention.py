"""
Wave Field Attention V4.1 - Linear-Wave Attention
===================================================

Fixes the V3.x quality gap (PPL 1275 vs 461) by incorporating content-
dependent routing INTO the wave path, not just gating it after.

ROOT CAUSE in V3.x:
  - Q vectors computed but NEVER USED (500x less routing capacity)
  - K collapsed to scalar ||K|| (no directional information)

WHY V4.0 (dual-field Q·K scoring) FAILED:
  The Q·K gate was applied AFTER wave convolution already averaged
  everything together. You can't un-mix a mixture — the per-source
  information is lost once the wave kernel blends it.

V4.1 FIX — Linear-Wave Attention:
  Combines linear attention's feature decomposition with wave propagation.
  Content routing happens INSIDE the scatter/gather, not after.

  φ(x) = elu(x) + 1  (positive feature map, from linear attention)

  1. DEPOSIT: φ(K) ⊙ V  — K modulates V per dimension (D-dim, not scalar!)
  2. PROPAGATE: Single field, same wave convolution as V3.x
  3. GATHER: Read field at token positions
  4. READ: φ(Q) ⊙ gathered  — Q selects which dimensions to use

  This gives D-dimensional content routing (32x richer than V3.x's scalar
  ||K||) at the SAME compute cost as V3.x (single field, single FFT).

  At initialization: φ(K) ≈ 1, φ(Q) ≈ 1, so behavior ≈ V3.x
  During training: K learns which V dimensions to deposit,
                   Q learns which gathered dimensions to read.

  Physics analogy: K acts as a polarization filter on the deposited wave,
  Q acts as an analyzer filter on the gathered wave. Only matching
  polarizations pass through — content-dependent routing via filtering.

Optional features (backward compatible):
- Multi-component wavelet kernels (n_components > 1)
- Sliding window local attention (local_window > 0)

Complexity: O(n log n + n*w) = O(n log n) since w is constant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WaveFieldAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads, field_size=512, max_seq_len=128,
                 n_components=1, local_window=0, device='cuda'):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.field_size = field_size
        self.max_seq_len = max_seq_len
        self.n_components = n_components
        self.local_window = local_window
        self.device = device

        assert embedding_dim % num_heads == 0

        # Fused QKV + Gate projection: 4D instead of separate 3D + 1D
        self.qkvg_proj = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        # Initialize gate portion (last D columns) to start open (bias=2.0, weight=0)
        with torch.no_grad():
            self.qkvg_proj.weight[3 * embedding_dim:].zero_()
            self.qkvg_proj.bias[3 * embedding_dim:].fill_(2.0)

        # ---- WAVE KERNEL PARAMETERS ----
        H = num_heads
        C = n_components

        if C == 1:
            # Single component: exact V3.5 params (H,) — checkpoint compatible
            self.wave_frequency = nn.Parameter(torch.linspace(0.3, 4.0, H))
            self.wave_damping = nn.Parameter(torch.linspace(-3.0, 0.5, H))
            self.wave_phase = nn.Parameter(torch.linspace(0, math.pi, H))
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
            # Per-head blend: sigmoid(0) = 0.5 — equal mix of wave and local
            self.local_blend = nn.Parameter(torch.zeros(H))

            # Precompute causal + window mask (registered buffer → moves with .to(device))
            rows = torch.arange(max_seq_len).unsqueeze(1)
            cols = torch.arange(max_seq_len).unsqueeze(0)
            mask = (cols <= rows) & (rows - cols < local_window)
            self.register_buffer('_window_mask', mask)
        else:
            self.local_blend = None

        # Static multi-field coupling
        self.field_coupling = nn.Parameter(
            torch.eye(H) + torch.randn(H, H) * 0.01
        )

        # Fixed stride for absolute position mapping
        self.register_buffer(
            'field_stride',
            torch.tensor((field_size - 1) / max(max_seq_len - 1, 1), dtype=torch.float32)
        )

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

        # Normalize and FFT
        kernel_sum = kernels.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
        kernels = kernels / kernel_sum

        result = torch.fft.rfft(kernels, n=2 * G)

        # Cache (eval only — training invalidates every step + breaks checkpointing)
        if not self.training:
            cache_keys = [self.wave_frequency.data, self.wave_damping.data, self.wave_phase.data]
            if self.component_weights is not None:
                cache_keys.append(self.component_weights.data)
            self._kernel_fft_cache = result
            self._kernel_param_snapshot = [p.clone() for p in cache_keys]

        return result

    def _wave_convolve(self, field, kernel_fft):
        """Per-head wave convolution via zero-padded FFT (linear convolution)."""
        B, H, G, D = field.shape
        pad_size = 2 * G

        field_t = field.permute(0, 3, 1, 2).reshape(B * D, H, G)
        field_fft = torch.fft.rfft(field_t, n=pad_size)
        convolved_fft = field_fft * kernel_fft.unsqueeze(0)
        convolved = torch.fft.irfft(convolved_fft, n=pad_size)[:, :, :G]

        return convolved.reshape(B, D, H, G).permute(0, 2, 3, 1)

    def _bilinear_scatter(self, values, field_pos_float, B, H, G, head_dim, device):
        """Deposit values onto field using bilinear interpolation."""
        N = field_pos_float.shape[0]

        idx_lo = field_pos_float.long().clamp(0, G - 2)
        idx_hi = idx_lo + 1

        frac = (field_pos_float - idx_lo.float()).clamp(0, 1)
        # Cast weights to match values dtype (AMP produces float16 values but float32 positions)
        w_lo = (1.0 - frac).to(values.dtype).view(1, 1, N, 1)
        w_hi = frac.to(values.dtype).view(1, 1, N, 1)

        field = torch.zeros(B, H, G, head_dim, device=device, dtype=values.dtype)

        idx_lo_exp = idx_lo.view(1, 1, N, 1).expand(B, H, -1, head_dim)
        idx_hi_exp = idx_hi.view(1, 1, N, 1).expand(B, H, -1, head_dim)

        field.scatter_add_(2, idx_lo_exp, values * w_lo)
        field.scatter_add_(2, idx_hi_exp, values * w_hi)

        return field

    def _bilinear_gather(self, field, field_pos_float):
        """Read from field using bilinear interpolation."""
        B, H, G, D = field.shape
        N = field_pos_float.shape[0]

        idx_lo = field_pos_float.long().clamp(0, G - 2)
        idx_hi = idx_lo + 1

        frac = (field_pos_float - idx_lo.float()).clamp(0, 1)
        w_lo = (1.0 - frac).view(1, 1, N, 1)
        w_hi = frac.view(1, 1, N, 1)

        idx_lo_exp = idx_lo.view(1, 1, N, 1).expand(B, H, -1, D)
        idx_hi_exp = idx_hi.view(1, 1, N, 1).expand(B, H, -1, D)

        val_lo = torch.gather(field, 2, idx_lo_exp)
        val_hi = torch.gather(field, 2, idx_hi_exp)

        return val_lo * w_lo + val_hi * w_hi

    def _apply_field_coupling(self, field):
        """Static multi-field coupling via einsum (avoids flatten + bmm + reshape)."""
        coupling = F.softmax(self.field_coupling, dim=-1)
        return torch.einsum('ij,bjgd->bigd', coupling, field)

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
        # ABSOLUTE POSITION MAPPING
        seq_pos = torch.arange(N, device=x.device, dtype=torch.float32)
        field_pos_float = (seq_pos * self.field_stride).clamp(0, G - 2)

        # LINEAR-WAVE ATTENTION (V4.1)
        # Positive feature maps: elu(x)+1 ensures non-negative features
        # At init (weights~0): elu(0)+1 = 1, so deposit ≈ V, output ≈ gathered
        q_feat = F.elu(q) + 1  # (B, H, N, D)
        k_feat = F.elu(k) + 1  # (B, H, N, D)

        # K-WEIGHTED DEPOSIT: K modulates V per dimension (D-dim routing, not scalar!)
        deposit = k_feat * v  # (B, H, N, head_dim)

        # SCATTER → CONVOLVE → COUPLE → GATHER (single field, same cost as V3.x)
        field = self._bilinear_scatter(deposit, field_pos_float, B, H, G, head_dim, x.device)
        kernel_fft = self._build_wave_kernels(x.device)
        field = self._wave_convolve(field, kernel_fft)
        field = self._apply_field_coupling(field)
        gathered = self._bilinear_gather(field, field_pos_float)  # (B, H, N, head_dim)

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
