"""
Wave Field Attention V3.5 - Generation Fix
============================================

THREE critical fixes for autoregressive generation:

1. ABSOLUTE POSITION MAPPING: Token i always maps to field position
   i * stride, regardless of sequence length. No more position shifting
   during generation (the root cause of garbage output in V3.0-V3.4).

2. LEFT-ALIGNED CAUSAL KERNEL: Kernel starts at position 0 (current)
   and decays backward. No center offset. Works correctly with absolute
   positions where tokens are packed in the left part of the field.

3. NO ENERGY CONSERVATION: Removed. With absolute positions, short
   sequences leave most of the field empty. Conservation was rescaling
   signal to match near-zero empty-field energy → crushing information.
   Residual connections + LayerNorm handle normalization instead.

Physics (unchanged):
- Damped wave kernel: k(t) = exp(-α*t) * cos(ω*t + φ) for t>=0
- Different heads = different fields with different wave speeds
- Static coupling = cross-head field interactions

Complexity: O(n log n) per head via FFT convolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WaveFieldAttention(nn.Module):
    
    def __init__(self, embedding_dim, num_heads, field_size=512, max_seq_len=128, device='cuda'):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.field_size = field_size
        self.max_seq_len = max_seq_len
        self.device = device
        
        assert embedding_dim % num_heads == 0
        
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # INNOVATION 1: Wave-parameterized kernels (per head)
        self.wave_frequency = nn.Parameter(
            torch.linspace(0.3, 4.0, num_heads)
        )
        self.wave_damping = nn.Parameter(
            torch.linspace(-3.0, 0.5, num_heads)
        )
        self.wave_phase = nn.Parameter(
            torch.linspace(0, math.pi, num_heads)
        )
        
        # INNOVATION 2: Content-dependent gating (bias=2.0, starts open)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)
        
        # INNOVATION 3: Static multi-field coupling
        self.field_coupling = nn.Parameter(
            torch.eye(num_heads) + torch.randn(num_heads, num_heads) * 0.01
        )
        
        # Fixed stride for absolute position mapping
        self.register_buffer(
            'field_stride',
            torch.tensor((field_size - 1) / max(max_seq_len - 1, 1), dtype=torch.float32)
        )
        
        self.scale = math.sqrt(self.head_dim)
    
    def _build_wave_kernels(self, device):
        """
        Build LEFT-ALIGNED causal wave kernels.
        
        V3.5: kernel[0] = current position, kernel[1] = 1 step back, etc.
        No center offset. Works with absolute positions where tokens
        are packed in the left portion of the field.
        """
        G = self.field_size
        H = self.num_heads
        
        t = torch.arange(G, dtype=torch.float32, device=device)
        
        alpha = F.softplus(self.wave_damping).unsqueeze(1)
        omega = self.wave_frequency.unsqueeze(1)
        phi = self.wave_phase.unsqueeze(1)
        
        # Left-aligned: position 0 = current, position 1 = 1 step back, ...
        kernels = torch.exp(-alpha * t.unsqueeze(0)) * torch.cos(omega * t.unsqueeze(0) + phi)
        
        kernel_sum = kernels.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
        kernels = kernels / kernel_sum
        
        return torch.fft.rfft(kernels, n=2 * G)
    
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
        w_lo = (1.0 - frac).view(1, 1, N, 1)
        w_hi = frac.view(1, 1, N, 1)
        
        field = torch.zeros(B, H, G, head_dim, device=device)
        
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
        """Static multi-field coupling."""
        B, H, G, D = field.shape
        coupling = F.softmax(self.field_coupling, dim=-1)
        field_flat = field.reshape(B, H, G * D)
        coupling_exp = coupling.unsqueeze(0).expand(B, -1, -1)
        coupled = torch.bmm(coupling_exp, field_flat)
        return coupled.reshape(B, H, G, D)
    
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
        
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, N, H, head_dim).transpose(1, 2)
        k = k.view(B, N, H, head_dim).transpose(1, 2)
        v = v.view(B, N, H, head_dim).transpose(1, 2)
        
        # ABSOLUTE POSITION MAPPING (V3.5)
        # Token i ALWAYS at field position i * stride, regardless of N
        seq_pos = torch.arange(N, device=x.device, dtype=torch.float32)
        field_pos_float = (seq_pos * self.field_stride).clamp(0, G - 2)
        
        # BILINEAR SCATTER
        k_mag = k.norm(dim=-1, keepdim=True)
        deposit = v * k_mag
        field = self._bilinear_scatter(deposit, field_pos_float, B, H, G, head_dim, x.device)
        
        # WAVE CONVOLUTION
        kernel_fft = self._build_wave_kernels(x.device)
        field = self._wave_convolve(field, kernel_fft)
        
        # STATIC COUPLING
        field = self._apply_field_coupling(field)
        
        # NO ENERGY CONSERVATION (V3.5: removed — destabilizes sparse fields)
        
        # CONTENT-DEPENDENT GATING
        gate = torch.sigmoid(self.gate_proj(x))
        gate = gate.view(B, N, H, head_dim).transpose(1, 2)
        
        # BILINEAR GATHER
        gathered = self._bilinear_gather(field, field_pos_float)
        
        output = gathered * gate
        
        output = output.transpose(1, 2).reshape(B, N, D)
        output = self.out_proj(output)
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output
