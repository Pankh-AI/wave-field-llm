"""
Gated Linear Attention (GLA) — Chunk-wise Parallel Training
============================================================

Per-token content-dependent state decay that FFT convolution structurally
cannot provide. Used as drop-in replacement for wave layers at strategic
positions (middle/late layers).

Recurrence: S_t = α_t ⊙ S_{t-1} + v_t ⊗ k_t
Output:     o_t = S_t @ q_t

Reference: Yang et al., "Gated Linear Attention Transformers with
Hardware-Efficient Training" (ICML 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def chunk_gla_forward(q, k, v, alpha, chunk_size=64):
    """
    Chunk-wise parallel GLA forward pass.

    Args:
        q: (B, H, N, D) query
        k: (B, H, N, D) key
        v: (B, H, N, D) value
        alpha: (B, H, N, D) per-token forget gates in (0, 1)
        chunk_size: chunk size C for intra-chunk parallel attention

    Returns:
        output: (B, H, N, D)
    """
    B, H, N, D = q.shape
    C = chunk_size

    # Pad sequence length to multiple of chunk_size
    pad = (C - N % C) % C
    if pad > 0:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        alpha = F.pad(alpha, (0, 0, 0, pad), value=1.0)  # gate=1 means no decay for padding

    N_padded = q.shape[2]
    num_chunks = N_padded // C

    # Reshape into chunks: (B, H, nc, C, D)
    q = q.view(B, H, num_chunks, C, D)
    k = k.view(B, H, num_chunks, C, D)
    v = v.view(B, H, num_chunks, C, D)
    alpha = alpha.view(B, H, num_chunks, C, D)

    # Log-space cumulative decay for numerical stability
    log_alpha = torch.log(alpha.clamp(min=1e-6))                    # (B, H, nc, C, D)
    log_alpha_cumsum = torch.cumsum(log_alpha, dim=3)               # (B, H, nc, C, D)

    # --- Intra-chunk: O(C^2) causal attention with decay ---
    # Decay matrix: decay[i,j] = exp(mean_D(cumsum[i] - cumsum[j]))
    # Shape: (B, H, nc, C, C)
    log_alpha_cumsum_mean = log_alpha_cumsum.mean(dim=-1)           # (B, H, nc, C)
    decay_matrix = torch.exp(
        log_alpha_cumsum_mean.unsqueeze(-1) - log_alpha_cumsum_mean.unsqueeze(-2)
    )                                                                # (B, H, nc, C, C)

    # Causal mask: only attend to positions j <= i
    causal_mask = torch.tril(torch.ones(C, C, device=q.device, dtype=q.dtype))
    decay_matrix = decay_matrix * causal_mask                       # (B, H, nc, C, C)

    # Intra-chunk attention
    attn_intra = torch.matmul(q, k.transpose(-1, -2)) * decay_matrix  # (B, H, nc, C, C)
    o_intra = torch.matmul(attn_intra, v)                           # (B, H, nc, C, D)

    # --- Inter-chunk: recurrent state passing ---
    # Decay from each position j to end of its chunk
    # log_cumsum at last position minus log_cumsum at j
    log_cumsum_end = log_alpha_cumsum[:, :, :, -1:, :]              # (B, H, nc, 1, D)
    decay_to_end = torch.exp(log_cumsum_end - log_alpha_cumsum)     # (B, H, nc, C, D)

    # k weighted by decay to chunk end
    k_decayed = k * decay_to_end                                    # (B, H, nc, C, D)

    # Outer product: kv_chunk[d_k, d_v] = sum_c k_decayed[c, d_k] * v[c, d_v]
    kv_chunk = torch.einsum('bhncf,bhnce->bhnfe', k_decayed, v)    # (B, H, nc, D, D)

    # Chunk-level decay: product of all alpha in each chunk = exp(sum of log_alpha)
    chunk_decay = torch.exp(log_alpha.sum(dim=3))                   # (B, H, nc, D)

    # Decay from start of chunk to each position within chunk
    # For position i: decay = exp(cumsum[i])
    # But we need decay from chunk start (before position 0), so use cumsum directly
    decay_from_start = torch.exp(log_alpha_cumsum)                  # (B, H, nc, C, D)

    # Sequential scan across chunks for inter-chunk contribution
    S = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
    o_inter = torch.zeros_like(o_intra)

    for c_idx in range(num_chunks):
        # Query reads from accumulated state, with per-position decay
        # o_inter[i] = q[i] @ (decay_from_start[i] * S)
        # = (q[i] * decay_from_start[i]) @ S
        q_decayed = q[:, :, c_idx, :, :] * decay_from_start[:, :, c_idx, :, :]  # (B, H, C, D)
        o_inter[:, :, c_idx, :, :] = torch.einsum(
            'bhcd,bhde->bhce', q_decayed, S
        )                                                            # (B, H, C, D)

        # Update state: S = S * chunk_decay + kv_chunk
        S = S * chunk_decay[:, :, c_idx, :].unsqueeze(-1) + kv_chunk[:, :, c_idx, :, :]

    # Combine intra + inter
    output = o_intra + o_inter                                       # (B, H, nc, C, D)

    # Reshape back and remove padding
    output = output.reshape(B, H, N_padded, D)
    if pad > 0:
        output = output[:, :, :N, :]

    return output


class GLALayer(nn.Module):
    """
    Gated Linear Attention layer with pre-norm residual + FFN.

    Drop-in replacement for WaveFieldTransformerLayer at strategic positions.
    Provides per-token content-dependent state decay (what FFT conv cannot do).

    Args:
        d_model: model dimension
        num_heads: number of attention heads
        d_head: per-head dimension (default: d_model // num_heads)
        ffn_dim: FFN hidden dimension (default: 4 * d_model)
        dropout: dropout rate
        chunk_size: chunk size for parallel training
    """

    def __init__(self, d_model, num_heads, d_head=None, ffn_dim=None,
                 dropout=0.1, chunk_size=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head or d_model // num_heads
        self.ffn_dim = ffn_dim or 4 * d_model
        self.chunk_size = chunk_size

        assert d_model % num_heads == 0 or d_head is not None, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        total_head_dim = self.num_heads * self.d_head

        # QKV projections (no bias, standard practice)
        self.q_proj = nn.Linear(d_model, total_head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, total_head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, total_head_dim, bias=False)

        # Forget gate projection (with bias for init control)
        self.g_proj = nn.Linear(d_model, total_head_dim, bias=True)

        # Output projection
        self.o_proj = nn.Linear(total_head_dim, d_model, bias=False)

        # Pre-norm for attention and FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN (same structure as wave layers)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        self.attn_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        # Xavier uniform for projections
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)

        # Gate bias: initialize slightly negative so sigmoid → ~0.3-0.5
        # This means moderate decay (not too aggressive, not too passive)
        nn.init.xavier_uniform_(self.g_proj.weight)
        nn.init.constant_(self.g_proj.bias, -0.5)

        # FFN init
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, d_model)
            mask: accepted but ignored (GLA is causal by construction)

        Returns:
            (B, N, d_model)
        """
        B, N, _ = x.shape
        H = self.num_heads
        D = self.d_head

        # Pre-norm attention
        x_norm = self.norm1(x)

        # Project Q, K, V, gate
        q = self.q_proj(x_norm).view(B, N, H, D).transpose(1, 2)   # (B, H, N, D)
        k = self.k_proj(x_norm).view(B, N, H, D).transpose(1, 2)   # (B, H, N, D)
        v = self.v_proj(x_norm).view(B, N, H, D).transpose(1, 2)   # (B, H, N, D)

        # L2 normalize Q and K for stable attention (no exploding dots)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Forget gate: sigmoid → (0, 1)
        alpha = torch.sigmoid(self.g_proj(x_norm))                   # (B, N, total_head_dim)
        alpha = alpha.view(B, N, H, D).transpose(1, 2)              # (B, H, N, D)

        # Chunk-wise GLA forward
        o = chunk_gla_forward(q, k, v, alpha, chunk_size=self.chunk_size)  # (B, H, N, D)

        # Reshape and project output
        o = o.transpose(1, 2).contiguous().view(B, N, H * D)        # (B, N, total_head_dim)
        attn_out = self.o_proj(o)                                    # (B, N, d_model)
        attn_out = self.attn_dropout(attn_out)

        # Residual connection
        x = x + attn_out

        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x


if __name__ == '__main__':
    """Smoke test: shape check + basic causality verification."""
    import sys

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing GLALayer on {device}")

    # Shape test
    m = GLALayer(384, 8).to(device)
    x = torch.randn(2, 64, 384, device=device)
    y = m(x)
    print(f"Input: {x.shape}, Output: {y.shape}, OK: {x.shape == y.shape}")
    assert x.shape == y.shape, "Shape mismatch!"

    # Param count
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Parameters: {n_params:,}")

    # Causality test
    m.eval()
    x = torch.randn(1, 32, 384, device=device)
    with torch.no_grad():
        out_full = m(x)
        out_trunc = m(x[:, :16, :])
        diff = (out_full[:, :16, :] - out_trunc).abs().max().item()
        print(f"Causality check: max diff = {diff:.2e} (should be < 1e-4)")
        assert diff < 1e-4, f"CAUSAL LEAK: {diff}"
        print("PASSED: causal, correct shapes")
