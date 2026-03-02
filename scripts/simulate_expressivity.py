"""
Mathematical Expressivity Simulation
=====================================
Compare attention matrix approximation quality of:
  1. Current Wave Field (1D Toeplitz + per-dim feature maps)
  2. Frequency-Bin Routing (B separate convolutions, content-gated)
  3. 2D Holographic Field (position × content coordinate)
  4. Holographic + Freq-Bin (combined: Idea 3 + Idea 1)

Metric: How well can each approximate standard softmax attention?
  - Frobenius norm error vs softmax A
  - Effective rank of attention matrix
  - Content selectivity (can it attend differently based on content?)

No training. Pure linear algebra on random Q, K, V.
"""

import torch
import torch.nn.functional as F
import math
import numpy as np

torch.manual_seed(42)

# ======================================================================
# CONFIG
# ======================================================================
N = 128       # sequence length
D = 64        # head dimension
H = 8         # num heads
B_bins = 8    # frequency bins for Idea 1
C_content = 16  # content coordinates for Idea 3

print("=" * 70)
print("  EXPRESSIVITY SIMULATION")
print(f"  N={N} tokens, D={D} dims/head, H={H} heads")
print(f"  Freq bins B={B_bins}, Content coords C={C_content}")
print("=" * 70)

# ======================================================================
# Generate random Q, K, V (simulating one head)
# ======================================================================
Q = torch.randn(N, D)
K = torch.randn(N, D)
V = torch.randn(N, D)

# ======================================================================
# 1. STANDARD SOFTMAX ATTENTION (ground truth)
# ======================================================================
def standard_attention(Q, K, V):
    """O(n²) softmax attention — the gold standard."""
    scores = Q @ K.T / math.sqrt(D)
    # causal mask
    mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    A = F.softmax(scores, dim=-1)
    return A, A @ V

A_std, out_std = standard_attention(Q, K, V)
rank_std = torch.linalg.matrix_rank(A_std).item()
print(f"\n[Standard Attention]")
print(f"  Rank of A: {rank_std}/{N}")
print(f"  A sparsity (>0.01): {(A_std > 0.01).float().mean():.3f}")

# ======================================================================
# 2. CURRENT WAVE FIELD (1D Toeplitz + per-dim feature maps)
# ======================================================================
def wave_field_attention(Q, K, V, n_kernels=1):
    """
    Current architecture:
      A_ij = k_wave(i-j) * Σ_d φ(q_i)_d * φ(k_j)_d

    k_wave is a damped sinusoid (Toeplitz).
    φ = ReLU (learned feature map, but random here).
    """
    # Feature maps: ReLU (Hedgehog-style)
    W_q = torch.randn(D, D) * 0.1
    W_k = torch.randn(D, D) * 0.1
    phi_Q = F.relu(Q @ W_q + Q)  # identity + learned
    phi_K = F.relu(K @ W_k + K)

    # Linear attention: per-dim outer product
    # A_linear[i,j] = Σ_d φ(q_i)_d * φ(k_j)_d
    A_linear = phi_Q @ phi_K.T

    # Toeplitz kernel (damped wave)
    positions = torch.arange(N).float()
    # Simulate 1 damped sinusoidal kernel
    alpha = 0.1  # damping
    omega = 0.5  # frequency
    kernel_matrix = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            if j <= i:  # causal
                dt = i - j
                kernel_matrix[i, j] = math.exp(-alpha * dt) * math.cos(omega * dt)

    # Combined: element-wise product
    A_wave = kernel_matrix * A_linear

    # Normalize rows
    row_sums = A_wave.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    A_wave = A_wave / row_sums

    return A_wave, A_wave @ V

A_wave, out_wave = wave_field_attention(Q, K, V)
rank_wave = torch.linalg.matrix_rank(A_wave).item()
err_wave = torch.norm(out_wave - out_std) / torch.norm(out_std)
# How well does A_wave approximate A_std?
a_err_wave = torch.norm(A_wave - A_std) / torch.norm(A_std)

print(f"\n[Current Wave Field — 1D Toeplitz + per-dim]")
print(f"  Rank of A: {rank_wave}/{N}")
print(f"  Output relative error vs standard: {err_wave:.4f}")
print(f"  Attention matrix error vs standard: {a_err_wave:.4f}")

# ======================================================================
# 3. FREQUENCY-BIN ROUTING (Idea 1)
# ======================================================================
def freq_bin_routing(Q, K, V, B=8):
    """
    B separate convolutions, content-gated.

    Each token selects which frequency bins to write/read via content:
      g_b(k_i) = softmax(W_gate_k @ k_i)[b]  — write gate
      r_b(q_j) = softmax(W_gate_q @ q_j)[b]  — read gate

    For each bin b:
      field_b = Σ_i g_b(k_i) * v_i * kernel_b(pos)
      output_j += r_b(q_j) * gather(field_b, pos_j)

    Effective attention:
      A_ij = Σ_b r_b(q_j) * g_b(k_i) * kernel_b(j-i)
    """
    # Content-dependent gates
    W_gate_k = torch.randn(B, D) * 0.5
    W_gate_q = torch.randn(B, D) * 0.5

    gate_k = F.softmax(K @ W_gate_k.T, dim=-1)  # (N, B)
    gate_q = F.softmax(Q @ W_gate_q.T, dim=-1)  # (N, B)

    # B different Toeplitz kernels (different frequencies)
    A_total = torch.zeros(N, N)
    for b in range(B):
        alpha = 0.05 + 0.1 * b  # different damping per bin
        omega = 0.2 + 0.3 * b   # different frequency per bin

        kernel_b = torch.zeros(N, N)
        for i in range(N):
            for j in range(N):
                if j <= i:
                    dt = i - j
                    kernel_b[i, j] = math.exp(-alpha * dt) * math.cos(omega * dt)

        # Content gating: outer product of read × write gates
        content_gate = gate_q[:, b:b+1] @ gate_k[:, b:b+1].T  # (N, N)
        A_total += kernel_b * content_gate

    # Normalize
    row_sums = A_total.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    A_total = A_total / row_sums

    return A_total, A_total @ V

A_freq, out_freq = freq_bin_routing(Q, K, V, B=B_bins)
rank_freq = torch.linalg.matrix_rank(A_freq).item()
err_freq = torch.norm(out_freq - out_std) / torch.norm(out_std)
a_err_freq = torch.norm(A_freq - A_std) / torch.norm(A_std)

print(f"\n[Frequency-Bin Routing — B={B_bins} bins]")
print(f"  Rank of A: {rank_freq}/{N}")
print(f"  Output relative error vs standard: {err_freq:.4f}")
print(f"  Attention matrix error vs standard: {a_err_freq:.4f}")

# ======================================================================
# 4. 2D HOLOGRAPHIC FIELD (Idea 3)
# ======================================================================
def holographic_2d_field(Q, K, V, C=16):
    """
    2D field: axis 1 = position, axis 2 = content coordinate.

    Each token maps to a content coordinate via learned projection:
      content_i = softmax(W_content @ k_i) → soft assignment over C bins

    Deposit onto 2D field, convolve with 2D kernel, gather.

    Effective attention:
      A_ij = Σ_c Σ_c' kernel_pos(j-i) * kernel_content(c-c') *
             p_content(q_j, c') * p_content(k_i, c)

    The 2D kernel naturally routes by BOTH position AND content distance.
    """
    # Content coordinate (soft assignment)
    W_content_k = torch.randn(C, D) * 0.5
    W_content_q = torch.randn(C, D) * 0.5

    # Soft content coordinates
    content_k = F.softmax(K @ W_content_k.T / math.sqrt(D), dim=-1)  # (N, C)
    content_q = F.softmax(Q @ W_content_q.T / math.sqrt(D), dim=-1)  # (N, C)

    # 2D kernel: position kernel × content kernel
    # Position kernel: damped wave (same as before)
    alpha_pos = 0.1
    omega_pos = 0.5

    # Content kernel: Gaussian-like (nearby content coords interact)
    sigma_content = 2.0

    A_total = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            if j <= i:  # causal
                dt = i - j
                k_pos = math.exp(-alpha_pos * dt) * math.cos(omega_pos * dt)

                # Content similarity: dot product of content assignments
                # Tokens in same content region interact strongly
                k_content = 0.0
                for c in range(C):
                    for c2 in range(C):
                        dc = abs(c - c2)
                        gaussian = math.exp(-dc**2 / (2 * sigma_content**2))
                        k_content += content_q[j, c2].item() * content_k[i, c].item() * gaussian

                A_total[i, j] = k_pos * k_content

    # Normalize
    row_sums = A_total.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    A_total = A_total / row_sums

    return A_total, A_total @ V

A_holo, out_holo = holographic_2d_field(Q, K, V, C=C_content)
rank_holo = torch.linalg.matrix_rank(A_holo).item()
err_holo = torch.norm(out_holo - out_std) / torch.norm(out_std)
a_err_holo = torch.norm(A_holo - A_std) / torch.norm(A_std)

print(f"\n[2D Holographic Field — C={C_content} content coords]")
print(f"  Rank of A: {rank_holo}/{N}")
print(f"  Output relative error vs standard: {err_holo:.4f}")
print(f"  Attention matrix error vs standard: {a_err_holo:.4f}")

# ======================================================================
# 5. HOLOGRAPHIC + FREQ-BIN (Idea 3 + Idea 1)
# ======================================================================
def holographic_freq_bin(Q, K, V, C=16, B=8):
    """
    Combined: 2D holographic field with frequency-bin routing.

    - Content coordinate determines WHICH 2D sub-field to use
    - Frequency bins within each sub-field provide spectral diversity
    - Each bin has its own 2D kernel

    A_ij = Σ_b Σ_c  r_b(q_j) * g_b(k_i) *
           p_c(q_j) * p_c(k_i) * kernel_b(j-i) * gauss(c_j - c_i)
    """
    # Content gates (which content region)
    W_ck = torch.randn(C, D) * 0.5
    W_cq = torch.randn(C, D) * 0.5
    content_k = F.softmax(K @ W_ck.T / math.sqrt(D), dim=-1)  # (N, C)
    content_q = F.softmax(Q @ W_cq.T / math.sqrt(D), dim=-1)  # (N, C)

    # Frequency bin gates
    W_fk = torch.randn(B, D) * 0.5
    W_fq = torch.randn(B, D) * 0.5
    freq_k = F.softmax(K @ W_fk.T, dim=-1)  # (N, B)
    freq_q = F.softmax(Q @ W_fq.T, dim=-1)  # (N, B)

    sigma_c = 2.0

    A_total = torch.zeros(N, N)
    for b in range(B):
        alpha = 0.05 + 0.1 * b
        omega = 0.2 + 0.3 * b

        for i in range(N):
            for j in range(i + 1):  # causal
                dt = i - j
                k_pos = math.exp(-alpha * dt) * math.cos(omega * dt)

                # Content similarity
                k_content = 0.0
                for c in range(C):
                    for c2 in range(C):
                        dc = abs(c - c2)
                        gaussian = math.exp(-dc**2 / (2 * sigma_c**2))
                        k_content += content_q[j, c2].item() * content_k[i, c].item() * gaussian

                # Frequency gating
                f_gate = freq_q[j, b].item() * freq_k[i, b].item()

                A_total[i, j] += k_pos * k_content * f_gate

    # Normalize
    row_sums = A_total.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    A_total = A_total / row_sums

    return A_total, A_total @ V

A_combo, out_combo = holographic_freq_bin(Q, K, V, C=C_content, B=B_bins)
rank_combo = torch.linalg.matrix_rank(A_combo).item()
err_combo = torch.norm(out_combo - out_std) / torch.norm(out_std)
a_err_combo = torch.norm(A_combo - A_std) / torch.norm(A_std)

print(f"\n[Holographic + Freq-Bin — C={C_content}, B={B_bins}]")
print(f"  Rank of A: {rank_combo}/{N}")
print(f"  Output relative error vs standard: {err_combo:.4f}")
print(f"  Attention matrix error vs standard: {a_err_combo:.4f}")

# ======================================================================
# 6. CONTENT SELECTIVITY TEST
# ======================================================================
print("\n" + "=" * 70)
print("  CONTENT SELECTIVITY TEST")
print("  Can the method attend differently when content changes?")
print("=" * 70)

# Create two scenarios: same positions, different content
Q2 = torch.randn(N, D)  # different Q
K2 = torch.randn(N, D)  # different K

_, out_std_2 = standard_attention(Q2, K2, V)
std_diff = torch.norm(out_std - out_std_2) / torch.norm(out_std)

A_wave2, out_wave2 = wave_field_attention(Q2, K2, V)
wave_diff = torch.norm(out_wave - out_wave2) / torch.norm(out_wave)

A_freq2, out_freq2 = freq_bin_routing(Q2, K2, V, B=B_bins)
freq_diff = torch.norm(out_freq - out_freq2) / torch.norm(out_freq)

A_holo2, out_holo2 = holographic_2d_field(Q2, K2, V, C=C_content)
holo_diff = torch.norm(out_holo - out_holo2) / torch.norm(out_holo)

A_combo2, out_combo2 = holographic_freq_bin(Q2, K2, V, C=C_content, B=B_bins)
combo_diff = torch.norm(out_combo - out_combo2) / torch.norm(out_combo)

print(f"\n  Output change when content changes (higher = more selective):")
print(f"  Standard:          {std_diff:.4f}")
print(f"  Wave Field:        {wave_diff:.4f}")
print(f"  Freq-Bin:          {freq_diff:.4f}")
print(f"  Holographic 2D:    {holo_diff:.4f}")
print(f"  Holo + Freq-Bin:   {combo_diff:.4f}")

# ======================================================================
# 7. COMPUTE COST ANALYSIS
# ======================================================================
print("\n" + "=" * 70)
print("  COMPUTE COST ANALYSIS (for N=4096)")
print("=" * 70)

N_real = 4096
D_real = 64
H_real = 8

std_flops = N_real * N_real * D_real * 2  # QK^T + AV
wave_flops = N_real * math.log2(N_real) * D_real * 2  # FFT conv
freq_flops = B_bins * N_real * math.log2(N_real) * D_real * 2
holo_flops = N_real * C_content * math.log2(N_real * C_content) * D_real * 2
combo_flops = B_bins * N_real * C_content * math.log2(N_real * C_content) * 2

print(f"\n  Per-head FLOPs (approx):")
print(f"  Standard:          {std_flops/1e6:>10.1f}M  (1.00x)")
print(f"  Wave Field:        {wave_flops/1e6:>10.1f}M  ({wave_flops/std_flops:.2f}x)")
print(f"  Freq-Bin B={B_bins}:      {freq_flops/1e6:>10.1f}M  ({freq_flops/std_flops:.2f}x)")
print(f"  Holographic C={C_content}:   {holo_flops/1e6:>10.1f}M  ({holo_flops/std_flops:.2f}x)")
print(f"  Holo+Freq B={B_bins},C={C_content}: {combo_flops/1e6:>10.1f}M  ({combo_flops/std_flops:.2f}x)")

# Memory (field size)
std_mem = N_real * N_real  # attention matrix
wave_mem = N_real * 4  # 1D field with 4x padding
freq_mem = B_bins * N_real * 4
holo_mem = N_real * C_content * 4
combo_mem = B_bins * N_real * C_content * 4

print(f"\n  Per-head memory (field/attention matrix):")
print(f"  Standard:          {std_mem * 4 / 1024:>10.1f}KB")
print(f"  Wave Field:        {wave_mem * 4 / 1024:>10.1f}KB")
print(f"  Freq-Bin:          {freq_mem * 4 / 1024:>10.1f}KB")
print(f"  Holographic:       {holo_mem * 4 / 1024:>10.1f}KB")
print(f"  Holo+Freq:         {combo_mem * 4 / 1024:>10.1f}KB")

# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"""
  Method              | Rank | A_err | Out_err | Selectivity | Cost
  --------------------|------|-------|---------|-------------|------
  Standard (target)   | {rank_std:>4} | 0.000 |  0.000  |  {std_diff:.3f}      | 1.00x
  Wave Field (current)| {rank_wave:>4} | {a_err_wave:.3f} |  {err_wave:.3f}  |  {wave_diff:.3f}      | {wave_flops/std_flops:.2f}x
  Freq-Bin B={B_bins}       | {rank_freq:>4} | {a_err_freq:.3f} |  {err_freq:.3f}  |  {freq_diff:.3f}      | {freq_flops/std_flops:.2f}x
  Holographic C={C_content}    | {rank_holo:>4} | {a_err_holo:.3f} |  {err_holo:.3f}  |  {holo_diff:.3f}      | {holo_flops/std_flops:.2f}x
  Holo+Freq           | {rank_combo:>4} | {a_err_combo:.3f} |  {err_combo:.3f}  |  {combo_diff:.3f}      | {combo_flops/std_flops:.2f}x

  Lower A_err = better approximation of softmax attention
  Higher Selectivity = more content-dependent routing
  Lower Cost = more efficient
""")
