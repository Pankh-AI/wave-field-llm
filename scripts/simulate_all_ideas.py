"""
Comprehensive Expressivity Simulation — ALL Novel Ideas + Hybrids
==================================================================
Compare every proposed architecture and their combinations against
standard softmax attention.

Ideas:
  1. Frequency-Bin Routing (B conv with content gates)
  2. Nonlinear Split-Step (|u|^2 * u between two FFTs)
  3. 2D Holographic Field (position x content coordinate)
  4. Wave Packet Superposition (content-dependent frequency emission)
  5. Stochastic Resonance (content-dependent noise injection)

Hybrids:
  H1. Holographic + Split-Step
  H2. Holographic + Wave Packets
  H3. Freq-Bin + Split-Step
  H4. Holographic + Freq-Bin (already tested, but refined)
  H5. Split-Step + Wave Packets

Metrics:
  - Effective rank of attention matrix
  - Output approximation error vs standard softmax
  - Content selectivity (output change when Q/K change)
  - Compute cost at N=4096
  - Memory cost at N=4096
"""

import torch
import torch.nn.functional as F
import math
import time

torch.manual_seed(42)

N = 128
D = 64
B_bins = 8
C_content = 16
F_packets = 8  # frequency components per wave packet

print("=" * 70)
print("  ALL IDEAS — EXPRESSIVITY SHOWDOWN")
print(f"  N={N}, D={D}, B={B_bins} bins, C={C_content} coords, F={F_packets} packets")
print("=" * 70)

Q = torch.randn(N, D)
K = torch.randn(N, D)
V = torch.randn(N, D)
Q2 = torch.randn(N, D)  # for selectivity test
K2 = torch.randn(N, D)

# ======================================================================
# HELPERS
# ======================================================================
def causal_toeplitz(N, alpha=0.1, omega=0.5):
    """Damped wave Toeplitz kernel (causal)."""
    K = torch.zeros(N, N)
    for i in range(N):
        for j in range(i + 1):
            dt = i - j
            K[i, j] = math.exp(-alpha * dt) * math.cos(omega * dt)
    return K

def normalize_A(A):
    """Row-normalize attention matrix."""
    row_sums = A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return A / row_sums

def evaluate(name, A, V, out_std, A_std, Q2, K2, V2_same, build_fn, cost_ratio, mem_kb):
    """Evaluate an attention matrix on all metrics."""
    rank = torch.linalg.matrix_rank(A).item()
    out = A @ V
    out_err = torch.norm(out - out_std) / torch.norm(out_std)
    a_err = torch.norm(A - A_std) / torch.norm(A_std)

    # Selectivity: rebuild with different Q, K
    A2 = build_fn(Q2, K2)
    out2 = A2 @ V
    selectivity = torch.norm(out - out2) / torch.norm(out)

    return {
        'name': name,
        'rank': rank,
        'out_err': out_err.item(),
        'a_err': a_err.item(),
        'selectivity': selectivity.item(),
        'cost': cost_ratio,
        'mem_kb': mem_kb,
    }

# ======================================================================
# STANDARD ATTENTION (ground truth)
# ======================================================================
def build_standard(Q, K):
    scores = Q @ K.T / math.sqrt(D)
    mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    return F.softmax(scores, dim=-1)

A_std = build_standard(Q, K)
out_std = A_std @ V

# ======================================================================
# 0. CURRENT WAVE FIELD
# ======================================================================
def build_wave_field(Q, K):
    torch.manual_seed(100)
    W_q = torch.randn(D, D) * 0.1
    W_k = torch.randn(D, D) * 0.1
    phi_Q = F.relu(Q @ W_q + Q)
    phi_K = F.relu(K @ W_k + K)
    A_linear = phi_Q @ phi_K.T
    K_pos = causal_toeplitz(N, 0.1, 0.5)
    return normalize_A(K_pos * A_linear)

# ======================================================================
# 1. FREQUENCY-BIN ROUTING
# ======================================================================
def build_freq_bin(Q, K, B=B_bins):
    torch.manual_seed(101)
    W_gk = torch.randn(B, D) * 0.5
    W_gq = torch.randn(B, D) * 0.5
    gate_k = F.softmax(K @ W_gk.T, dim=-1)
    gate_q = F.softmax(Q @ W_gq.T, dim=-1)

    A = torch.zeros(N, N)
    for b in range(B):
        alpha = 0.05 + 0.1 * b
        omega = 0.2 + 0.3 * b
        K_b = causal_toeplitz(N, alpha, omega)
        content_gate = gate_q[:, b:b+1] @ gate_k[:, b:b+1].T
        A += K_b * content_gate
    return normalize_A(A)

# ======================================================================
# 2. NONLINEAR SPLIT-STEP
# ======================================================================
def build_split_step(Q, K, gamma=0.3):
    """
    Split-step: linear conv -> nonlinear |u|^2 * u -> linear conv again.
    The nonlinearity makes large-signal tokens propagate differently.

    We model this as: A = K_pos * nonlinear(K_pos * A_linear)
    Where nonlinear(x) = x * (1 + gamma * x^2)  (Kerr-like)
    """
    torch.manual_seed(102)
    W_q = torch.randn(D, D) * 0.1
    W_k = torch.randn(D, D) * 0.1
    phi_Q = F.relu(Q @ W_q + Q)
    phi_K = F.relu(K @ W_k + K)
    A_linear = phi_Q @ phi_K.T

    K_pos = causal_toeplitz(N, 0.1, 0.5)

    # First linear step
    field_1 = K_pos * A_linear

    # Nonlinear step (Kerr effect: amplitude-dependent)
    field_nl = field_1 * (1.0 + gamma * field_1.abs())

    # Second linear step (different kernel — shorter range)
    K_pos2 = causal_toeplitz(N, 0.15, 0.7)
    A = K_pos2 * field_nl

    return normalize_A(A)

# ======================================================================
# 3. 2D HOLOGRAPHIC FIELD
# ======================================================================
def build_holographic(Q, K, C=C_content):
    torch.manual_seed(103)
    W_ck = torch.randn(C, D) * 0.5
    W_cq = torch.randn(C, D) * 0.5
    content_k = F.softmax(K @ W_ck.T / math.sqrt(D), dim=-1)
    content_q = F.softmax(Q @ W_cq.T / math.sqrt(D), dim=-1)

    # Content similarity: P_q @ P_k^T (rank C)
    content_sim = content_q @ content_k.T

    K_pos = causal_toeplitz(N, 0.1, 0.5)
    A = K_pos * content_sim
    return normalize_A(A)

# ======================================================================
# 4. WAVE PACKET SUPERPOSITION
# ======================================================================
def build_wave_packets(Q, K, F_comp=F_packets):
    """
    Each token emits a wave packet with content-dependent frequency spectrum.
    Key determines emission frequencies, Query determines reception frequencies.

    A_ij = sum_f  a_f(q_j) * a_f(k_i) * exp(-alpha*|i-j|) * cos(f * (i-j))

    This is F separate convolutions, each weighted by content-dependent
    frequency amplitudes.
    """
    torch.manual_seed(104)
    # Content -> frequency amplitudes
    W_freq_k = torch.randn(F_comp, D) * 0.3
    W_freq_q = torch.randn(F_comp, D) * 0.3

    # Soft frequency selection (which frequencies each token uses)
    amp_k = F.softmax(K @ W_freq_k.T, dim=-1)  # (N, F)
    amp_q = F.softmax(Q @ W_freq_q.T, dim=-1)  # (N, F)

    A = torch.zeros(N, N)
    alpha = 0.1
    for f_idx in range(F_comp):
        omega = 0.3 + 0.4 * f_idx  # spread frequencies
        K_f = causal_toeplitz(N, alpha, omega)
        # Content gating via frequency amplitude matching
        freq_gate = amp_q[:, f_idx:f_idx+1] @ amp_k[:, f_idx:f_idx+1].T
        A += K_f * freq_gate
    return normalize_A(A)

# ======================================================================
# 5. STOCHASTIC RESONANCE
# ======================================================================
def build_stochastic_resonance(Q, K, noise_scale=0.5):
    """
    Content-dependent noise injection before convolution.
    Strong signals (high |v|) survive noise, weak ones don't.
    Noise pattern depends on content -> content-dependent routing.
    """
    torch.manual_seed(105)
    W_q = torch.randn(D, D) * 0.1
    W_k = torch.randn(D, D) * 0.1
    phi_Q = F.relu(Q @ W_q + Q)
    phi_K = F.relu(K @ W_k + K)
    A_linear = phi_Q @ phi_K.T

    # Content-dependent noise: MLP(k) -> noise pattern
    W_noise = torch.randn(D, N) * 0.1
    noise_k = torch.tanh(K @ W_noise) * noise_scale  # (N, N) noise field
    noise_q = torch.tanh(Q @ W_noise) * noise_scale

    # Add noise to linear attention (content-dependent perturbation)
    A_noisy = A_linear + noise_k.T + noise_q

    # ReLU to enforce positivity (noise kills weak/wrong connections)
    A_noisy = F.relu(A_noisy)

    K_pos = causal_toeplitz(N, 0.1, 0.5)
    A = K_pos * A_noisy
    return normalize_A(A)

# ======================================================================
# HYBRIDS
# ======================================================================

# H1: Holographic + Split-Step
def build_holo_splitstep(Q, K, C=C_content, gamma=0.3):
    """2D holographic field with nonlinear propagation."""
    torch.manual_seed(106)
    W_ck = torch.randn(C, D) * 0.5
    W_cq = torch.randn(C, D) * 0.5
    content_k = F.softmax(K @ W_ck.T / math.sqrt(D), dim=-1)
    content_q = F.softmax(Q @ W_cq.T / math.sqrt(D), dim=-1)
    content_sim = content_q @ content_k.T

    K_pos = causal_toeplitz(N, 0.1, 0.5)

    # First step: position conv * content
    field_1 = K_pos * content_sim

    # Nonlinear step
    field_nl = field_1 * (1.0 + gamma * field_1.abs())

    # Second position conv (different kernel)
    K_pos2 = causal_toeplitz(N, 0.15, 0.7)
    A = K_pos2 * field_nl
    return normalize_A(A)

# H2: Holographic + Wave Packets
def build_holo_packets(Q, K, C=C_content, F_comp=F_packets):
    """2D field where content coords ALSO determine wave packet frequencies."""
    torch.manual_seed(107)
    W_ck = torch.randn(C, D) * 0.5
    W_cq = torch.randn(C, D) * 0.5
    content_k = F.softmax(K @ W_ck.T / math.sqrt(D), dim=-1)
    content_q = F.softmax(Q @ W_cq.T / math.sqrt(D), dim=-1)
    content_sim = content_q @ content_k.T

    # Wave packets: multi-frequency position kernel
    W_freq_k = torch.randn(F_comp, D) * 0.3
    W_freq_q = torch.randn(F_comp, D) * 0.3
    amp_k = F.softmax(K @ W_freq_k.T, dim=-1)
    amp_q = F.softmax(Q @ W_freq_q.T, dim=-1)

    A = torch.zeros(N, N)
    for f_idx in range(F_comp):
        omega = 0.3 + 0.4 * f_idx
        K_f = causal_toeplitz(N, 0.1, omega)
        freq_gate = amp_q[:, f_idx:f_idx+1] @ amp_k[:, f_idx:f_idx+1].T
        A += K_f * freq_gate * content_sim  # triple product

    return normalize_A(A)

# H3: Freq-Bin + Split-Step
def build_freqbin_splitstep(Q, K, B=B_bins, gamma=0.3):
    """Freq-bin routing with nonlinear step between convolutions."""
    torch.manual_seed(108)
    W_gk = torch.randn(B, D) * 0.5
    W_gq = torch.randn(B, D) * 0.5
    gate_k = F.softmax(K @ W_gk.T, dim=-1)
    gate_q = F.softmax(Q @ W_gq.T, dim=-1)

    A = torch.zeros(N, N)
    for b in range(B):
        alpha = 0.05 + 0.1 * b
        omega = 0.2 + 0.3 * b
        K_b = causal_toeplitz(N, alpha, omega)
        content_gate = gate_q[:, b:b+1] @ gate_k[:, b:b+1].T
        field_b = K_b * content_gate

        # Nonlinear step per bin
        field_b = field_b * (1.0 + gamma * field_b.abs())

        # Second conv
        K_b2 = causal_toeplitz(N, alpha * 1.5, omega * 1.2)
        A += K_b2 * field_b
    return normalize_A(A)

# H4: Holographic + Freq-Bin (refined — freq bins along content axis)
def build_holo_freqbin(Q, K, C=C_content, B=B_bins):
    """Content coords + frequency bins as ORTHOGONAL routing dimensions."""
    torch.manual_seed(109)
    W_ck = torch.randn(C, D) * 0.5
    W_cq = torch.randn(C, D) * 0.5
    content_k = F.softmax(K @ W_ck.T / math.sqrt(D), dim=-1)
    content_q = F.softmax(Q @ W_cq.T / math.sqrt(D), dim=-1)
    content_sim = content_q @ content_k.T

    W_gk = torch.randn(B, D) * 0.5
    W_gq = torch.randn(B, D) * 0.5
    gate_k = F.softmax(K @ W_gk.T, dim=-1)
    gate_q = F.softmax(Q @ W_gq.T, dim=-1)

    A = torch.zeros(N, N)
    for b in range(B):
        alpha = 0.05 + 0.1 * b
        omega = 0.2 + 0.3 * b
        K_b = causal_toeplitz(N, alpha, omega)
        freq_gate = gate_q[:, b:b+1] @ gate_k[:, b:b+1].T
        A += K_b * content_sim * freq_gate  # triple product
    return normalize_A(A)

# H5: Split-Step + Wave Packets
def build_splitstep_packets(Q, K, F_comp=F_packets, gamma=0.3):
    """Multi-frequency wave packets with nonlinear propagation."""
    torch.manual_seed(110)
    W_freq_k = torch.randn(F_comp, D) * 0.3
    W_freq_q = torch.randn(F_comp, D) * 0.3
    amp_k = F.softmax(K @ W_freq_k.T, dim=-1)
    amp_q = F.softmax(Q @ W_freq_q.T, dim=-1)

    A = torch.zeros(N, N)
    for f_idx in range(F_comp):
        omega = 0.3 + 0.4 * f_idx
        K_f = causal_toeplitz(N, 0.1, omega)
        freq_gate = amp_q[:, f_idx:f_idx+1] @ amp_k[:, f_idx:f_idx+1].T

        field_f = K_f * freq_gate
        # Nonlinear step
        field_f = field_f * (1.0 + gamma * field_f.abs())
        K_f2 = causal_toeplitz(N, 0.15, omega * 1.3)
        A += K_f2 * field_f
    return normalize_A(A)

# H6: THE KITCHEN SINK — Holographic + Split-Step + Wave Packets
def build_kitchen_sink(Q, K, C=C_content, F_comp=4, gamma=0.2):
    """Everything combined: 2D field, multi-freq, nonlinear."""
    torch.manual_seed(111)
    W_ck = torch.randn(C, D) * 0.5
    W_cq = torch.randn(C, D) * 0.5
    content_k = F.softmax(K @ W_ck.T / math.sqrt(D), dim=-1)
    content_q = F.softmax(Q @ W_cq.T / math.sqrt(D), dim=-1)
    content_sim = content_q @ content_k.T

    W_freq_k = torch.randn(F_comp, D) * 0.3
    W_freq_q = torch.randn(F_comp, D) * 0.3
    amp_k = F.softmax(K @ W_freq_k.T, dim=-1)
    amp_q = F.softmax(Q @ W_freq_q.T, dim=-1)

    A = torch.zeros(N, N)
    for f_idx in range(F_comp):
        omega = 0.3 + 0.4 * f_idx
        K_f = causal_toeplitz(N, 0.1, omega)
        freq_gate = amp_q[:, f_idx:f_idx+1] @ amp_k[:, f_idx:f_idx+1].T

        field_f = K_f * content_sim * freq_gate
        field_f = field_f * (1.0 + gamma * field_f.abs())

        K_f2 = causal_toeplitz(N, 0.15, omega * 1.3)
        A += K_f2 * field_f
    return normalize_A(A)


# ======================================================================
# RUN ALL
# ======================================================================
N_real = 4096
std_cost = N_real * N_real * D * 2
std_mem_kb = N_real * N_real * 4 / 1024

configs = [
    ("Current Wave Field",
     lambda q, k: build_wave_field(q, k),
     N_real * math.log2(N_real) * D * 2 / std_cost,
     N_real * 4 * 4 / 1024),

    ("1. Freq-Bin B=8",
     lambda q, k: build_freq_bin(q, k),
     B_bins * N_real * math.log2(N_real) * D * 2 / std_cost,
     B_bins * N_real * 4 * 4 / 1024),

    ("2. Split-Step (Kerr)",
     lambda q, k: build_split_step(q, k),
     2 * N_real * math.log2(N_real) * D * 2 / std_cost,  # 2x FFT
     N_real * 4 * 4 / 1024),

    ("3. Holographic 2D C=16",
     lambda q, k: build_holographic(q, k),
     N_real * C_content * (math.log2(N_real) + math.log2(C_content)) * D * 2 / std_cost,
     N_real * C_content * 4 / 1024),

    ("4. Wave Packets F=8",
     lambda q, k: build_wave_packets(q, k),
     F_packets * N_real * math.log2(N_real) * D * 2 / std_cost,
     F_packets * N_real * 4 * 4 / 1024),

    ("5. Stochastic Resonance",
     lambda q, k: build_stochastic_resonance(q, k),
     N_real * math.log2(N_real) * D * 2 / std_cost,  # ~same as wave
     N_real * 4 * 4 / 1024),

    ("H1. Holo + Split-Step",
     lambda q, k: build_holo_splitstep(q, k),
     2 * N_real * C_content * (math.log2(N_real) + math.log2(C_content)) * D * 2 / std_cost,
     N_real * C_content * 4 / 1024),

    ("H2. Holo + Wave Packets",
     lambda q, k: build_holo_packets(q, k),
     F_packets * N_real * C_content * (math.log2(N_real) + math.log2(C_content)) * 2 / std_cost,
     F_packets * N_real * C_content * 4 / 1024),

    ("H3. Freq-Bin + Split-Step",
     lambda q, k: build_freqbin_splitstep(q, k),
     2 * B_bins * N_real * math.log2(N_real) * D * 2 / std_cost,
     B_bins * N_real * 4 * 4 / 1024),

    ("H4. Holo + Freq-Bin",
     lambda q, k: build_holo_freqbin(q, k),
     B_bins * N_real * C_content * (math.log2(N_real) + math.log2(C_content)) * 2 / std_cost,
     B_bins * N_real * C_content * 4 / 1024),

    ("H5. Split-Step + Packets",
     lambda q, k: build_splitstep_packets(q, k),
     2 * F_packets * N_real * math.log2(N_real) * D * 2 / std_cost,
     F_packets * N_real * 4 * 4 / 1024),

    ("H6. KITCHEN SINK",
     lambda q, k: build_kitchen_sink(q, k),
     4 * N_real * C_content * (math.log2(N_real) + math.log2(C_content)) * D * 2 / std_cost,
     4 * N_real * C_content * 4 / 1024),
]

results = []
for name, build_fn, cost, mem in configs:
    t0 = time.time()
    A = build_fn(Q, K)
    dt = time.time() - t0

    rank = torch.linalg.matrix_rank(A).item()
    out = A @ V
    out_err = (torch.norm(out - out_std) / torch.norm(out_std)).item()

    # Selectivity
    A2 = build_fn(Q2, K2)
    out2 = A2 @ V
    selectivity = (torch.norm(out - out2) / torch.norm(out)).item()

    results.append({
        'name': name, 'rank': rank, 'out_err': out_err,
        'selectivity': selectivity, 'cost': cost, 'mem': mem,
        'time': dt
    })
    print(f"  [{dt:.1f}s] {name}")

# ======================================================================
# RESULTS TABLE
# ======================================================================
rank_std_val = torch.linalg.matrix_rank(A_std).item()
A_std2 = build_standard(Q2, K2)
sel_std = (torch.norm(A_std @ V - A_std2 @ V) / torch.norm(A_std @ V)).item()

print("\n" + "=" * 110)
print(f"  {'Method':<30} | {'Rank':>6} | {'OutErr':>10} | {'Select':>8} | {'Cost':>8} | {'Mem(KB)':>8} | {'Grade':>6}")
print("-" * 110)
print(f"  {'Standard Attention':<30} | {rank_std_val:>5}* | {'0.0000':>10} | {sel_std:>8.3f} | {'1.000x':>8} | {std_mem_kb:>7.0f}K | {'A+':>6}")
print("-" * 110)

# Grade based on composite score
for r in results:
    # Composite: rank/128 * selectivity * (1/cost) — higher is better
    # But also penalize high error
    rank_score = r['rank'] / N
    sel_score = min(r['selectivity'] / sel_std, 2.0)  # cap at 2x standard
    cost_score = min(1.0 / max(r['cost'], 0.001), 100)  # efficiency
    err_penalty = 1.0 / (1.0 + r['out_err'])

    composite = rank_score * sel_score * err_penalty

    if composite > 0.5:
        grade = "A"
    elif composite > 0.3:
        grade = "B+"
    elif composite > 0.15:
        grade = "B"
    elif composite > 0.05:
        grade = "C"
    else:
        grade = "D"

    # Boost grade if rank is full AND selectivity > 0.5
    if r['rank'] >= N - 5 and r['selectivity'] > 0.3:
        grade = "A" if grade >= "B" else grade

    r['grade'] = grade
    r['composite'] = composite

    print(f"  {r['name']:<30} | {r['rank']:>5}  | {r['out_err']:>10.2f} | {r['selectivity']:>8.3f} | "
          f"{r['cost']:>7.3f}x | {r['mem']:>7.0f}K | {grade:>6}")

# ======================================================================
# TOP 3 ANALYSIS
# ======================================================================
print("\n" + "=" * 110)
print("  TOP CANDIDATES (sorted by rank * selectivity, filtered cost < 0.5x)")
print("=" * 110)

viable = [r for r in results if r['cost'] < 0.5]
viable.sort(key=lambda r: r['rank'] * r['selectivity'], reverse=True)

for i, r in enumerate(viable[:5]):
    print(f"\n  #{i+1}. {r['name']}")
    print(f"      Rank: {r['rank']}/{N}  |  Selectivity: {r['selectivity']:.3f}  |  "
          f"Cost: {r['cost']:.3f}x  |  Memory: {r['mem']:.0f}KB")

    # Strengths / weaknesses
    strengths = []
    weaknesses = []
    if r['rank'] >= N - 5:
        strengths.append("Full rank (can represent diverse patterns)")
    elif r['rank'] > N // 2:
        strengths.append(f"Good rank ({r['rank']}/{N})")
    else:
        weaknesses.append(f"Low rank ({r['rank']}/{N} -- limited pattern diversity)")

    if r['selectivity'] > sel_std * 0.8:
        strengths.append("Strong content selectivity")
    elif r['selectivity'] > sel_std * 0.3:
        strengths.append("Moderate content selectivity")
    else:
        weaknesses.append("Weak content selectivity (may not route well)")

    if r['cost'] < 0.05:
        strengths.append(f"Very efficient ({r['cost']:.3f}x standard)")
    elif r['cost'] < 0.2:
        strengths.append(f"Efficient ({r['cost']:.3f}x standard)")

    if r['mem'] < 500:
        strengths.append(f"Low memory ({r['mem']:.0f}KB)")

    for s in strengths:
        print(f"      [+] {s}")
    for w in weaknesses:
        print(f"      [-] {w}")

# ======================================================================
# PARETO FRONTIER
# ======================================================================
print("\n" + "=" * 110)
print("  PARETO FRONTIER: Rank vs Cost (non-dominated solutions)")
print("=" * 110)

all_points = results.copy()
pareto = []
for r in all_points:
    dominated = False
    for other in all_points:
        if other['rank'] >= r['rank'] and other['cost'] <= r['cost'] and \
           (other['rank'] > r['rank'] or other['cost'] < r['cost']):
            dominated = True
            break
    if not dominated:
        pareto.append(r)

pareto.sort(key=lambda r: r['cost'])
print(f"\n  {'Method':<30} | {'Rank':>5} | {'Cost':>7} | {'Select':>7}")
print("  " + "-" * 60)
for r in pareto:
    marker = " <-- BEST" if r['rank'] >= N - 5 and r['cost'] < 0.15 else ""
    print(f"  {r['name']:<30} | {r['rank']:>5} | {r['cost']:>6.3f}x | {r['selectivity']:>7.3f}{marker}")

# ======================================================================
# IMPLEMENTATION RECOMMENDATION
# ======================================================================
print("\n" + "=" * 110)
print("  IMPLEMENTATION RECOMMENDATION")
print("=" * 110)

# Find best Pareto point with full rank
best = None
for r in pareto:
    if r['rank'] >= N - 5:
        if best is None or r['cost'] < best['cost']:
            best = r

if best:
    print(f"""
  WINNER: {best['name']}
    Rank:        {best['rank']}/{N} (full)
    Selectivity: {best['selectivity']:.3f} (std={sel_std:.3f})
    Cost:        {best['cost']:.3f}x standard
    Memory:      {best['mem']:.0f}KB per head

  This method achieves full rank at the lowest cost on the Pareto frontier.
  Full rank means it CAN (in principle) represent any attention pattern
  that standard attention can. Whether it LEARNS to do so depends on
  training dynamics, initialization, and the optimizer.
""")

# ======================================================================
# HYBRID RECOMMENDATION
# ======================================================================
print("  HYBRID ANALYSIS:")
print("  Which combinations actually HELP vs hurt?\n")

base_holo = [r for r in results if r['name'] == "3. Holographic 2D C=16"][0]
for r in results:
    if r['name'].startswith("H"):
        rank_delta = r['rank'] - base_holo['rank']
        sel_delta = r['selectivity'] - base_holo['selectivity']
        cost_delta = r['cost'] - base_holo['cost']
        print(f"  {r['name']:<30} vs pure Holographic:")
        print(f"    Rank: {'+' if rank_delta >= 0 else ''}{rank_delta}  |  "
              f"Select: {'+' if sel_delta >= 0 else ''}{sel_delta:.3f}  |  "
              f"Cost: +{cost_delta:.3f}x")
        verdict = "HELPS" if (rank_delta >= 0 and sel_delta > 0.05) else \
                  "MARGINAL" if (rank_delta >= 0 or sel_delta > 0) else "HURTS"
        print(f"    Verdict: {verdict}")
        print()
