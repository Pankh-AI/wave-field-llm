# V4.3 SPECTRE-Wave Hybrid Design

## Problem

Wave Field LLM V4.1 achieves PPL 543 at 15M tokens. Standard Transformer achieves PPL 473 at 5M tokens. The ~3x data inefficiency comes from static wave kernels — the same `exp(-αt)·cos(ωt+φ)` regardless of input content. SPECTRE (arXiv:2502.18394) proves FFT attention CAN beat standard: PPL 39.0 vs 39.4 on PG-19, using content-adaptive spectral gating.

## Root Cause

Our wave kernels are **input-independent**. The kernel FFT is computed purely from learnable parameters `(α, ω, φ)` — no conditioning on what tokens are actually present. SPECTRE's key insight: a small MLP conditioned on the mean query can modulate the kernel in frequency domain, making the effective receptive field adapt per-sample.

## Design: Content-Adaptive Spectral Gate

### New Module: `SpectralGate`

A lightweight MLP that takes the **mean query vector** (global query summary) and produces per-frequency scale and bias to modulate the base kernel FFT.

```
q_bar = LayerNorm(mean(q, dim=seq))        # (B, H, head_dim) → (B, H, head_dim)
q_bar_flat = flatten(q_bar)                 # (B, H*head_dim)
h = GELU(Linear(H*d → H*d))               # hidden layer
delta = Linear(H*d → 2*H*freq_bins)        # scale + bias per head per freq bin
delta_scale, delta_bias = split(delta)      # each (B, H, freq_bins)
modulated_fft = base_kernel_fft * (1 + δ_scale) + δ_bias
```

### Integration into Forward Pass

```
q, k, v = project(x)
k_feat, q_feat = φ(k), φ(q)              # learned feature maps (existing V4.3)
deposit = k_feat * v
field = scatter(deposit)

base_kernel_fft = build_wave_kernels()     # existing HiPPO-init kernels
modulated_fft = spectral_gate(q, base_fft) # NEW: content-adaptive modulation
field = wave_convolve(field, modulated_fft) # use modulated kernel

field = apply_field_coupling(field)
gathered = gather(field)
output = q_feat * gathered
```

### Key Properties

- **Still O(n log n)**: MLP runs on mean(q) which is O(n) to compute, modulation is O(freq_bins) per head
- **Content-adaptive**: Different inputs produce different effective kernels
- **Preserves physics**: Base kernel still enforces causality; MLP only modulates magnitude/phase in frequency domain
- **Low parameter overhead**: ~2 × D × D ≈ 131K params (1.6% of 8M model)

### Initialization

- `spectral_gate` MLP output initialized near zero (normal(0, 0.01))
- At init: `modulated_fft ≈ base_kernel_fft * (1 + 0) + 0 = base_kernel_fft`
- Model starts identical to V4.3 without spectral gate, then gradually learns content-adaptive modulation

## Files to Modify

| File | Change |
|------|--------|
| `src/wave_field_attention.py` | Add `SpectralGate` class, integrate into `forward()` |
| `src/wave_field_transformer.py` | Update `_init_weights()` for spectral gate near-zero init |
| `benchmarks/benchmark_v43.py` | New: 3-config benchmark (V4.3 base, V4.3+spectral, Standard) |

## Success Criterion

V4.3+SPECTRE at 5M tokens: PPL ≤ 500 (within 6% of Standard's ~473).
Stretch goal: PPL ≤ 473 (match or beat Standard).

## Literature Grounding

- SPECTRE (arXiv:2502.18394): Content-adaptive spectral gating, PPL 39.0 vs Standard 39.4
- Hedgehog (ICLR 2024): Learned feature maps with identity init
- S4D (arXiv:2206.11893): HiPPO initialization for damped oscillator kernels
