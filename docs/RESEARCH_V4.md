# Wave Field LLM — Research Synthesis for V4.x

> Literature review for closing the quality gap between Wave Field Attention (O(n log n)) and Standard Transformer (O(n^2)).
>
> Current state: V4.1 linear-wave hits PPL 543 at 15M tokens, Standard hits 473 at 5M.
> The bottleneck is the feature map, not the wave kernel.

---

## 1. The Core Problem: Why Wave Field is Slower to Learn

### 1.1 Standard Attention Works Immediately

Standard attention computes `score = Q . K^T / sqrt(d)`. Both Q and K start as random projections of the input. Their dot product captures pairwise correlations from step 1 — no learning needed for the scoring mechanism itself.

### 1.2 Linear-Wave Attention Has a Cold Start

V4.1 uses `phi(x) = elu(x) + 1` as a positive feature map:
- deposit = phi(K) * V (K modulates V per dimension)
- output = phi(Q) * gathered (Q selects dimensions)

At initialization (weights ~ N(0, 0.02)):
- elu(0) + 1 = 1 everywhere
- phi(K) = 1, phi(Q) = 1 → no routing, no content dependence
- Model defaults to V3.x behavior (pure wave, no content routing)
- Takes ~3M tokens for phi(K) and phi(Q) to differentiate

### 1.3 Theoretical Foundation

**"Training Dynamics of In-Context Learning in Linear Attention"** (arXiv:2501.16265)

The escape time from the plateau is:

```
tau ~ (1 / ||Lambda^2||) * ln(1 / w_init)
```

This is **logarithmic in initialization scale** — smaller init = exponentially longer plateau. Our `elu(0)+1 = 1.0` uniform initialization is the worst case: effectively zero variance in the feature space.

The paper also shows that loss trajectories under small initialization are **sigmoidal** — a long flat plateau followed by an abrupt drop. This matches our V4.1 training curve exactly (stuck at PPL 1370 for 3M tokens, then rapid descent).

**"Loss Plateaus in Transformers"** (arXiv:2506.13688)

Three degenerate behaviors during the stuck phase:
1. Model learns easy tokens only (partial solutions)
2. Strong repetition bias
3. Representation collapse (hidden state cosine similarity > 0.90)

Key finding: **attention maps are silently learning during the plateau**. The loss looks flat, but attention patterns gradually develop. Artificially amplifying optimal attention patterns by 10x eliminates the plateau entirely.

---

## 2. Feature Maps: What the Literature Says

### 2.1 elu(x) + 1 — What We Use (Weak)

**Source**: Katharopoulos et al. 2020, "Transformers are RNNs"

The original linear attention feature map. For positive inputs, elu(x) is linear. For negative inputs, exponential continuity. Ensures non-negativity for the linear attention kernel trick.

**Problem**: It's a poor softmax approximation. It can't produce the **sharp, peaked distributions** that softmax creates. At initialization, it's essentially constant (1.0), providing zero routing signal.

**Our V4.2 experiment confirmed this**: even with 3x elevated LR (which accelerated learning by 16%), we only reached PPL 997 at 5M tokens — still 2x worse than Standard's 473.

### 2.2 Taylor Feature Map — Much Better (BASED, Hazy Research 2024)

**Source**: Arora et al. 2024, "Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff"
- Paper: https://hazyresearch.stanford.edu/blog/2024-03-03-based
- Key paper from Hazy Research (Stanford)

**Feature map**: 2nd-order Taylor expansion of exp(x):

```
phi(x) = [1, x_1, x_2, ..., x_d, x_1*x_1, x_1*x_2, ..., x_d*x_d]
```

This expands feature dimension from d to O(d^2), capturing multiplicative interactions that 1st-order maps (like elu+1) cannot.

**Results**:
- Outperforms Performers, CosFormer, and elu+1 on recall benchmarks
- 56% faster than FlashAttention-2 at prompt processing (4K seq, 1.3B params)
- 24x higher throughput than FlashAttention-2 in next-token prediction

**Why it's better**: The 2nd-order terms capture pairwise feature correlations. With elu+1, two tokens with features [1, 0] and [0, 1] look identical after the feature map (both map to ~1.0). With Taylor, they produce distinct outer products.

**Tradeoff**: Expands dimension from d to d + d^2. For head_dim=32, that's 32 + 1024 = 1056 dimensions. A low-rank approximation (project to 8 dims first, then outer product = 64) keeps it manageable.

### 2.3 Learned Feature Map MLP — Best Quality (Hedgehog, ICLR 2024)

**Source**: Zhang & Bhatia et al. 2024, "The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry"
- Paper: https://arxiv.org/abs/2402.04347
- ICLR 2024 paper

**Feature map**: A single-layer MLP (Linear(head_dim, head_dim)) trained end-to-end. The MLP learns to produce **spiky** (low-entropy) and **dot-product-monotonic** outputs — the two key properties of softmax.

**Critical initialization**: **Identity matrix**. At init, phi(q) = q and phi(k) = k, so linear attention degenerates to q^T * k dot products — a reasonable starting point that works from step 1.

**Results**:
- Recovers **>99% of Standard Transformer performance** when training from scratch
- Outperforms prior linear attentions by **up to 6 PPL points** on WikiText-103
- Converting pretrained GPT-2: achieves 16.7 PPL on WikiText-103

**Why this is the fix we need**: Our elu+1 maps to a constant. Hedgehog's identity-init MLP maps to the input itself. The difference between "everything looks the same" and "every token is distinct" from step 1.

**Cost**: One Linear(32, 32) per feature map = 1024 params per head, 8192 per layer. For our 6-layer, 8-head model: +49,152 params (0.6% of 7.8M). Negligible.

### 2.4 cosFormer — Physics-Aligned Alternative (ICLR 2022)

**Source**: Qin et al. 2022, "cosFormer: Rethinking Softmax in Attention"
- Paper: https://arxiv.org/abs/2202.08791

Uses ReLU for non-negativity + **cosine re-weighting by position distance**:

```
weight(i,j) = ReLU(Q_i) * ReLU(K_j) * cos(pi * (i-j) / (2*M))
```

The cosine term naturally decays with distance. Decomposable via Ptolemy's theorem for linear-time computation.

**Relevance to Wave Field**: Our wave kernel already provides `cos(omega*t + phi) * exp(-alpha*t)` weighting. cosFormer suggests this wave kernel could serve AS the re-weighting function for linear attention, merging the two into one operation.

### 2.5 Symmetry-Aware Taylor (2025) — Higher-Order Efficiently

**Source**: arXiv:2602.00294

A P=4 Taylor expansion achieves Float16-level accuracy. Exploits symmetry in tensor products — instead of d^p monomials per degree p, uses the upper hyper-triangular region: C(d+p-1, p) features. Orders of magnitude fewer than naive expansion.

---

## 3. FFT-Based Attention: What Others Do

### 3.1 SPECTRE (Feb 2025) — Closest to Our Architecture

**Source**: Fein-Ashley 2025, "An FFT-Based Efficient Drop-In Replacement to Self-Attention"
- Paper: https://arxiv.org/abs/2502.18394

Replaces each attention head with: `FFT → learnable spectral gate → modReLU → IFFT`

The spectral gate is **conditioned on a global context vector** (mean-pooled input) fed through a small MLP that produces per-frequency-bin scaling and bias. The modReLU applies ReLU to complex magnitude while preserving phase.

**Results**: Up to 7x faster than FlashAttention-2 at 128K tokens. Matches or exceeds baseline performance.

**This is remarkably close to our architecture.** Key difference: their frequency-domain filter is **input-dependent** (content-adaptive). Ours is fixed (learned wave_frequency/damping/phase, same for all inputs). Making our kernel input-dependent would give content-adaptive wave propagation at minimal extra cost.

We already have the `FieldInterferenceModule` computing causal cumulative mean — it could provide the global context vector to modulate the kernel FFT.

### 3.2 Hyena Hierarchy (2023) — Implicitly Parameterized Kernels

**Source**: Poli et al. 2023, "Hyena Hierarchy: Towards Larger Convolutional Language Models"
- Paper: https://arxiv.org/abs/2302.10866

Kernel: `h(t) = exp(-alpha * t) * FFN(gamma(t))` where FFN uses **sine activations** and gamma(t) is a positional encoding.

Key initialization: decay rate alpha **varies across channels** (not just heads). Different channels within the same head get different temporal reach.

**Results**: Reaches Transformer quality with **20% less training compute** at 2K context. Crossover vs FlashAttention at 4096-8192 tokens.

**Our architecture is a special case** where the FFN is replaced by a single cosine. Adding per-channel decay variation within each head would give multi-resolution attention without the multi-component wavelet overhead.

### 3.3 FNet (Google 2021) — Unparameterized FFT

**Source**: Lee-Thorp et al. 2021, "FNet: Mixing Tokens with Fourier Transforms"

Replaces attention with raw 2D DFT (no learnable parameters). Gets 92% of BERT accuracy, 7x faster.

**Critical finding**: A **hybrid** FNet (FFT in early layers, standard attention in last 2 layers) recovers **97% of BERT accuracy**. Attention at the TOP of the model matters most.

**Implication for us**: Consider a hybrid architecture where the last 1-2 layers use full local attention while earlier layers use wave convolution. The early layers learn representations; the final layers need precise token-level discrimination.

### 3.4 FlashFFTConv (ICLR 2024) — Hardware Optimization

**Source**: Fu et al. 2024, "FlashFFTConv: Efficient Convolutions for Long Sequences"
- Paper: https://github.com/HazyResearch/flash-fft-conv

Uses Monarch decomposition to break FFT into tensor-core-friendly matrix multiplies. Up to 8.7x faster than PyTorch FFT, 62.3% FLOP utilization.

Relevant for scaling to longer sequences (>4K), not for quality improvements.

---

## 4. State Space Models: Initialization & Gating

### 4.1 S4 HiPPO Initialization — The Gold Standard

**Source**: Gu et al. 2022, "On the Parameterization and Initialization of Diagonal State Space Models"
- Paper: https://arxiv.org/abs/2206.11893

The HiPPO matrix projects past information onto Legendre polynomials. In diagonal form (S4D):

```
lambda_n = -1/2 + i * pi * (n + 1/2)    (S4D-Lin)
```

Real part (-1/2) = uniform damping. Imaginary part = linearly spaced harmonic frequencies.

**Direct mapping to our wave parameters:**

| S4D | Wave Field | Correspondence |
|-----|-----------|----------------|
| Re(lambda) = -1/2 | wave_damping | Exponential decay rate |
| Im(lambda) = pi*(n+0.5) | wave_frequency | Oscillation speed |
| N/A | wave_phase | Extra DoF (not in S4) |

**Our current init**:
- damping: linspace(-3.0, 0.5, H) — **varied across heads**
- frequency: linspace(0.3, 4.0, H) — **much lower than HiPPO**

**HiPPO-inspired init would be**:
```python
wave_frequency = pi * (2*n + 1) / 2  for n = 0..H-1  # harmonic series
wave_damping = -0.69 for all heads  # softplus(-0.69) = 0.5 = uniform
```

Key insight: **equal damping across all heads** (not varied!) ensures all timescales are equally represented at initialization.

### 4.2 Mamba — Selective State Spaces

**Source**: Gu & Dao 2023, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- Paper: https://arxiv.org/abs/2312.00752

Makes SSM parameters (B, C, delta) **input-dependent** through linear projections. The selection mechanism allows content-based filtering.

Key finding: **strict HiPPO initialization matters less** when you have content-dependent gating. The gating mechanism compensates for imperfect kernel init because it learns to route around initialization issues.

Delta (discretization step) initialized as `1/dt_min ≈ sequence_length` (dt_min=0.001 for ~1000 tokens).

**Implication**: Our wave kernel params may not need perfect init IF we have strong content routing (which we don't yet, because elu+1 is weak).

### 4.3 RetNet — Multi-Scale Decay

**Source**: Sun et al. 2023, "Retentive Network: A Successor to Transformer for Large Language Models"
- Paper: https://arxiv.org/pdf/2307.08621

Per-head exponential decay rates:

```
gamma = 1 - exp(linspace(log(1/32), log(1/512), num_heads))
```

Some heads attend locally (~32 tokens back), others globally (~512 tokens back). Same decay across all layers for simplicity.

Uses GroupNorm instead of LayerNorm + swish gate for non-linearity.

---

## 5. Gated Linear Attention: State of the Art

### 5.1 GLA (ICML 2024)

**Source**: Yang et al. 2024, "Gated Linear Attention Transformers with Hardware-Efficient Training"
- Paper: https://arxiv.org/abs/2312.06635

Gate: `alpha_t = sigma(x_t * W1 * W2) * tau` (low-rank factored)
Recurrence: `S_t = G_t * S_{t-1} + v_t * k_t^T`

Competitive with LLaMA-architecture Transformer and Mamba. Trained on 2K tokens, generalizes to 20K+.

Training: warmup 1K steps to lr=2e-3, cosine decay to 3e-5, batch 1024, 100K steps.

### 5.2 Gated DeltaNet (ICLR 2025) — Current SOTA

**Source**: arXiv:2412.06464
- Adopted by **Qwen3-Next** as its linear-attention layer

Combines gating (from Mamba2) with the **delta rule** (error-correction):

```python
S = S * alpha_t                              # Decay old memory
kv_mem = (S * k_t).sum(dim=-2)              # Predict current value from memory
delta = (v_t - kv_mem) * beta_t             # Error-corrected update
S = S + k_t * delta                          # Write correction
y_t = (S * q_t).sum(dim=-2)                 # Read output
```

**Two gates**:
- Alpha (decay): `exp(-A * softplus(W_alpha(x) + dt_bias))` — controls forgetting
- Beta (update): `sigmoid(W_beta(x))` — controls writing

Output gate uses **SiLU** instead of sigmoid for better gradient flow.

Architecture: **3:1 ratio** — three DeltaNet layers per one full-attention layer in hybrid mode.

**This is the most relevant competitor.** The delta rule provides **targeted memory modification** (error correction), not just multiplicative gating. Our current wave field deposits everything and reads everything — no error correction.

### 5.3 RWKV-6 (2024)

**Source**: https://github.com/BlinkDL/RWKV-LM

Gate: `alpha_t = exp(-exp(x_t * W))` — double exponential for strict positivity.

**Critical**: K and R matrices initialized to **ZERO** for fast, stable convergence. Counterintuitive but proven.

**Data-dependent decay**: Each channel changes independently based on current input (V6 improvement over V5's fixed learned vector).

### 5.4 Gated Attention (NeurIPS 2025 Oral)

**Source**: arXiv:2505.06708

Adding a simple per-head sigmoid gate after standard attention consistently:
- Improves performance and training stability
- Enables larger learning rates
- Eliminates the "attention sink" phenomenon

Gate init: **zero parameters → sigmoid = 0.5** (gates start at 50%). This contradicts our bias=2.0 approach (88% open).

Our V4.2 experiment actually confirmed this: the "gate bug" (bias=0 → sigmoid=0.5) performed BETTER than the "fix" (bias=2.0) because 0.5 provides regularization on noisy early outputs.

---

## 6. Hybrid Architectures: Local + Global

### 6.1 Longformer / BigBird

Local sliding window + global tokens. O(n*w) complexity.

**Key insight for training efficiency**: Local attention should **dominate early in training** because it's easier to learn. The global mechanism (wave propagation in our case) gradually takes over.

Consider initializing `local_blend` biased toward local (~0.88 = sigmoid(2.0)) rather than equal (0.5).

### 6.2 BASED Architecture (Hazy Research)

Combines sliding window (w=64) for local precision with Taylor feature map linear attention for global. This is exactly our hybrid approach but with a much better feature map.

### 6.3 FNet Hybrid Pattern

FFT in early layers, full attention in final layers. The "TOP" placement of attention layers gives the best results.

---

## 7. Distillation Approaches

### 7.1 LoLCATs (Hazy Research 2024)

**Source**: arXiv:2410.10254

Two-step conversion from softmax to linear attention:
1. Train feature maps to minimize output MSE vs softmax attention (pure distillation)
2. LoRA fine-tuning to recover remaining quality

Uses only **0.2% of prior methods' parameters** and **0.4% of training tokens**. First to linearize 70B and 405B LLMs.

**Application**: Could bootstrap wave field attention from a pretrained standard transformer. Train wave kernels + feature maps to match the attention output, then fine-tune. Completely sidesteps the cold-start problem.

---

## 8. Prioritized Recommendations

Ordered by expected impact on closing the PPL gap (997 → 473):

### Tier 1: Must Do (Highest Impact)

| # | Change | Source | Expected Impact | Cost |
|---|--------|--------|-----------------|------|
| 1 | **Replace elu+1 with learned feature map MLP** (identity-init Linear(d,d)) | Hedgehog ICLR 2024 | Eliminate stuck phase, up to 6 PPL improvement | +49K params (0.6%) |
| 2 | **HiPPO-inspired kernel init** (uniform damping, harmonic freqs) | S4D | Better long-range patterns from step 0 | Zero cost |
| 3 | **Content-adaptive spectral gate** (global context modulates kernel FFT) | SPECTRE 2025 | Input-dependent wave propagation | Small MLP |

### Tier 2: Should Do (Medium Impact)

| # | Change | Source | Expected Impact | Cost |
|---|--------|--------|-----------------|------|
| 4 | **Hybrid: last 1-2 layers use full local attention** | FNet, BASED | Precise token discrimination where it matters | O(n*w) for top layers |
| 5 | **Delta rule memory updates** (error-corrected deposits) | Gated DeltaNet | Targeted memory modification vs blind deposit | Architecture change |
| 6 | **modReLU in frequency domain** | SPECTRE | Nonlinear spectral interactions | 3 lines of code |

### Tier 3: Could Do (Lower Impact / Higher Risk)

| # | Change | Source | Expected Impact | Cost |
|---|--------|--------|-----------------|------|
| 7 | Taylor feature map (1 + x + x^2/2) | BASED | Richer routing at O(d^2) | Memory scales with d^2 |
| 8 | Self-distillation warmup (local attention teaches wave) | LoLCATs | Fast convergence for wave kernels | Training loop change |
| 9 | FlashFFTConv | Fu et al. | 4-8x speedup at long seqs | External CUDA dependency |
| 10 | Per-channel decay variation within heads | Hyena | Multi-resolution without multi-component | Param reshape |

---

## 9. Paper References

### Linear Attention & Feature Maps
- Katharopoulos et al. 2020 — "Transformers are RNNs" (elu+1 feature map)
- Arora et al. 2024 — "BASED: Simple Linear Attention" (Taylor feature maps, Hazy Research)
- Zhang & Bhatia et al. 2024 — "Hedgehog: Expressive Linear Attentions with Softmax Mimicry" (ICLR 2024, learned MLP feature maps)
- Qin et al. 2022 — "cosFormer: Rethinking Softmax in Attention" (ICLR 2022, cosine re-weighting)
- arXiv:2602.00294 — Symmetry-Aware Taylor Approximation (2025, higher-order)
- arXiv:2501.16265 — Training Dynamics of In-Context Learning in Linear Attention (plateau theory)
- arXiv:2506.13688 — Loss Plateaus in Transformers (2025, attention silently learns during plateau)

### FFT-Based Models
- Fein-Ashley 2025 — "SPECTRE: FFT-Based Drop-In Replacement for Self-Attention" (content-adaptive spectral gate)
- Poli et al. 2023 — "Hyena Hierarchy" (implicit long convolution via FFT)
- Lee-Thorp et al. 2021 — "FNet: Mixing Tokens with Fourier Transforms" (Google, hybrid FFT+attention)
- Fu et al. 2024 — "FlashFFTConv" (ICLR 2024, hardware-optimized FFT)
- arXiv:2601.08602 — "WaveFormer: Frequency-Time Decoupled Vision Modeling" (2025, wave equation attention)

### State Space Models & Initialization
- Gu et al. 2022 — "S4D: On the Parameterization and Initialization of Diagonal State Space Models" (HiPPO)
- Gu & Dao 2023 — "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- Sun et al. 2023 — "RetNet: A Successor to Transformer" (multi-scale decay)
- arXiv:2508.20441 — Spectral Bias in Diagonal State Space Models (2025)

### Gated Linear Attention
- Yang et al. 2024 — "GLA: Gated Linear Attention with Hardware-Efficient Training" (ICML 2024)
- arXiv:2412.06464 — "Gated DeltaNet" (ICLR 2025, adopted by Qwen3-Next)
- RWKV-6 — https://github.com/BlinkDL/RWKV-LM (zero-init K/R, data-dependent decay)
- arXiv:2404.07904 — "HGRN2: Gated Linear RNNs with State Expansion" (hierarchical forget gates)
- arXiv:2505.06708 — "Gated Attention for LLMs" (NeurIPS 2025 Oral)

### Distillation & Conversion
- arXiv:2410.10254 — "LoLCATs: Low-Rank Linearizing of LLMs" (Hazy Research, softmax→linear conversion)

### Hybrid Architectures
- Beltagy et al. 2020 — "Longformer: The Long-Document Transformer" (local + global attention)
- Zaheer et al. 2020 — "BigBird: Transformers for Longer Sequences" (sparse attention)

---

## 10. V4.2 Experiment Results (for reference)

What we tried and what happened:

| Config | PPL (5M) | Acc | Notes |
|--------|----------|-----|-------|
| V4.1 Baseline | 1225 | 6.3% | elu+1, no fixes |
| Gate Fix Only | 1338 | 5.2% | bias=2.0 passed noise through |
| Gate + QK Bias | 1243 | 6.2% | bias diversity alone: minimal |
| Gate + Elevated LR | 1023 | 7.5% | 3x LR dominant factor |
| V4.2 Full | 997 | 7.7% | All three, first under 1000 |
| Standard Transformer | 473 | 11.6% | Reference |
| V4.1 at 15M tokens | 543 | 10.6% | More data compensates for weak feature map |

**Lesson**: Initialization tricks on a fundamentally weak feature map (elu+1) get ~18% improvement. The architecture needs a better feature map, not better initialization.

---

## 11. Deep-Dive: Exact Forward Passes of Key Competitors

> Added after downloading and reading the actual papers (docs/papers/).

### 11.1 SPECTRE — Exact Forward Pass

The closest architecture to ours. Same FFT convolution idea but with critical differences.

```
Input: X ∈ R^(N×D)

1. Project: Q, V = X @ W_q, X @ W_v

2. Global FFT:
   F = RFFT(V)  along sequence dim → C^((N/2+1)×D)

3. Content-Adaptive Spectral Gate:
   q_bar = LayerNorm(mean(Q))        # global query summary
   delta_s, delta_b = MLP(q_bar)     # 2-layer MLP → complex scaling + bias
   W = W_base * (1 + delta_s)        # modulate base filter
   b = b_base + delta_b
   F_gated = F * W + b               # element-wise in frequency domain

4. modReLU activation (in frequency domain):
   z = r * e^(iθ)
   modReLU(z) = max(r + b_learnable, 0) * e^(iθ)
   # Thresholds magnitude, preserves phase

5. Inverse FFT:
   Y = IRFFT(F_gated) → R^(N×D)
```

**Key insight for Wave Field**: SPECTRE's spectral gate is INPUT-DEPENDENT.
Our wave kernel is FIXED (same kernel regardless of input content).
This is potentially a bigger gap than the feature map issue.

**PPL comparison**: SPECTRE+WRM gets 39.0 vs Standard's 39.4 on PG-19.
That's BETTER than standard transformer. Proof that FFT-based attention CAN beat O(n²).

### 11.2 BASED — Exact Forward Pass

The best sub-quadratic language model architecture for recall tasks.

```
Architecture: Interleaved layers (NOT combined within one layer!)
  ~20% linear attention layers (global)
  ~20% sliding window attention layers (local, w=64)
  ~60% gated convolution layers

Linear attention layer:
  1. Project: q, k = X @ W_q, X @ W_k  (project to d'=16 first!)
  2. Taylor feature map:
     φ(x) = [1, x₁..x_d', x₁x₁/√2, x₁x₂/√2, ..., x_d'x_d'/√2]
     Total dim: 1 + 16 + 256 = 273
  3. Causal cumsum:
     S_i = S_{i-1} + φ(k_i) ⊗ v_i
     z_i = z_{i-1} + φ(k_i)
  4. Output:
     y_i = (φ(q_i)^T @ S_i) / (φ(q_i)^T @ z_i)

Sliding window layer:
  Standard causal softmax attention within window w=64.
  Full Q·K^T/√d → softmax → V. Just windowed.
```

**Critical insight**: BASED does NOT combine local + global in one layer.
They are SEPARATE layer types, interleaved. This is fundamentally different
from our approach of blending wave + local within one attention layer.

**PPL**: 7.43 vs Transformer++ 7.26 at 1.3B params. Only 2.3% gap.
24x faster generation throughput.

### 11.3 FNet Hybrid — Exact Architecture

```
Layers 1-10: FFT mixing (no learned params in attention!)
  z = Real(FFT_seq(FFT_hidden(x)))
  z = LayerNorm(z + x)
  y = LayerNorm(FFN(z) + z)

Layers 11-12: Standard self-attention
  Full O(n²) attention for final token discrimination.
```

**Recovery**: 97-99% of BERT quality with just 2 attention layers at the TOP.

**Critical insight**: Attention at the END of the model matters most.
Early layers do representation building (FFT is fine for this).
Final layers need precise token-level discrimination (needs attention).

### 11.4 Implications for Wave Field LLM

Three patterns emerge from ALL successful sub-quadratic architectures:

1. **Content-adaptive frequency filtering** (SPECTRE): The kernel/filter
   MUST depend on the input. Fixed kernels can't compete.

2. **Separate layer types, not blended** (BASED, FNet): Don't mix
   global + local within one layer. Interleave them as separate layers.

3. **Attention at the top** (FNet, Gated DeltaNet's 3:1 ratio):
   The final layers need real attention for precise recall.

Our current architecture violates all three:
- Wave kernel is input-independent (fixed frequency/damping/phase)
- We blend wave + local within one layer (local_window)
- All layers are identical wave layers (no attention at top)

---

## 12. V4.3 Mathematical Foundations — Complex Exponential Kernels

### 12.1 The Key Identity: Real Cosines ARE Complex Exponentials

Our wave kernel `exp(-alpha*t) * cos(omega*t + phi)` is the real part of a complex exponential:

```
exp(-alpha*t) * cos(omega*t + phi) = Re[ e^{i*phi} * exp((-alpha + i*omega)*t) ]
```

Using conjugate pairs (lambda and lambda*):

```
k(t) = c * exp(lambda*t) + conj(c) * exp(conj(lambda)*t)

where lambda = -alpha + i*omega, c = exp(i*phi)
```

This produces a REAL kernel from complex poles — same output as our real cosine, but with different computational and mathematical properties.

### 12.2 The Analytic Z-Transform (S4D-Style)

Instead of building G time-domain samples and FFTing them (O(G log G)), compute the DFT analytically using the geometric series formula:

```
H_lambda(z_k) = sum_{t=0}^{G-1} exp(lambda*t) * z_k^{-t}
              = (1 - exp(lambda*G) * z_k^{-G}) / (1 - exp(lambda) * z_k^{-1})
```

where `z_k = exp(i * 2*pi*k / 2G)` for frequency bin k.

The full real kernel DFT:
```
K(z_k) = c * H_lambda(z_k) + conj(c) * H_{conj(lambda)}(z_k)
```

IMPORTANT: `H_{conj(lambda)}(z_k) != conj(H_lambda(z_k))` because conj(z_k^{-1}) = z_k (not z_k^{-1}). The conjugate pole's transfer function must be computed separately using conj(exp(lambda)) but the SAME z_k^{-1}.

### 12.3 Why Analytic Kernels Are Better

| Property | Time-domain + rfft | Analytic Z-transform |
|----------|-------------------|---------------------|
| Causality | Must enforce via `_enforce_causal_kernel` | Automatic (sums only t >= 0) |
| Gradient flow | Through G discretized samples | Directly through pole params (alpha, omega) |
| Kramers-Kronig | External constraint | Built-in mathematical property |
| Minimum-phase | Not guaranteed | Automatic when alpha > 0 |
| Memory | O(G) for time-domain kernel | O(freq_bins) only |
| Compute | O(G log G) for rfft | O(H * freq_bins) for closed-form |

Verified: analytic and legacy kernels produce identical output (cosine similarity = 1.000000, max_diff < 5e-7 across all heads).

### 12.4 Kramers-Kronig Relations and Causality

For a causal system (impulse response h(t) = 0 for t < 0), the real and imaginary parts of its frequency response H(omega) are Hilbert transform pairs:

```
Re[H(omega)] = (1/pi) * P.V. integral{ Im[H(omega')] / (omega' - omega) } d(omega')
```

They CANNOT be chosen independently. Our SpectralGate multiplies the kernel FFT by arbitrary per-frequency-bin gates, which breaks this constraint (the cause of the V4.1 causality leak). The `_enforce_causal_kernel` fix projects back to causal space after gate modulation.

With analytic kernels, the BASE kernel satisfies Kramers-Kronig automatically. Only the spectral gate modulation needs the causal projection.

### 12.5 Connection to S4D State Space Models

S4D parameterizes a diagonal state space model:
```
x'(t) = diag(lambda_1, ..., lambda_N) * x(t) + B * u(t)
y(t) = C * x(t) + D * u(t)
```

The discrete convolution kernel is `k(t) = C * exp(Lambda * t) * B`, summed over eigenvalue pairs. Our wave field is equivalent to a diagonal S4D where:
- Each head = one conjugate eigenvalue pair
- lambda_n = -softplus(wave_damping_n) + i * wave_frequency_n
- C * exp(i*phase) = the output coupling coefficient
- The "state" is the wave field itself

### 12.6 V4.2 Ablation Update (Corrected Results)

After applying the causality fix:

| Config | PPL (5M) | vs Baseline |
|--------|----------|-------------|
| A) Baseline (causal-fixed) | 912.3 | - |
| B) + Q/K bias diversity | 922.6 | +1% (noise, DISPROVEN) |
| **C) + Elevated QK LR (3x)** | **724.0** | **-21%** |
| D) + Residual scaling | 808.5 | -11% |
| E) Full V4.2 (B+C+D) | 783.1 | -14% (worse than C alone) |
| F) Standard Transformer | 412.6 | reference |

**Lesson**: Only 1 of 3 changes worked. Research-first approach for V4.3.

---

## 13. V4.3 Changes and Their Literature Backing

| Change | Paper | Confidence | Impact |
|--------|-------|-----------|--------|
| Analytic Z-transform kernel | S4D (arXiv:2206.11893) | 90% | Better gradient flow |
| 2-layer Hedgehog feature maps | Hedgehog (ICLR 2024) | 90% | Closes 68.6% of gap |
| Local window hybrid (w=64) | BASED (Hazy 2024) | 85% | Pareto-optimal local+global |
| Elevated QK LR (3x) default | Our V4.2 ablation | 100% | -21% PPL |
| Local blend bias toward local | BASED, Longformer | 85% | Better early training |

---

## 14. Downloaded Papers

All papers stored in `docs/papers/`:

| File | Paper | Key Contribution |
|------|-------|-----------------|
| spectre_fft_attention.pdf | SPECTRE (2025) | Content-adaptive spectral gate, modReLU |
| based_linear_attention.pdf | BASED (2024) | Taylor feature maps, interleaved architecture |
| fnet_fourier_mixing.pdf | FNet (2021) | Unparameterized FFT mixing, hybrid pattern |
| hedgehog_linear_attention.pdf | Hedgehog (2024) | Identity-init learned feature maps |
| s4d_diagonal_ssm.pdf | S4D (2022) | HiPPO initialization for sequence kernels |
| gated_deltanet.pdf | Gated DeltaNet (2025) | Delta rule + gating, SOTA linear attention |
| linear_attn_training_dynamics.pdf | Training Dynamics (2025) | Plateau theory for linear attention |
| hyena_hierarchy.pdf | Hyena (2023) | Implicit long convolution, per-channel decay |
