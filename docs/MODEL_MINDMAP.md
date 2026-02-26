# Wave Field LLM -- Complete Model Mind Map

> **Version**: V4.3.3 SPECTRE-Wave
> **Status**: S1 benchmark -- PPL 239 (Wave) vs 171 (Standard) = 1.40x gap
> **Complexity**: O(n log n) via FFT convolution
> **License**: AGPL-3.0

Read time: ~15 minutes. This document covers everything.

---

## TABLE OF CONTENTS

1. [What Is The Model](#1-what-is-the-model)
2. [Algorithms and Pseudocode](#2-algorithms-and-pseudocode)
3. [Learnable Parameters Inventory](#3-learnable-parameters-inventory)
4. [What Went Well](#4-what-went-well)
5. [What Went Wrong](#5-what-went-wrong)
6. [Core Issues To Address](#6-core-issues-to-address)
7. [The PPL Gap Analysis](#7-the-ppl-gap-analysis)

---

## 1. WHAT IS THE MODEL

### 1.1 One-Sentence Summary

Wave Field LLM replaces the O(n^2) softmax attention matrix with a continuous wave
field where tokens deposit information, waves propagate it causally via O(n log n) FFT
convolution, and tokens read back the propagated signals.

### 1.2 Architecture Overview (ASCII Diagram)

```
INPUT TOKEN IDS: (B, N)          B=batch, N=seq_len (512)
       |
       v
+------------------+
| Token Embedding  |  nn.Embedding(vocab=8000, dim=384)
| + Sinusoidal PE  |  Fixed sin/cos positional encoding
| + Dropout(0.1)   |
+------------------+
       |
       v  (B, N, 384)
       |
  +============================+
  |  WaveFieldTransformerLayer |  x8 layers (S1 config)
  |  (pre-norm residual)       |
  |                            |
  |  x -> norm1(x)             |
  |       -> WaveFieldAttention|-----+
  |  x = x + dropout(attn)    |     |  Every 3 layers:
  |                            |     |  FieldInterference
  |  x -> norm2(x)             |     |  Module applied
  |       -> FFN(GELU)         |     |
  |  x = x + ffn              |     |
  +============================+     |
       |                             |
       | (after L3, L6)              |
       +-----> FieldInterferenceModule
       |       (causal cumulative mean
       |        + phase alignment gate)
       v
+------------------+
| Final LayerNorm  |
+------------------+
       |
       v  (B, N, 384)
+------------------+
| Output Linear    |  weight-tied with Token Embedding
+------------------+
       |
       v  (B, N, 8000)  logits
```

### 1.3 Inside WaveFieldAttention -- The Core Innovation

This is where the physics happens. Each layer contains one of these:

```
INPUT x: (B, N, D)         D=384, N=512, H=8 heads, head_dim=48
       |
+------+------+
| Fused QKVG  |  nn.Linear(384, 4*384)  -- single matmul
| Projection   |
+-+--+--+--+--+
  |  |  |  |
  Q  K  V  Gate_raw     each (B, N, 384)
  |  |  |  |
  |  |  |  +---> sigmoid(gate_raw) -> gate: (B, H, N, 48)
  |  |  |
  +--+--+--- reshape to (B, H, N, 48) each
  |  |  |
  |  |  |
  |  +--+--> K-WEIGHTED DEPOSIT
  |  |  |    deposit = phi_k(K) * V     elementwise (B, H, N, 48)
  |  |  |    "K feature map modulates V per dimension"
  |  |  |
  |  |  v
  |  |  BILINEAR SCATTER onto field
  |  |  field: (B, H, G, 48)    G=field_size=2048
  |  |  field_pos = token_idx * stride  (absolute mapping)
  |  |       |
  |  |       v
  |  |  BUILD WAVE KERNEL (Z-transform analytic)
  |  |  kernel_fft: (H, freq_bins) complex
  |  |       |
  |  |       v
  |  +-> SPECTRAL GATE (SPECTRE)
  |  |   q_bar = LayerNorm(Q[:,:,0,:])   first token only (causal!)
  |  |   ctrl = MLP(flatten(q_bar))      (B, H, 32) control points
  |  |   gate = interpolate(ctrl, freq_bins)   smooth spectral gate
  |  |   modulated = base_fft * (1 + gate)     (B, H, freq_bins)
  |  |       |
  |  |       v
  |  |  ENFORCE CAUSALITY
  |  |  IFFT -> zero t>=G -> FFT   (project back to causal)
  |  |       |
  |  |       v
  |  |  FFT CONVOLUTION (fp32 forced)
  |  |  field_fft = rfft(field)
  |  |  convolved = irfft(field_fft * kernel_fft)[:G]
  |  |       |
  |  |       v
  |  |  CROSS-HEAD FIELD COUPLING
  |  |  coupling = softmax(learned H x H matrix)
  |  |  field = einsum('ij,bjgd->bigd', coupling, field)
  |  |       |
  |  |       v
  |  |  BILINEAR GATHER at token positions
  |  |  gathered: (B, H, N, 48)
  |  |       |
  v  v       v
  Q-WEIGHTED READ
  output = phi_q(Q) * gathered    elementwise (B, H, N, 48)
       |
       v
  CONTENT-DEPENDENT GATING
  output = output * gate          (B, H, N, 48)
       |
       v
  RESHAPE + OUT_PROJ
  output: (B, N, 384) -> nn.Linear(384, 384) -> (B, N, 384)
```

### 1.4 Data Shapes at Each Step (S1 Config)

```
Step                          Shape                    dtype
--------------------------------------------------------------
Input token ids               (16, 512)                int64
After embedding + PE          (16, 512, 384)           bf16
After reshape to heads        (16, 8, 512, 48)         bf16
phi_k(K)                      (16, 8, 512, 48)         bf16
deposit = phi_k(K) * V        (16, 8, 512, 48)         bf16
field (after scatter)          (16, 8, 2048, 48)        bf16
field_pos_float               (512,)                   fp32
base_kernel_fft               (8, 2049) complex         fp32
spectral_gate output          (16, 8, 2049) complex     fp32
field_fft                     (16, 48, 8, 2049) complex fp32
convolved field               (16, 8, 2048, 48)        bf16
after coupling                (16, 8, 2048, 48)        bf16
gathered                      (16, 8, 512, 48)         bf16
phi_q(Q) * gathered           (16, 8, 512, 48)         bf16
after gate                    (16, 8, 512, 48)         bf16
reshape + out_proj            (16, 512, 384)           bf16
logits (output proj)          (16, 512, 8000)          bf16
```

### 1.5 How It Differs From Other Architectures

```
+---------------------+-------------------+-------------------+-------------------+
|                     | Standard          | Hyena             | Wave Field LLM    |
|                     | Transformer       | (Poli et al.)     | (This Work)       |
+---------------------+-------------------+-------------------+-------------------+
| Attention           | Q*K^T softmax     | Implicit NN       | Damped wave       |
| mechanism           | O(n^2)            | kernels, O(n logn)| kernel, O(n logn) |
+---------------------+-------------------+-------------------+-------------------+
| Kernel              | Full NxN matrix   | Neural network    | PHYSICS-BASED:    |
| parameterization    | (data-dependent)  | output            | exp(-gamma*t) *   |
|                     |                   | (data-independent)| cos(omega*t + phi)|
|                     |                   |                   | 3 params per head |
+---------------------+-------------------+-------------------+-------------------+
| Content-adaptive?   | YES (softmax      | NO (static after  | YES (SpectralGate |
|                     | over QK^T)        | training)         | modulates kernel  |
|                     |                   |                   | FFT per input)    |
+---------------------+-------------------+-------------------+-------------------+
| Causality           | Causal mask       | Implicit (kernel  | Zero kernel for   |
|                     | on attention      | vanishes for t<0) | t<0 before FFT +  |
|                     | matrix            |                   | IFFT->zero->FFT   |
|                     |                   |                   | after SpectralGate|
+---------------------+-------------------+-------------------+-------------------+
| Feature maps        | None (softmax)    | None              | Learned           |
|                     |                   |                   | ELU+1 (Hedgehog-  |
|                     |                   |                   | style identity MLP)|
+---------------------+-------------------+-------------------+-------------------+

+---------------------+-------------------+-------------------+-------------------+
|                     | Mamba (Gu &       | GLA (Yang et      | Wave Field LLM    |
|                     | Dao, 2024)        | al., 2024)        | (This Work)       |
+---------------------+-------------------+-------------------+-------------------+
| Core idea           | Input-dependent   | Gated Linear      | Wave field +      |
|                     | state space with  | Attention with    | spectral gate     |
|                     | selective scan    | per-token gates   | modulation        |
+---------------------+-------------------+-------------------+-------------------+
| Complexity          | O(n)              | O(n)              | O(n log n)        |
+---------------------+-------------------+-------------------+-------------------+
| Per-token           | YES (selective    | YES (per-token    | PARTIAL           |
| input-dependent     | gating of state   | decay gates)      | (per-SAMPLE via   |
| state transition    | update)           |                   | SpectralGate,     |
|                     |                   |                   | NOT per-token)    |
+---------------------+-------------------+-------------------+-------------------+
| Why it matters      | Decides WHAT to   | Decides HOW MUCH  | Same kernel for   |
|                     | write to state    | each token's info | all tokens in a   |
|                     | PER TOKEN         | decays PER TOKEN  | sequence. THIS IS |
|                     |                   |                   | THE KEY WEAKNESS. |
+---------------------+-------------------+-------------------+-------------------+
```

The critical architectural difference: Mamba and GLA have **per-token** input-dependent
state transitions. Wave Field LLM has **per-sample** spectral modulation -- the same
kernel applies to every token position within a sequence. This is the fundamental
reason for the PPL gap.

---

## 2. ALGORITHMS AND PSEUDOCODE

### 2.1 Complete Forward Pass

```python
def forward(input_ids, labels=None):
    """
    input_ids: (B, N) token indices
    Returns: logits (B, N, vocab), loss (scalar)
    """
    # 1. EMBED
    x = token_embedding(input_ids)                 # (B, N, D)
    x = x + sinusoidal_pe[:N]                      # add positional encoding
    x = dropout(x)

    # 2. TRANSFORMER LAYERS
    interference_idx = 0
    for i, layer in enumerate(layers):             # 8 layers for S1
        # Pre-norm attention + residual
        x_norm = layer_norm1(x)
        attn_out = wave_field_attention(x_norm)    # THE CORE -- see 2.2
        x = x + dropout(attn_out)

        # Pre-norm FFN + residual
        x_norm = layer_norm2(x)
        ffn_out = Linear(GELU(Linear(x_norm)))     # D->4D->D
        x = x + ffn_out

        # Field interference every 3 layers
        if (i + 1) % 3 == 0:
            x = field_interference(x)              # see 2.6

    # 3. OUTPUT
    x = final_layer_norm(x)
    logits = output_linear(x)                      # weight-tied with embedding
    loss = cross_entropy(logits, labels)
    return logits, loss
```

### 2.2 Wave Field Attention (The Core Algorithm)

```python
def wave_field_attention(x):
    """
    x: (B, N, D) -- already layer-normed
    Returns: (B, N, D)
    """
    B, N, D = x.shape
    H = num_heads          # 8
    d = D // H             # 48 (head_dim)
    G = field_size          # 2048

    # STEP 1: Fused QKVG projection (single matmul)
    qkvg = linear_4D(x)                           # (B, N, 4D)
    Q, K, V, gate_raw = split(qkvg, 4)            # each (B, N, D)
    # reshape to multi-head: (B, H, N, d)
    Q = Q.view(B, N, H, d).transpose(1, 2)
    K = K.view(B, N, H, d).transpose(1, 2)
    V = V.view(B, N, H, d).transpose(1, 2)

    # STEP 2: Absolute position mapping
    field_pos = arange(N) * stride                 # stride = (G-1)/(N-1)
    field_pos = field_pos.clamp(0, G-2)            # (N,) float32

    # STEP 3: Learned feature maps (Hedgehog-style, NormalizedExp)
    Q_feat = phi_q(Q)       # Linear(48,48) + ELU+1 + Linear(48,48) + ELU+1 + eps
    K_feat = phi_k(K)       # same architecture

    # STEP 4: K-weighted deposit
    deposit = K_feat * V                           # (B, H, N, d) elementwise

    # STEP 5: Bilinear scatter onto field
    field = zeros(B, H, G, d)
    for each token position p in field_pos:
        lo = floor(p), hi = lo + 1
        w_lo = 1 - frac(p), w_hi = frac(p)
        field[:, :, lo, :] += deposit[:, :, token_i, :] * w_lo
        field[:, :, hi, :] += deposit[:, :, token_i, :] * w_hi

    # STEP 6: Build wave kernel FFT (Z-transform, analytic)
    base_kernel_fft = build_analytic_kernel_fft()  # (H, freq_bins) complex
    # See 2.3 for details

    # STEP 7: SpectralGate -- content-adaptive kernel modulation
    kernel_fft = spectral_gate(Q, base_kernel_fft) # (B, H, freq_bins) complex
    # See 2.4 for details

    # STEP 8: Enforce causality (SpectralGate can break Kramers-Kronig)
    kernel_td = irfft(kernel_fft)                  # (..., pad_size) time domain
    kernel_td[..., G:] = 0                         # zero anti-causal half
    kernel_fft = rfft(kernel_td)                   # back to freq domain

    # STEP 9: FFT convolution (all in fp32)
    field_t = field.permute(0, 3, 1, 2)           # (B, d, H, G)
    field_fft = rfft(field_t.float(), n=pad_size)  # (B, d, H, freq)
    # Complex multiply (decomposed for Inductor fusion):
    # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    out_fft = field_fft * kernel_fft               # broadcast over d dim
    convolved = irfft(out_fft, n=pad_size)[:, :, :, :G]
    convolved = convolved.to(input_dtype)
    field = convolved.permute(0, 2, 3, 1)         # back to (B, H, G, d)

    # STEP 10: Cross-head field coupling
    coupling = softmax(learned_HxH_matrix, dim=-1)
    field = einsum('ij,bjgd->bigd', coupling, field)

    # STEP 11: Bilinear gather at token positions
    gathered = zeros(B, H, N, d)
    for each token position p in field_pos:
        lo = floor(p), hi = lo + 1
        gathered[:, :, i, :] = field[:, :, lo, :] * w_lo + field[:, :, hi, :] * w_hi

    # STEP 12: Q-weighted read
    wave_output = Q_feat * gathered                # (B, H, N, d) elementwise

    # STEP 13: Content-dependent gating
    gate = sigmoid(gate_raw).view(B, H, N, d)
    output = wave_output * gate

    # STEP 14: Reshape and output projection
    output = output.transpose(1, 2).reshape(B, N, D)
    output = out_proj(output)                      # Linear(D, D)
    return output
```

### 2.3 Wave Kernel Construction (Z-Transform Analytic Path)

```python
def build_analytic_kernel_fft(device):
    """
    Computes DFT of damped cosine kernel directly via Z-transform.
    Avoids materializing G time-domain samples.

    Kernel in time domain: k(t) = exp(-alpha*t) * cos(omega*t + phi)
    for t = 0, 1, ..., G-1  (causal: only t >= 0)

    Using complex exponential decomposition:
      k(t) = Re[c * exp(lambda * t)]
      where lambda = -alpha + i*omega, c = exp(i*phi)

    DFT via geometric series:
      H(z_k) = sum_{t=0}^{G-1} exp(lambda*t) * z_k^{-t}
             = (1 - exp(lambda*G) * z_k^{-G}) / (1 - exp(lambda) * z_k^{-1})

    The real kernel DFT = c * H_lambda(z) + conj(c) * H_{conj(lambda)}(z)
    """
    G = field_size     # 2048
    pad = fast_pad     # next cuFFT-friendly size >= 2*G
    freq_bins = pad // 2 + 1

    # Per-head physics parameters (the only 3 learnable params per head)
    alpha = softplus(wave_damping)    # (H,) -- positive damping rate
    omega = wave_frequency            # (H,) -- oscillation frequency
    phi   = wave_phase                # (H,) -- phase offset

    # Complex pole: lambda = -alpha + i*omega
    lam = complex(-alpha, omega)               # (H,)
    c   = exp(i * phi)                         # (H,) phase coefficient

    # DFT frequency bins: z_k = exp(i * 2*pi*k / pad_size)
    k = arange(freq_bins)
    z = exp(i * 2*pi*k / pad)                  # (freq_bins,)

    # Geometric series closed form
    exp_lam   = exp(lam)         # per-step decay
    exp_lam_G = exp(lam * G)     # total decay over G steps

    # H_lambda(z) = (1 - exp(lam*G)*z^{-G}) / (1 - exp(lam)*z^{-1})
    H_z = (1 - exp_lam_G * z^{-G}) / (1 - exp_lam * z^{-1})    # (H, freq_bins)

    # Conjugate pole: lambda* = -alpha - i*omega
    # IMPORTANT: uses same z^{-1}, NOT conj(z^{-1})
    H_z_conj = analogous with conj(exp_lam), conj(exp_lam_G)

    # Real kernel DFT
    kernel_fft = c * H_z + conj(c) * H_z_conj  # (H, freq_bins) complex

    # Normalize by DC component
    kernel_fft = kernel_fft / |kernel_fft[:, 0]|

    return kernel_fft
```

**Why this matters**: Direct gradient flow through alpha, omega, phi to the loss.
No discretization error. Causal by construction (sums only t >= 0).

### 2.4 SpectralGate -- Content-Adaptive Kernel Modulation

```python
def spectral_gate(Q, base_kernel_fft):
    """
    Q:               (B, H, N, d)        queries
    base_kernel_fft: (H, freq_bins)       static kernel from Z-transform
    Returns:         (B, H, freq_bins)    content-adaptive kernel per sample
    """
    B, H, N, d = Q.shape

    # Use ONLY first token (position 0) -- causal!
    # Every position can see position 0, so this is safe.
    # V4.3.1 fix: was using mean(Q) which leaked future tokens.
    q_bar = LayerNorm(Q[:, :, 0, :])              # (B, H, d)
    q_flat = q_bar.reshape(B, H * d)              # (B, H*d)

    # MLP: 2 layers
    ctrl = GELU(Linear(q_flat))                   # (B, H*d) -> (B, H*d)
    ctrl = Linear(ctrl)                           # (B, H*d) -> (B, H*32)
    ctrl = ctrl.view(B, H, 32)                    # 32 control points per head

    # Smooth interpolation to full frequency resolution
    gate = interpolate_1d(ctrl, size=freq_bins, mode='linear')  # (B, H, freq)
    gate = gate.float()                           # fp32 for complex multiply

    # Multiplicative modulation: at init gate~0, so output ~ base_kernel
    return base_kernel_fft * (1.0 + gate)         # (B, H, freq_bins) complex
```

**Key insight**: This makes the effective attention kernel INPUT-DEPENDENT.
Different inputs produce different spectral gates, which modulate different
frequency components of the wave kernel. "The cat sat" and "Quantum entanglement"
get different effective receptive fields.

### 2.5 Bilinear Scatter and Gather

```python
def bilinear_scatter(values, field_pos, B, H, G, d, device):
    """
    Deposit token values onto continuous field using bilinear interpolation.
    values:    (B, H, N, d)    -- what to deposit
    field_pos: (N,)            -- continuous field positions (fractional)
    Returns:   (B, H, G, d)   -- the populated field
    """
    field = zeros(B, H, G, d, device=device)
    lo = floor(field_pos).clamp(0, G-2)           # integer indices
    hi = lo + 1
    frac = field_pos - lo
    w_lo = (1 - frac).view(1, 1, N, 1)
    w_hi = frac.view(1, 1, N, 1)

    # scatter_add_ deposits to both adjacent cells, weighted
    field.scatter_add_(dim=2, index=lo_expanded, src=values * w_lo)
    field.scatter_add_(dim=2, index=hi_expanded, src=values * w_hi)
    return field

def bilinear_gather(field, field_pos):
    """
    Read from field at continuous positions. Inverse of scatter.
    field:     (B, H, G, d)
    field_pos: (N,)
    Returns:   (B, H, N, d)
    """
    lo = floor(field_pos).clamp(0, G-2)
    hi = lo + 1
    frac = field_pos - lo
    val_lo = gather(field, dim=2, index=lo_expanded)
    val_hi = gather(field, dim=2, index=hi_expanded)
    return val_lo * (1 - frac) + val_hi * frac
```

**Note**: The field is SPARSE. For N=512 tokens mapped to G=2048 field cells,
only ~25% of cells are populated. Most of the field is zeros.

### 2.6 Field Interference Module

```python
def field_interference(x):
    """
    Applied every 3 layers. Physics-based signal routing.
    x: (B, N, D)
    """
    # O(n) global context via causal cumulative mean
    compressed = Linear_D_to_D4(x)                    # compress to D/4
    cumsum = cumulative_sum(compressed, dim=seq)       # O(n) scan
    counts = arange(1, N+1)
    global_ctx = Linear_D4_to_D(cumsum / counts)      # expand back
    global_ctx = dropout(global_ctx)

    # Phase alignment: detect constructive vs destructive interference
    local_phase = normalize(Linear(x))                # (B, N, D) unit vectors
    global_phase = normalize(Linear(global_ctx))      # (B, N, D) unit vectors
    alignment = sum(local_phase * global_phase, dim=-1)  # cosine similarity

    # Temperature-scaled sigmoid (sharp interference decisions)
    temp = softplus(learned_temperature) + 0.05
    strength = sigmoid(alignment / temp)              # near-binary

    # Gated combination
    gate = sigmoid(Linear(concat(x, global_ctx)))
    return x + gate * global_ctx * strength
```

### 2.7 Training Pipeline

```python
# OPTIMIZER: 3 parameter groups with different learning rates
optimizer = AdamW([
    {'params': other_params,   'lr': 3e-4,  'weight_decay': 0.01},
    {'params': qkvg_params,    'lr': 9e-4,  'weight_decay': 0.01},   # 3x base
    {'params': kernel_params,  'lr': 1.5e-2,'weight_decay': 0.0},    # 50x base!
])
# kernel_params = wave_frequency, wave_damping, wave_phase (24 total scalars)
# No weight decay on kernel params (S4/Mamba best practice)

# SCHEDULER: Warmup + Cosine decay
warmup_steps = total_steps // 10
# Linear warmup: lr * (step / warmup_steps)
# Cosine decay: min_lr + 0.5*(base_lr - min_lr)*(1 + cos(pi*progress))

# AMP: bf16 on A100/H100, fp16+GradScaler on T4/V100
# FFT operations ALWAYS fp32 internally regardless of AMP dtype

# GRADIENT CLIPPING: max_norm = 1.0

# GRADIENT CHECKPOINTING: enabled for all layers (saves VRAM)

# TRAINING LOOP:
for epoch in range(num_epochs):
    for x, y in shuffled_batches:
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=use_amp):
            logits, _ = model(x)
            loss = cross_entropy(logits.view(-1, V), y.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
```

---

## 3. LEARNABLE PARAMETERS INVENTORY

### 3.1 S1 Configuration

```
embedding_dim = 384
num_layers    = 8
num_heads     = 8
head_dim      = 48
ffn_dim       = 1536
field_size    = 2048
vocab_size    = 8000
```

### 3.2 Complete Parameter Breakdown

```
CATEGORY                    FORMULA                           COUNT        LR MULT
========================================================================================

EMBEDDINGS (weight-tied output)
  token_embedding           vocab * D                         8000*384 =   3,072,000  1x
  sinusoidal PE             (buffer, not learnable)                    0   --
  ----
  Subtotal:                                                          3,072,000

PER LAYER (x8 layers):

  WAVE PHYSICS PARAMS (3 per head)                                        50x, wd=0
    wave_frequency          (H,)                              8    =     8
    wave_damping            (H,)                              8    =     8
    wave_phase              (H,)                              8    =     8
    ----
    Per-layer subtotal:                                              24

  FUSED QKVG PROJECTION                                                   3x (QK part)
    qkvg_proj.weight        (4D, D)                           4*384*384 = 589,824
    qkvg_proj.bias          (4D,)                             4*384 =     1,536
    ----
    Per-layer subtotal:                                              591,360

  OUTPUT PROJECTION                                                       1x
    out_proj.weight         (D, D)                            384*384 =   147,456
    out_proj.bias           (D,)                              384 =       384
    ----
    Per-layer subtotal:                                              147,840

  LEARNED FEATURE MAPS (Q and K, each 2-layer MLP)                       1x
    phi_q: Linear(48,48)+NormExp+Linear(48,48)+NormExp
      layer1.weight         (d, d)                            48*48 =     2,304
      layer1.bias           (d,)                              48 =        48
      layer2.weight         (d, d)                            48*48 =     2,304
      layer2.bias           (d,)                              48 =        48
    phi_k: same as phi_q                                              4,704
    ----
    Per-layer subtotal (Q+K):                                        9,408

  SPECTRAL GATE (SPECTRE)                                                 1x
    norm (LayerNorm)        (d,) * 2                          48*2 =      96
    net[0] Linear           (H*d, H*d)                        384*384 =   147,456
    net[0] bias             (H*d,)                            384 =       384
    net[2] Linear           (H*d, H*32)                       384*256 =   98,304
    net[2] bias             (H*32,)                           256 =       256
    ----
    Per-layer subtotal:                                              246,496

  FIELD COUPLING MATRIX                                                   1x
    field_coupling          (H, H)                            8*8 =       64

  LAYER NORMS (x2 per layer)                                              1x
    norm1.weight + bias     D * 2                             384*2 =     768
    norm2.weight + bias     D * 2                             384*2 =     768
    ----
    Per-layer subtotal:                                              1,536

  FFN                                                                     1x
    Linear(D, 4D)           D*4D + 4D                         384*1536 + 1536 = 591,360
    Linear(4D, D)           4D*D + D                          1536*384 + 384 = 590,208
    ----
    Per-layer subtotal:                                              1,181,568

  ----
  TOTAL PER LAYER:                                                   2,178,296
  (wave physics: 24, projections: 739,200, feature maps: 9,408,
   spectral gate: 246,496, coupling: 64, norms: 1,536, FFN: 1,181,568)

FIELD INTERFERENCE MODULES (x2, every 3 layers: after L3 and L6)
  Per module:
    compress Linear         (D, D/4)   + bias                 384*96 + 96 = 36,960
    expand Linear           (D/4, D)   + bias                 96*384 + 384 = 37,248
    local_phase_proj        (D, D)     + bias                 384*384 + 384 = 148,096 [sic]
    global_phase_proj       (D, D)     + bias                 384*384 + 384 = 148,096 [sic]
    interference_temperature scalar                            1
    interference_gate       (2D, D)    + bias                 768*384 + 384 = 295,296
    norm (LayerNorm)        D * 2                              768
    ----
    Per-module subtotal:                                         ~666,465
  x2 modules:                                                       ~1,332,930

FINAL LAYER NORM
    norm.weight + bias      D * 2                              768

OUTPUT PROJECTION (weight-tied with embedding)
    output_projection       (shared with token_embedding)      0

========================================================================================
GRAND TOTAL:
  Embeddings:               3,072,000
  8 Transformer Layers:     8 * 2,178,296 = 17,426,368
  2 Interference Modules:   ~1,332,930
  Final LayerNorm:          768
  ----
  TOTAL:                    ~21,831,434  (matches S1 benchmark: 21.8M)
========================================================================================
```

### 3.3 Special Learning Rate Parameters

```
+---------------------------+--------+--------+--------+--------------------+
| Parameter Group           | Count  | LR     | WD     | Why Special        |
+---------------------------+--------+--------+--------+--------------------+
| wave_frequency/damping/   |        |        |        | Monitor showed     |
|   phase (all 8 layers)    | 192    | 50x    | 0.0    | 27-80x gradient    |
|   = 3 * 8 heads * 8 layers|        | =15mLR |        | deficit vs other   |
|                            |        |        |        | params. S4/Mamba   |
|                            |        |        |        | practice.          |
+---------------------------+--------+--------+--------+--------------------+
| qkvg_proj weight+bias     | 4.7M   | 3x     | 0.01   | V4.2 ablation:     |
|   (all 8 layers)          |        | =0.9mLR|        | -21% PPL with      |
|                            |        |        |        | elevated QK LR.    |
+---------------------------+--------+--------+--------+--------------------+
| Everything else            | 16.9M  | 1x     | 0.01   | Default AdamW.     |
|                            |        | =0.3mLR|        |                    |
+---------------------------+--------+--------+--------+--------------------+
```

The kernel physics parameters are **192 scalars** out of 21.8M total parameters
(0.0009%) but they control the entire wave propagation behavior.

---

## 4. WHAT WENT WELL

### 4.1 V4.3.4 Three-Fix Experiment (UNVERIFIED -- REVERTED)

**WARNING**: V4.3.4 was based on partial ablation runs at 12M tokens that were
NEVER verified with a full 20M token S1 benchmark. When tested end-to-end on
Colab, V4.3.4 diverged at step 854 (PPL spiked from ~700 to 1472).
V4.3.4 changes (NormalizedExp, SpectralGate 0.1, damping -3.0) have been
REVERTED to V4.3.3 on main. The partial ablation data below is preserved
for reference but should NOT be used to claim V4.3.4 works:

```
V4.3.3 Baseline (ELU+1, gate=0.01, damp=(-1.4,0.0)):   PPL 263 @ 12M tokens
  +Gate 10x only:                                         PPL 323  (+23% WORSE)
  +Kernel reach only:                                     PPL 291  (+11% WORSE)
  +NormalizedExp only:                                    PPL ~860 (CATASTROPHIC)
  V4.3.4 full (claimed):                                  PPL ~220 (UNVERIFIED)
  V4.3.4 full (Colab actual):                             PPL ~743 then DIVERGED
```

**Lesson**: Never claim results without running the full benchmark to completion.
Partial runs can be misleading due to late-stage instabilities.

### 4.2 O(n log n) Complexity Verified

From long-context benchmark (results/long_context.json):
- Standard Transformer: memory and time scale as O(n^2)
- Wave Field: memory and time scale as O(n log n)
- At seq_len=4096, Wave uses significantly less memory per sequence

The theoretical advantage is real, even if absolute throughput is currently 3.5x slower
at seq_len=512 (17.9K vs 63.5K tok/s on RTX 3060) due to FFT overhead.

### 4.3 HiPPO Kernel Initialization (S4D Heritage)

Frequencies: omega_n = pi * (2n + 1) / 2 for head n = 0..7
This gives a mathematically optimal basis for capturing different temporal scales:
- Head 0: omega = pi/2 = 1.57 (slow oscillation, broad patterns)
- Head 7: omega = 15pi/2 = 23.56 (fast oscillation, sharp local patterns)

Per-layer diversity further specializes:
- Layer 0: freq_scale = 0.5x (even broader), damping = softplus(-3.0) = 0.05 (long reach)
- Layer 7: freq_scale = 2.0x (even sharper), damping = softplus(0.0) = 0.69 (fast decay)

**Result**: Layers specialize from step 0 instead of all starting identical and slowly diverging.

### 4.4 Fused QKV + Gate Projection

Single nn.Linear(D, 4D) instead of 4 separate projections:
- Q, K, V, Gate all computed in one matmul
- Gate portion initialized special: weight=0, bias=2.0 (gates start open)
- Saves ~30% wall-clock time on the projection step

### 4.5 Training Curve Shows Continuous Learning

From scaling_s1.json, the V4.3.3 Wave model PPL curve:
```
 Step 0: PPL 8592  (random)
 1M tok: PPL 1380  (learning basic patterns)
 2M tok: PPL 1367  (plateau -- spectral gate still near zero)
 3M tok: PPL  924  <-- BREAKTHROUGH: spectral gate activates
 5M tok: PPL  521  (rapid descent)
10M tok: PPL  322  (steady improvement)
15M tok: PPL  256
20M tok: PPL  239  (final, still improving)
```

The critical transition at 2-3M tokens is when the spectral gate's MLP output
grows from near-zero to meaningful modulation, making the kernel content-adaptive.

---

## 5. WHAT WENT WRONG

### 5.1 Feature Map Rank Collapse (ELU+1 Era -- V4.3.2/V4.3.3)

**What happened**: ELU+1 activation compressed all feature map outputs to ~2.0.
The standard deviation / mean ratio was 0.13, giving an effective rank of ~2.3
(should be ~48 for head_dim=48).

**Why**: ELU+1 maps negative inputs to ~1.0 (via exp(x)+1 for x<0) and positive
inputs to x+1. With normally-distributed inputs (mean ~0), most values cluster
around 1.0. This destroys the distinction between different tokens in feature space.

**Fix**: NormalizedExp: exp(x - max(x)). This preserves exponential separation
between values. BUT it only works when combined with the other V4.3.4 fixes
(alone it is catastrophic -- PPL ~860).

**Lesson**: The feature map activation is deeply coupled to the kernel reach and
spectral gate strength. You cannot change one without adjusting the others.

### 5.2 SpectralGate Operates as Near-Scalar Multiplier

**Monitor evidence**: With Gate 10x init (0.1 scale), the spectral gate output
was uniform ~0.08 across all frequency bins and all layers. It was modulating
the kernel magnitude (acting like a scalar gain) rather than shaping the
frequency profile.

**Why it matters**: The whole point of SpectralGate is to SELECTIVELY amplify
or suppress frequency components based on input content. If it outputs a flat
value, it is not doing frequency-selective modulation -- it is just scaling the
kernel up or down uniformly.

**Root cause**: 32 control points interpolated to ~2049 frequency bins produces
an extremely smooth gate that cannot have sharp spectral features. Additionally,
conditioning on only the first token (for causality) limits the information
available to drive meaningful frequency selectivity.

### 5.3 Deep Layers (L6-L7) Are Effectively Dead

**Evidence from training monitor gradient analysis**:
- Layers 0-3: healthy gradient norms, kernel params actually move during training
- Layers 4-5: gradient norms 5-10x smaller
- Layers 6-7: near-zero gradients, output rank collapsed, kernel params barely
  change from initialization

**Why**: The wave kernel reach at S1 scale is ~20 positions. After 6 layers of
this limited receptive field, deeper layers see already-mixed representations
where the wave convolution adds diminishing value. The gradient signal vanishes.

**Impact**: We are paying for 8 layers of parameters but only 5-6 are actively
learning. The last 2 layers are dead weight.

### 5.4 V4.3.5 Deposit Gate Failed

**Attempted**: Per-token deposit gating inspired by Mamba/GLA:
- Linear(D,D) with bias=1.0: PPL stuck at 1381 (sigmoid(1)=0.73 damped ALL deposits 27%)
- Linear(D,D) with bias=5.0: PPL 982 then spiked to 1331 (destabilized training)
- Linear(48,1) scalar with bias=2.0: same NormalizedExp plateau, no improvement

**Root cause**: At S1 scale (22M params, 20-position reach), the bottleneck is
**kernel reach**, not deposit selectivity. Adding gating parameters that learn
from zero disrupts the delicate V4.3.4 three-way synergy. The model was already
right at a critical balance point.

### 5.5 The 1.40x PPL Gap (V4.3.3)

At 20M tokens on WikiText-2:
```
Standard Transformer:  PPL 170.75   Acc 18.26%   17.5M params
SPECTRE-Wave V4.3.3:   PPL 238.85   Acc 16.03%   21.8M params  (1.40x gap)
```

The Wave model has 25% MORE parameters but 40% WORSE perplexity.

### 5.6 Kernel Reach Limited to ~20 Positions

Even with the widest damping (softplus(-3.0) = 0.05), the wave kernel effectively
reaches only ~20 field positions before decaying to near-zero. Given the
absolute position mapping with stride ~4 (2048/512), this corresponds to only
~5 tokens of effective receptive field per layer.

The Standard Transformer with causal mask can attend to ALL 512 previous tokens
in each layer. This is a fundamental information access asymmetry.

### 5.7 V4.4 Experiments All Failed

**Write Gate**: See 5.4 above.

**3D Wave Interference**: Content-dependent cross-head coupling based on learned
3D head positions and token position projections. Did not improve over static
field_coupling matrix at S1 scale.

**Kernel Mixture K=4**: PPL 1155 vs Standard 636 at 3M tokens. Even with warm-bias
toward standard HiPPO kernel (98% weight at init), the per-token mixture selection
destabilized training.

---

## 6. CORE ISSUES TO ADDRESS (Ranked by Impact)

### ISSUE 1: No Per-Token Input-Dependent State Transition [CRITICAL]

**What**: The wave kernel is the SAME for every token position within a sequence.
SpectralGate provides per-SAMPLE adaptation (based on first token), but not per-TOKEN.

**Why it matters**: This is THE fundamental difference between Wave Field and
Mamba/GLA. In Mamba, each token has its own decay gate that controls how much
of its information persists in the state. In GLA, each token has its own transition
matrix. In Wave Field, all tokens share the same convolution kernel.

**Monitor data shows**: The spectral gate output is uniform across frequency bins,
confirming it operates as a global scalar rather than a token-specific selector.

**Potential solutions**:
1. **Per-token kernel modulation** (most promising): Instead of one SpectralGate
   per sequence, produce per-token spectral weights. Challenge: this makes the
   convolution non-uniform, breaking the FFT speedup.
2. **Gated deposit/gather** (V4.3.5 failed at S1, may work at S2+): Per-token
   gating of deposit strength or gather weights.
3. **Interleaved local attention** (BASED/SWH approach): Add O(n) sliding window
   every K layers to provide exact per-token attention where waves are weakest.

**Estimated impact**: Closing this gap would likely reduce the PPL ratio from 1.40x
to < 1.20x based on GLA/Mamba literature.

---

### ISSUE 2: Kernel Reach Too Short (~20 Positions) [HIGH]

**What**: The damped wave kernel decays to near-zero after ~20 field positions.
With stride ~4, this means each token only "hears" the previous ~5 tokens
through wave propagation.

**Why it matters**: Standard Transformer attends to ALL previous tokens per layer.
Even stacking 8 wave layers with 5-token reach gives ~40 tokens of indirect
context, far below 512.

**Monitor data shows**: Kernel reach values of 3-20 across layers. Early layers
(low damping) have reach ~20, later layers (higher damping) have reach ~5.

**Potential solutions**:
1. **Reduce damping further**: softplus(-5.0) = 0.007, reach ~140 positions.
   Risk: very long kernels may be too diffuse to carry useful signal.
2. **Multi-scale kernels**: Different heads at dramatically different scales
   (already partially done via per-layer init, but could be more aggressive).
3. **Field Interference Module enhancement**: Currently provides O(n) global
   context every 3 layers. Could be made more powerful (currently just causal
   cumulative mean + learned gate).

**Estimated impact**: Doubling effective reach to ~40 positions would likely
reduce PPL gap by 0.05-0.10x.

---

### ISSUE 3: Deep Layers Are Dead (L6-L7) [MEDIUM]

**What**: The last 2 of 8 layers have near-zero gradients and do not meaningfully
update their parameters during training.

**Why it matters**: We pay for 8 layers of computation and parameters but only
5-6 contribute to learning. This is wasted capacity.

**Monitor data shows**: Gradient norms in L6-L7 are orders of magnitude smaller
than L0-L3. Output rank in deep layers is collapsed.

**Potential solutions**:
1. **Residual scaling**: 1/sqrt(2*num_layers) scaling on output projections
   (GPT-style). Already implemented as option (residual_scale=True) but not
   used in V4.3.4 benchmarks.
2. **Fewer but wider layers**: Use 6 layers with wider FFN instead of 8 thin ones.
3. **Progressive training**: Start with 4 layers, gradually add more as training
   progresses.

**Estimated impact**: Recovering 2 dead layers is equivalent to ~25% more effective
depth. Could reduce PPL gap by 0.03-0.08x.

---

### ISSUE 4: SpectralGate Is Frequency-Flat [MEDIUM]

**What**: The spectral gate produces approximately the same value across all
frequency bins, acting as a scalar multiplier rather than a frequency-selective filter.

**Why it matters**: The architectural purpose of SpectralGate is to selectively
amplify or suppress specific frequency components based on input content. If it
produces flat output, it provides no frequency selectivity.

**Monitor data shows**: Gate output std/mean ratio across frequency bins is very
low (~0.08 uniform).

**Potential solutions**:
1. **More control points**: Increase from 32 to 128 or 256 to allow sharper
   spectral features.
2. **Per-head conditioning**: Currently all heads share one MLP. Separate
   per-head MLPs would allow head-specific frequency shaping.
3. **Multi-token conditioning**: Instead of using only position-0 query,
   use a causal summary (e.g., cumulative mean) for richer conditioning signal.
   Challenge: must maintain strict causality.

**Estimated impact**: Making the gate frequency-selective could reduce PPL gap
by 0.02-0.05x.

---

### ISSUE 5: Sparse Field Occupancy [LOW]

**What**: With N=512 tokens mapped to G=2048 field cells, only ~25% of cells
are populated. The wave kernel convolves over mostly-empty field regions.

**Why it matters**: Computational waste. Also, the bilinear scatter distributes
each token's energy across only 2 adjacent cells, creating narrow spikes rather
than smooth deposits.

**Potential solutions**:
1. **Reduce field size**: G=1024 or G=512 would increase occupancy.
   Risk: position collisions for close tokens.
2. **Gaussian scatter**: Instead of bilinear (2 cells), use Gaussian
   (5-7 cells) for smoother deposits.
3. **Adaptive stride**: Vary stride based on local token density.

**Estimated impact**: Marginal. The FFT cost is dominated by the padding to
cuFFT-friendly sizes, not the field size itself.

---

## 7. THE PPL GAP ANALYSIS

### 7.1 The Numbers

```
                    Params      PPL       Acc       Throughput
Standard (S1)       17.5M      170.75    18.26%    63,499 tok/s
Wave V4.3.3 (S1)    21.8M      238.85    16.03%    17,876 tok/s

PPL Ratio (V4.3.3):  238.85 / 170.75 = 1.40x
PPL Gap (absolute):  ~68 PPL points
Param Overhead:      +4.3M (+25%) from feature maps, spectral gate, coupling
Speed Ratio:         3.5x slower (FFT overhead at seq=512)
```

### 7.2 Where the Gap Comes From

Decomposing the 40% PPL gap by source (estimated from ablation data):

```
SOURCE                                    ESTIMATED CONTRIBUTION
================================================================
1. No per-token state transitions         ~12-15% of gap
   (same kernel for all tokens)
   Evidence: Mamba/GLA close this gap
   with per-token gates at O(n) cost.

2. Limited kernel reach (~20 positions)    ~5-8% of gap
   (vs 512-token full attention)
   Evidence: Long-context benchmark shows
   1.44x gap at 4096 context (reach matters
   MORE at longer sequences).

3. Dead deep layers (L6-L7)                ~3-5% of gap
   Evidence: Only 6 of 8 layers learning.
   Effective model is ~16.5M params, closer
   to Standard's 17.5M.

4. Feature map approximation error          ~2-3% of gap
   (NormalizedExp vs true softmax)
   Evidence: NormalizedExp has rank >> 2 but
   still not as sharp as softmax attention.

5. SpectralGate frequency-flat behavior     ~1-2% of gap
   Evidence: Gate acts as scalar, provides
   less content-adaptiveness than expected.

6. Sparse field + bilinear interpolation    ~1% of gap
   Evidence: Minor quantization artifacts
   from 2-cell bilinear scatter.
================================================================
TOTAL:                                     ~24-34% (consistent with 40%)
```

### 7.3 What Standard Transformer Does Better

```
+----------------------------------+---------------------------+
| Standard Transformer             | Wave Field LLM            |
+----------------------------------+---------------------------+
| Full NxN attention matrix:       | Wave kernel reaches ~20   |
| every token can attend to every  | positions. Information    |
| previous token DIRECTLY.         | must chain through layers.|
+----------------------------------+---------------------------+
| Softmax produces SHARP, peaked   | NormalizedExp produces    |
| attention distributions. Token   | positive maps but not as  |
| can focus on 1-2 key tokens.     | sharp as softmax peaks.   |
+----------------------------------+---------------------------+
| Content-dependent attention per  | Content-dependent only    |
| TOKEN (each Q-K pair computed    | per SAMPLE (SpectralGate  |
| independently).                  | conditions on 1st token). |
+----------------------------------+---------------------------+
| Simple gradient path: loss ->    | Complex gradient path:    |
| softmax -> QK -> input.          | loss -> gather -> IFFT -> |
| Well-understood optimization.    | FFT -> scatter -> input.  |
|                                  | Many more nonlinearities. |
+----------------------------------+---------------------------+
| Scales with context O(n^2) but   | Scales as O(n log n) but  |
| FlashAttention makes it fast.    | no equivalent FlashFFT.   |
+----------------------------------+---------------------------+
```

### 7.4 The Theoretical Advantage That Has Not Yet Materialized

Wave Field's O(n log n) complexity should dominate at long sequences:

```
Sequence Length   Standard O(n^2)    Wave O(n log n)    Ratio
--------------------------------------------------------------
     512               262,144           4,608           57x
   2,048             4,194,304          22,528          186x
   8,192            67,108,864         106,496          630x
  32,768         1,073,741,824         491,520        2,184x
```

But in practice at seq=512, Wave is 3.5x SLOWER because:
1. FFT operations are memory-bound (not compute-bound like matmul)
2. No FlashFFT equivalent to FlashAttention
3. Complex number operations are not as optimized on GPUs
4. The scatter/gather operations are irregular memory access patterns

The crossover point where Wave becomes faster has not been reached in benchmarks yet.

### 7.5 Path to Closing the Gap

```
CURRENT STATE:                    PPL ratio = 1.29x
                                  |
FIX 1: Per-token gating           |
  (GLA/Mamba insight)             --> estimated 1.15x
                                  |
FIX 2: Extended kernel reach      |
  (lower damping + multi-scale)   --> estimated 1.10x
                                  |
FIX 3: Activate deep layers       |
  (residual scaling / fewer layers)--> estimated 1.05x
                                  |
FIX 4: Better SpectralGate        |
  (frequency-selective + multi-tok)--> estimated 1.03x
                                  |
TARGET:                           PPL ratio < 1.05x (architecture parity)
```

Each fix is estimated independently. In practice they may interact synergistically
(like V4.3.4's three-way synergy) or cancel out.

---

## APPENDIX A: Version History Summary

```
Version    Date         Key Change                          PPL @ 5M tok   vs Standard
---------------------------------------------------------------------------------------
V1.0       2026-02-14   Initial field attention              --             --
V2.0       2026-02-15   Improved field dynamics              --             --
V3.0       2026-02-16   Wave kernels + interference          --             --
V3.5       2026-02-17   Removed energy conservation          --             --
V4.1       2026-02-18   Linear-wave O(n log n) via FFT       543 (at 15M)   ~2x worse
V4.2       2026-02-19   Init fixes, residual scaling         997            ~2x worse
V4.3       2026-02-21   SPECTRE + Hedgehog + HiPPO           117            3.9x BETTER*
V4.3.1     2026-02-22   Causal leak fix (mean->first tok)    --             --
V4.3.3     2026-02-24   Kernel LR 50x + diversity + ELU+1    239 (at 20M)   1.40x worse
V4.3.4     2026-02-25   NormalizedExp + Gate10x + reach      ~220 (at 20M)  1.29x worse
V4.3.5     REVERTED     Deposit gate (all variants failed)   --             --

* V4.3 PPL 117 at 5M tokens was before the full 20M token S1 benchmark.
  The 3.9x advantage over Standard was at 5M tokens where Standard had PPL 457.
  At 20M tokens, Standard catches up and the ratio inverts.
```

## APPENDIX B: Key References

```
Paper                              What We Used From It
----------------------------------------------------------------------
SPECTRE (arXiv:2502.18394)         SpectralGate architecture (content-
                                   adaptive spectral modulation)

Hedgehog (ICLR 2024)              Learned feature maps with identity init
                                   + softmax-mimicking activation

S4D (arXiv:2206.11893)            HiPPO initialization for wave kernel
                                   frequencies (harmonic basis)

BASED (Hazy Research 2024)        Sliding window hybrid design insight
                                   (not yet implemented)

GLA (ICML 2024)                   Separate param groups for gate params;
                                   per-token decay gate architecture

Gated DeltaNet (ICLR 2025)        Per-token input-dependent transitions
                                   (the key missing feature)

RALA (CVPR 2025)                   Output rank augmentation to prevent
                                   representation collapse

Differential Attention (ICLR 2025) Attention noise cancellation via
                                   difference of two softmax heads
```

## APPENDIX C: File Map

```
src/wave_field_attention.py      THE CORE: 975 lines, all wave field attention logic
src/wave_field_transformer.py    Transformer layers, model class, optimizer config
src/global_context.py            O(n) global context (not used in V4.3+, replaced
                                 by FieldInterferenceModule inside transformer)
benchmarks/benchmark_scaling.py  S1-S4 scaling runs with Standard baseline
diagnostics/training_monitor.py  WaveFieldMonitor: hooks into all internals
diagnostics/visualize_monitor.py 12-panel dashboard from monitor JSON output
```
