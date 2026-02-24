"""VRAM Calculator for Wave Field Transformer V4.3 scaling on RTX 3060 Laptop (6.4GB)"""
import math

def count_params(vocab_size, dim, layers, heads, ffn_dim, field_size, seq_len=512, interference_interval=3):
    head_dim = dim // heads

    # === EMBEDDING (weight-tied with output) ===
    embedding = vocab_size * dim

    # === PER ATTENTION LAYER (WaveFieldAttention) ===
    # qkvg_proj: Linear(dim, 4*dim)
    qkvg = dim * 4 * dim + 4 * dim
    # out_proj: Linear(dim, dim)
    out_proj = dim * dim + dim
    # q_feature_map: Linear(head_dim, head_dim) + bias
    q_feat_map = head_dim * head_dim + head_dim
    # k_feature_map: same
    k_feat_map = head_dim * head_dim + head_dim

    # SpectralGate:
    #   norm: LayerNorm(head_dim) => 2 * head_dim
    #   net[0]: Linear(dim, dim)
    #   net[2]: Linear(dim, heads*32)
    n_control = 32
    sg_norm = 2 * head_dim
    sg_linear1 = dim * dim + dim
    sg_linear2 = dim * (heads * n_control) + (heads * n_control)
    spectral_gate = sg_norm + sg_linear1 + sg_linear2

    # wave_frequency, wave_damping, wave_phase: each (H,) for n_components=1
    wave_params = 3 * heads
    # field_coupling: (H, H)
    field_coupling = heads * heads

    attn_per_layer = qkvg + out_proj + q_feat_map + k_feat_map + spectral_gate + wave_params + field_coupling

    # === PER FFN LAYER ===
    ffn_per_layer = (dim * ffn_dim + ffn_dim) + (ffn_dim * dim + dim)

    # === PER TRANSFORMER LAYER ===
    # norm1, norm2: each LayerNorm(dim) => 2*dim each
    norms_per_layer = 2 * (2 * dim)
    layer_total = attn_per_layer + ffn_per_layer + norms_per_layer

    # === INTERFERENCE MODULES ===
    num_interference = layers // interference_interval
    interf_per = (
        (dim * dim + dim) +           # local_phase_proj
        (dim * dim + dim) +           # global_phase_proj
        1 +                           # interference_temperature
        (2 * dim * dim + dim) +       # interference_gate
        (dim * (dim // 4) + dim // 4) +  # compress
        ((dim // 4) * dim + dim) +    # expand
        2 * dim                       # norm
    )
    interference_total = num_interference * interf_per

    # === OUTPUT ===
    output_norm = 2 * dim

    total = embedding + layers * layer_total + interference_total + output_norm
    return total


def calc_vram(name, vocab_size, dim, layers, heads, ffn_dim, field_size, seq_len=512):
    params = count_params(vocab_size, dim, layers, heads, ffn_dim, field_size, seq_len)
    head_dim = dim // heads
    padded_field = 2 * field_size

    print(f"\n{'='*70}")
    print(f"Scale {name}: dim={dim}, layers={layers}, heads={heads}, ffn={ffn_dim}, field={field_size}")
    print(f"{'='*70}")
    print(f"Parameters: {params:,} ({params/1e6:.1f}M)")

    # 1. Model weights (fp16)
    weights_bytes = params * 2
    print(f"\n1. Model weights (fp16):         {weights_bytes / 1024**2:>8.1f} MB")

    # 2. Optimizer states: fp32 master weights + momentum + variance = 12 bytes/param
    optimizer_bytes = params * 12
    print(f"2. Optimizer states (AdamW fp32): {optimizer_bytes / 1024**2:>8.1f} MB")

    # 3. Gradients (fp16)
    grad_bytes = params * 2
    print(f"3. Gradients (fp16):             {grad_bytes / 1024**2:>8.1f} MB")

    # Fixed overhead (model + optimizer + gradients)
    fixed_bytes = weights_bytes + optimizer_bytes + grad_bytes
    print(f"--- Fixed total:                 {fixed_bytes / 1024**2:>8.1f} MB  ({fixed_bytes/1024**3:.2f} GB)")

    # 4. Activation memory per batch element per layer
    stored_layers = math.ceil(math.sqrt(layers))

    # Per layer activations:
    # a) Input to layer: seq * dim * 2 (fp16)
    input_act = seq_len * dim * 2
    # b) QKVG projection output: seq * 4*dim * 2
    qkvg_act = seq_len * 4 * dim * 2
    # c) Q,K,V reshaped (views but kept for backward): 3 * heads * seq * head_dim * 2
    qkv_act = 3 * heads * seq_len * head_dim * 2
    # d) Feature map outputs: 2 * heads * seq * head_dim * 2
    feat_map_act = 2 * heads * seq_len * head_dim * 2
    # e) Deposit (k_feat * v): heads * seq * head_dim * 2
    deposit_act = heads * seq_len * head_dim * 2
    # f) Field after scatter: heads * field_size * head_dim * 2
    field_act = heads * field_size * head_dim * 2
    # g) FFT intermediates (complex64 = 8 bytes, float32 = 4 bytes):
    #    _wave_convolve: reshape to (B*D, H, G), rfft(n=2G) => (B*D, H, G+1) complex64
    #    So per sample: head_dim * heads * (field_size+1) * 8
    fft_field = head_dim * heads * (field_size + 1) * 8
    #    kernel_fft (spectral gate output): heads * (field_size+1) * 8
    fft_kernel = heads * (field_size + 1) * 8
    #    convolved_fft: same shape as fft_field
    fft_convolved = head_dim * heads * (field_size + 1) * 8
    #    irfft output before slice: head_dim * heads * padded_field * 4 (float32)
    irfft_out = head_dim * heads * padded_field * 4
    fft_total = fft_field + fft_kernel + fft_convolved + irfft_out

    # h) Gathered output: heads * seq * head_dim * 2
    gather_act = heads * seq_len * head_dim * 2
    # i) Gate + gated output
    gate_act = seq_len * dim * 2 + heads * seq_len * head_dim * 2
    # j) FFN intermediate: seq * ffn_dim * 2
    ffn_act = seq_len * ffn_dim * 2
    # k) Spectral gate intermediates
    sg_act = heads * 32 * 4 + heads * (field_size + 1) * 4

    per_layer_per_sample = (input_act + qkvg_act + qkv_act + feat_map_act + deposit_act +
                            field_act + fft_total + gather_act + gate_act + ffn_act + sg_act)

    print(f"\n4. Activation memory per layer per sample:")
    print(f"   Input:           {input_act/1024**2:>7.3f} MB")
    print(f"   QKVG proj:       {qkvg_act/1024**2:>7.3f} MB")
    print(f"   QKV reshaped:    {qkv_act/1024**2:>7.3f} MB")
    print(f"   Feature maps:    {feat_map_act/1024**2:>7.3f} MB")
    print(f"   Deposit:         {deposit_act/1024**2:>7.3f} MB")
    print(f"   Field (scatter): {field_act/1024**2:>7.3f} MB")
    print(f"   FFT total:       {fft_total/1024**2:>7.3f} MB   <-- DOMINANT")
    print(f"   Gathered:        {gather_act/1024**2:>7.3f} MB")
    print(f"   Gate:            {gate_act/1024**2:>7.3f} MB")
    print(f"   FFN:             {ffn_act/1024**2:>7.3f} MB")
    print(f"   Spectral gate:   {sg_act/1024**2:>7.3f} MB")
    print(f"   --- Per layer:   {per_layer_per_sample/1024**2:>7.2f} MB/sample")

    print(f"   Stored layers (grad ckpt): {stored_layers} of {layers}")
    act_per_sample = per_layer_per_sample * stored_layers
    print(f"   --- Total per sample (ckpt): {act_per_sample/1024**2:.2f} MB")

    # 5. Recommended batch size
    vram_total = 6.4 * 1024**3
    vram_overhead = 0.5 * 1024**3  # PyTorch context, CUDA kernels, fragmentation
    vram_available = vram_total - vram_overhead - fixed_bytes

    if vram_available <= 0:
        max_batch = 0
        recommended = 0
        print(f"\n5. DOES NOT FIT: Fixed costs ({fixed_bytes/1024**3:.2f} GB) exceed available VRAM")
    else:
        max_batch = int(vram_available / act_per_sample)
        recommended = max(1, int(max_batch * 0.8))
        print(f"\n5. Available for activations: {vram_available/1024**3:.2f} GB")
        print(f"   Max batch size: {max_batch}")
        print(f"   Recommended batch (80% headroom): {recommended}")

    # 6. Fit assessment
    print(f"\n6. Fit assessment:")
    print(f"   Fixed overhead:  {fixed_bytes/1024**3:.2f} GB")
    print(f"   Per-sample acts: {act_per_sample/1024**2:.1f} MB")
    if max_batch >= 8:
        verdict = "COMFORTABLE FIT"
        print(f"   Verdict: {verdict} (batch {max_batch}+)")
    elif max_batch >= 4:
        verdict = "SMALL BATCHES"
        print(f"   Verdict: {verdict} (batch {max_batch}, use grad accum for effective >= 16)")
    elif max_batch >= 1:
        verdict = "TIGHT FIT"
        print(f"   Verdict: {verdict} (batch {max_batch}, need aggressive grad accum)")
    else:
        verdict = "NO FIT"
        print(f"   Verdict: {verdict} even at batch=1")

    # Dataset recommendations
    chinchilla_tokens = params * 20
    practical_tokens = params * 10
    min_useful_tokens = params * 5
    print(f"\n   Chinchilla-optimal tokens: {chinchilla_tokens/1e6:.0f}M")
    print(f"   Practical minimum tokens:  {practical_tokens/1e6:.0f}M")
    print(f"   Bare minimum tokens:       {min_useful_tokens/1e6:.0f}M")

    wt2_ok = min_useful_tokens <= 2.6e6
    wt103_ok = min_useful_tokens <= 103e6
    print(f"   WikiText-2 (2.6M tokens): {'OK' if wt2_ok else 'TOO SMALL'}")
    print(f"   WikiText-103 (103M tokens): {'OK' if wt103_ok else 'TOO SMALL'}")

    return {
        'params': params,
        'fixed_gb': fixed_bytes / 1024**3,
        'act_per_sample_mb': act_per_sample / 1024**2,
        'max_batch': max_batch,
        'recommended': recommended,
        'verdict': verdict,
        'chin': chinchilla_tokens,
        'prac': practical_tokens,
        'mini': min_useful_tokens,
        'wt2': wt2_ok,
        'wt103': wt103_ok,
    }


if __name__ == '__main__':
    vocab = 8192
    seq = 512

    print("VRAM ANALYSIS FOR RTX 3060 LAPTOP GPU (6.4 GB)")
    print("AMP (fp16 forward, fp32 optimizer), AdamW, Gradient Checkpointing")
    print(f"Vocab: {vocab}, Seq: {seq}")

    results = {}
    results['current'] = calc_vram('CURRENT (8.5M)', vocab, 256, 6, 8, 1024, 1024, seq)
    results['A'] = calc_vram('A (25M)', vocab, 384, 8, 8, 1536, 1536, seq)
    results['B'] = calc_vram('B (50M)', vocab, 512, 12, 8, 2048, 2048, seq)
    results['C'] = calc_vram('C (100M)', vocab, 768, 16, 12, 3072, 2048, seq)
    results['D'] = calc_vram('D (200M)', vocab, 1024, 24, 16, 4096, 2048, seq)

    # ====== SUMMARY TABLE ======
    print(f"\n\n{'='*95}")
    print(f"SUMMARY TABLE -- RTX 3060 Laptop (6.4 GB VRAM)")
    print(f"{'='*95}")
    hdr = f"{'Scale':<12} {'Params':<10} {'Fixed(GB)':<10} {'Act/samp':<12} {'MaxBatch':<10} {'RecBatch':<10} {'Verdict':<20}"
    print(hdr)
    print(f"{'-'*95}")

    for key, label in [('current','Current'), ('A','A 25M'), ('B','B 50M'), ('C','C 100M'), ('D','D 200M')]:
        r = results[key]
        print(f"{label:<12} {r['params']/1e6:>7.1f}M  {r['fixed_gb']:>7.2f}    {r['act_per_sample_mb']:>8.1f} MB  {r['max_batch']:>8}    {r['recommended']:>8}    {r['verdict']}")

    print(f"\n{'='*95}")
    print("DATASET RECOMMENDATIONS")
    print(f"{'='*95}")
    print(f"{'Scale':<12} {'Chinchilla':<15} {'Practical':<15} {'Minimum':<15} {'WT-2':<12} {'WT-103':<12}")
    print(f"{'-'*95}")
    for key, label in [('current','Current'), ('A','A 25M'), ('B','B 50M'), ('C','C 100M'), ('D','D 200M')]:
        r = results[key]
        wt2 = "OK" if r['wt2'] else "TOO SMALL"
        wt103 = "OK" if r['wt103'] else "TOO SMALL"
        print(f"{label:<12} {r['chin']/1e6:>10.0f}M    {r['prac']/1e6:>10.0f}M    {r['mini']/1e6:>10.0f}M    {wt2:<12} {wt103:<12}")

    print(f"\n{'='*95}")
    print("STRETCH LIMIT ANALYSIS")
    print(f"{'='*95}")
    print("""
The RTX 3060 Laptop 6.4GB stretch limit depends on willingness to sacrifice batch size.

Comfortable training (batch >= 8):     ~25M params (Scale A)
Workable training (batch 4-8):          ~50M params (Scale B)
Aggressive (batch 1-2, grad accum 16):  ~100M params (Scale C)
Impossible without model parallelism:   200M params (Scale D)

KEY BOTTLENECK: FFT intermediates scale with field_size * head_dim * num_heads.
The wave field architecture has ~2-3x higher activation memory than a standard
transformer of equivalent size due to complex64 FFT buffers.

RECOMMENDATIONS:
  1. Scale A (25M) is the sweet spot for this GPU -- comfortable batch sizes,
     meaningful capacity, trainable on WikiText-103.
  2. Scale B (50M) is achievable with gradient accumulation.
  3. Scale C (100M) requires batch=1-2 + grad accum -- very slow but possible.
     Consider reducing field_size to 1024 (halves FFT memory).
  4. For C/D, reducing seq_len from 512 to 256 roughly halves activation memory.
  5. Scale D (200M) does not fit on this GPU under any batch configuration.

DATASET HIERARCHY:
  - WikiText-2 (2.6M tokens):   Only useful for <=8.5M params (current scale)
  - WikiText-103 (103M tokens):  Good for up to 50M params, marginal for 100M
  - OpenWebText (~8B tokens):    Required for 100M+ params (Chinchilla-optimal)
  - Use HuggingFace `datasets` library: datasets.load_dataset('wikitext', 'wikitext-103-raw-v1')
""")
