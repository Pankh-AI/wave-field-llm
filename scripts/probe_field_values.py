"""
Probe: What are actual field values during forward pass?
This tells us if the Kerr nonlinearity can even "see" anything.
"""
import torch
import torch.nn.functional as F
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.wave_field_transformer import WaveFieldTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

# S1 config
model = WaveFieldTransformer(
    vocab_size=8000, embedding_dim=384, num_layers=8,
    num_heads=8, ffn_dim=1536, field_size=512,
    max_seq_len=514, dropout=0.0, use_checkpoint=False,
    interference_interval=3, n_components=1,
    local_window=0, n_frozen_heads=4,
    use_split_step=True, device=device,
).to(device)

# Hook into the attention layers to measure field values
field_stats = {}

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        # We need to intercept the field INSIDE forward
        # Instead, let's monkey-patch _wave_convolve to also log
        pass
    return hook_fn

# Better approach: just run forward and add debug prints
# Monkey-patch the first attention layer
attn = model.layers[0].attention

original_forward = attn.forward

def debug_forward(x, mask=None):
    """Intercept to measure field values."""
    B, N, D = x.shape
    H = attn.num_heads
    head_dim = D // H
    G = attn.field_size

    # Run QKV projection
    qkvg = attn.qkvg_proj(x)
    qkvg = qkvg.view(B, N, H, 4 * head_dim).transpose(1, 2)
    q, k, v, g = qkvg.chunk(4, dim=-1)

    # Feature maps
    q_feat = attn.q_feature_map(q)
    k_feat = attn.k_feature_map(k)

    # Deposit
    deposit = k_feat * v

    # Indices
    idx_lo = attn._cached_idx_lo[:N]
    idx_hi = attn._cached_idx_hi[:N]
    frac = attn._cached_frac[:N]

    # Scatter
    field = attn._bilinear_scatter(deposit, idx_lo, idx_hi, frac, B, H, G, head_dim, x.device)

    print(f"\n  Layer 0 field stats:")
    print(f"    deposit:    mean={deposit.abs().mean():.6f}  max={deposit.abs().max():.6f}  std={deposit.std():.6f}")
    print(f"    pre-convolve field: mean={field.abs().mean():.6f}  max={field.abs().max():.6f}  std={field.std():.6f}")
    print(f"    field nonzero frac: {(field.abs() > 1e-8).float().mean():.4f}")

    # Convolve
    base_kernel_fft = attn._build_analytic_kernel_fft(x.device)
    kernel_fft = attn.spectral_gate(q, base_kernel_fft)
    field_conv = attn._wave_convolve(field, kernel_fft)

    print(f"    post-convolve field: mean={field_conv.abs().mean():.6f}  max={field_conv.abs().max():.6f}  std={field_conv.std():.6f}")

    # Kerr effect
    field_abs = field_conv.abs()
    kerr_term = attn.kerr_gamma.view(1, -1, 1, 1) * field_abs / (1.0 + field_abs)
    print(f"    kerr_gamma: {attn.kerr_gamma.data}")
    print(f"    |field|/(1+|field|): mean={field_abs.mean()/(1+field_abs.mean()):.6f}  max={field_abs.max()/(1+field_abs.max()):.6f}")
    print(f"    kerr_term (gamma * saturable): mean={kerr_term.abs().mean():.6f}  max={kerr_term.abs().max():.6f}")
    print(f"    kerr relative effect: {kerr_term.abs().mean() / (1.0 + kerr_term.abs().mean()) * 100:.4f}%")

    # Now run the real forward
    return original_forward(x, mask)

attn.forward = debug_forward

# Run with random input
print("=" * 60)
print("PROBE: Field values at initialization")
print("=" * 60)

x = torch.randint(0, 8000, (2, 512)).to(device)
with torch.no_grad():
    logits, _ = model(x)

print(f"\n  Output logits: mean={logits.abs().mean():.4f}  std={logits.std():.4f}")

# Now simulate what happens after some training steps
print("\n" + "=" * 60)
print("PROBE: After 10 training steps")
print("=" * 60)

import numpy as np
cache_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'cache')
train_path = os.path.join(cache_dir, 'wt2_train.npy')
if os.path.exists(train_path):
    train_ids = np.load(train_path)

    optimizer = model.configure_optimizer(base_lr=3e-4, weight_decay=0.01)
    model.train()

    for step in range(10):
        start = step * 513
        chunk = torch.tensor(train_ids[start:start+513], dtype=torch.long).to(device)
        inp = chunk[:512].unsqueeze(0)
        tgt = chunk[1:513].unsqueeze(0)

        optimizer.zero_grad()
        logits, _ = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, 8000), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 9:
            print(f"\n  Step {step}: loss={loss.item():.3f}")
else:
    print("  No cached data, skipping training probe")
