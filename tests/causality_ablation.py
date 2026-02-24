"""
Causality Ablation: Which component causes the leak?
=====================================================
Loads trained checkpoint, disables components one by one,
re-runs causality probe.

Suspects:
  1. Field coupling matrix (cross-head mixing at shared positions)
  2. Bilinear scatter overlap (adjacent tokens share field cells)
  3. Spectral gate (uses q from all positions)
  4. Wave convolution itself (FFT wraparound)
"""

import torch
import torch.nn.functional as F
import sys, os, math, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.wave_field_transformer import WaveFieldTransformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VOCAB_SIZE = 8000
CFG = {
    'embedding_dim': 384, 'num_layers': 8, 'num_heads': 8,
    'ffn_dim': 1536, 'field_size': 2048, 'seq_len': 512,
}


def build_model():
    return WaveFieldTransformer(
        vocab_size=VOCAB_SIZE,
        embedding_dim=CFG['embedding_dim'],
        num_layers=CFG['num_layers'],
        num_heads=CFG['num_heads'],
        ffn_dim=CFG['ffn_dim'],
        field_size=CFG['field_size'],
        max_seq_len=CFG['seq_len'] + 2,
        dropout=0.0,
        use_checkpoint=False,
        interference_interval=3,
        n_components=1,
        local_window=0,
        device=DEVICE,
    ).to(DEVICE)


def load_trained():
    model = build_model()
    ckpt = os.path.join(os.path.dirname(__file__), '..', 'results', 'spectre-wave_s1.pt')
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def quick_probe(model, seq_len=256, split_pos=128, n_trials=3):
    model.eval()
    max_diffs = []
    for _ in range(n_trials):
        x = torch.randint(0, VOCAB_SIZE, (1, seq_len), device=DEVICE)
        x2 = x.clone()
        x2[:, split_pos:] = torch.randint(0, VOCAB_SIZE, (1, seq_len - split_pos), device=DEVICE)
        logits1, _ = model(x)
        logits2, _ = model(x2)
        max_d = (logits1[:, :split_pos, :] - logits2[:, :split_pos, :]).abs().max().item()
        max_diffs.append(max_d)
    return max(max_diffs)


def test_baseline(model):
    """Full trained model - should show leak."""
    d = quick_probe(model)
    return d


def test_no_coupling(model):
    """Replace field coupling with identity - isolates coupling leak."""
    H = CFG['num_heads']
    for layer in model.layers:
        attn = layer.attention
        attn.field_coupling.data = torch.eye(H, device=DEVICE)
    d = quick_probe(model)
    return d


def test_no_spectral_gate(model):
    """Bypass spectral gate - use base kernel directly."""
    # Monkey-patch spectral gate to return base kernel unchanged
    for layer in model.layers:
        attn = layer.attention
        original_forward = attn.spectral_gate.forward
        def bypass_gate(q, base_kernel_fft, _orig=original_forward):
            B = q.shape[0]
            return base_kernel_fft.unsqueeze(0).expand(B, -1, -1)
        attn.spectral_gate.forward = bypass_gate
    d = quick_probe(model)
    return d


def test_no_feature_maps(model):
    """Replace learned feature maps with identity (passthrough)."""
    for layer in model.layers:
        attn = layer.attention
        # Reset feature map weights to identity
        with torch.no_grad():
            attn.q_feature_map.net[0].weight.copy_(torch.eye(attn.head_dim, device=DEVICE))
            attn.q_feature_map.net[0].bias.zero_()
            attn.k_feature_map.net[0].weight.copy_(torch.eye(attn.head_dim, device=DEVICE))
            attn.k_feature_map.net[0].bias.zero_()
    d = quick_probe(model)
    return d


def test_no_wave_convolve(model):
    """Skip wave convolution entirely - just scatter and gather."""
    for layer in model.layers:
        attn = layer.attention
        attn._wave_convolve = lambda field, kfft: field
    d = quick_probe(model)
    return d


def test_increased_stride(model):
    """Double the field stride so tokens are further apart (less scatter overlap)."""
    for layer in model.layers:
        attn = layer.attention
        attn.field_stride = attn.field_stride * 2
    d = quick_probe(model, seq_len=128, split_pos=64)  # shorter seq to fit
    return d


def main():
    print("=" * 65)
    print("  CAUSALITY ABLATION - Which component leaks?")
    print("=" * 65)

    results = []

    # 1. Baseline (full model)
    print("\n  [1/6] Baseline (full trained model)...", flush=True)
    model = load_trained()
    d = test_baseline(model)
    results.append(("Baseline (full model)", d))
    print(f"         max_diff = {d:.6f}")
    del model; torch.cuda.empty_cache()

    # 2. No field coupling
    print("\n  [2/6] Disable field coupling (identity matrix)...", flush=True)
    model = load_trained()
    d = test_no_coupling(model)
    results.append(("No field coupling", d))
    print(f"         max_diff = {d:.6f}")
    del model; torch.cuda.empty_cache()

    # 3. No spectral gate
    print("\n  [3/6] Disable spectral gate (use base kernel)...", flush=True)
    model = load_trained()
    d = test_no_spectral_gate(model)
    results.append(("No spectral gate", d))
    print(f"         max_diff = {d:.6f}")
    del model; torch.cuda.empty_cache()

    # 4. No learned feature maps
    print("\n  [4/6] Reset feature maps to identity...", flush=True)
    model = load_trained()
    d = test_no_feature_maps(model)
    results.append(("No learned feature maps", d))
    print(f"         max_diff = {d:.6f}")
    del model; torch.cuda.empty_cache()

    # 5. No wave convolution
    print("\n  [5/6] Skip wave convolution entirely...", flush=True)
    model = load_trained()
    d = test_no_wave_convolve(model)
    results.append(("No wave convolution", d))
    print(f"         max_diff = {d:.6f}")
    del model; torch.cuda.empty_cache()

    # 6. Increased stride
    print("\n  [6/6] Double field stride (reduce scatter overlap)...", flush=True)
    model = load_trained()
    d = test_increased_stride(model)
    results.append(("2x field stride (seq=128)", d))
    print(f"         max_diff = {d:.6f}")
    del model; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 65}")
    print(f"  ABLATION SUMMARY")
    print(f"  {'Component disabled':<35} {'max_diff':>12} {'Verdict':>12}")
    print(f"  {'-'*35} {'-'*12} {'-'*12}")
    for name, d in results:
        if d < 1e-4:
            verdict = "FIXED"
        elif d < 0.01:
            verdict = "reduced"
        elif d < 1.0:
            verdict = "partial"
        else:
            verdict = "STILL LEAKS"
        print(f"  {name:<35} {d:>12.6f} {verdict:>12}")

    print(f"\n  INTERPRETATION:")
    print(f"    If disabling X makes diff -> 0: X is the leak source")
    print(f"    If disabling X barely changes: X is not the problem")
    print(f"    If multiple reduce partially: multiple leak paths")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
