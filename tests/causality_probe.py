"""
Causality Probe: Test TRAINED model for future token leakage.
==============================================================
If changing future tokens affects past logits → model is cheating.

Tests both random-init AND trained checkpoint.
"""

import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.wave_field_transformer import WaveFieldTransformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# S1 config (must match training)
CFG = {
    'embedding_dim': 384, 'num_layers': 8, 'num_heads': 8,
    'ffn_dim': 1536, 'field_size': 2048, 'seq_len': 512,
}
VOCAB_SIZE = 8000  # BPE vocab used in training


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


@torch.no_grad()
def causality_probe(model, seq_len=256, split_pos=128, n_trials=5):
    """
    For each trial:
      1. Generate random input x
      2. Clone x2, scramble everything AFTER split_pos
      3. Compare logits at positions BEFORE split_pos

    Causal model: diff ~ 0 (floating point noise only)
    Leaky model: diff >> 0
    """
    model.eval()
    diffs = []
    max_diffs = []

    for trial in range(n_trials):
        x = torch.randint(0, VOCAB_SIZE, (1, seq_len), device=DEVICE)
        x2 = x.clone()
        x2[:, split_pos:] = torch.randint(0, VOCAB_SIZE, (1, seq_len - split_pos), device=DEVICE)

        logits1, _ = model(x)
        logits2, _ = model(x2)

        prefix_diff = (logits1[:, :split_pos, :] - logits2[:, :split_pos, :]).abs()
        mean_d = prefix_diff.mean().item()
        max_d = prefix_diff.max().item()
        diffs.append(mean_d)
        max_diffs.append(max_d)

    return sum(diffs) / len(diffs), max(max_diffs)


def run_probe(label, model):
    print(f"\n  {label}")
    print(f"  {'-' * 55}")

    results = []
    for seq_len in [64, 128, 256, 512]:
        split = seq_len // 2
        mean_d, max_d = causality_probe(model, seq_len=seq_len, split_pos=split, n_trials=5)
        verdict = "OK (causal)" if max_d < 1e-4 else "LEAK DETECTED"
        print(f"    seq={seq_len:>3}, split={split:>3} | mean_diff={mean_d:.6f} | max_diff={max_d:.6f} | {verdict}")
        results.append((seq_len, mean_d, max_d))

    # Also test: change ONLY the very next token (most sensitive test)
    print(f"\n    Single-token probe (change only position split_pos):")
    for seq_len in [128, 256, 512]:
        split = seq_len // 2
        model.eval()
        diffs_single = []
        for _ in range(5):
            x = torch.randint(0, VOCAB_SIZE, (1, seq_len), device=DEVICE)
            x2 = x.clone()
            x2[0, split] = (x[0, split] + 1) % VOCAB_SIZE  # change just ONE future token

            logits1, _ = model(x)
            logits2, _ = model(x2)

            d = (logits1[0, :split, :] - logits2[0, :split, :]).abs().max().item()
            diffs_single.append(d)

        max_d = max(diffs_single)
        verdict = "OK" if max_d < 1e-4 else "LEAK"
        print(f"    seq={seq_len:>3}, changed pos {split} | max_diff={max_d:.6f} | {verdict}")

    return results


def main():
    print("=" * 65)
    print("  CAUSALITY PROBE — Random Init vs Trained Checkpoint")
    print("=" * 65)

    # Test 1: Random init
    model_rand = build_model()
    run_probe("A) Random Init (untrained)", model_rand)
    del model_rand
    torch.cuda.empty_cache()

    # Test 2: Trained checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'spectre-wave_s1.pt')
    if os.path.exists(ckpt_path):
        model_trained = build_model()
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model_trained.load_state_dict(state)
        run_probe("B) Trained Checkpoint (spectre-wave_s1.pt)", model_trained)
        del model_trained
        torch.cuda.empty_cache()
    else:
        print(f"\n  Checkpoint not found: {ckpt_path}")
        print(f"  Skipping trained model test.")

    print(f"\n{'=' * 65}")
    print(f"  INTERPRETATION:")
    print(f"    max_diff < 1e-5  → Perfectly causal (FP noise only)")
    print(f"    max_diff 1e-5 to 1e-3 → Borderline (numerical, not semantic)")
    print(f"    max_diff > 0.01  → Future leakage confirmed")
    print(f"    max_diff > 0.1   → Severe leakage (model exploiting it)")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
