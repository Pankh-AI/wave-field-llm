"""
Causality Test: Does changing a FUTURE token affect PAST token outputs?
========================================================================
If yes → the model can see the future → explains 99% accuracy + garbage generation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from src.wave_field_transformer import WaveFieldTransformer


def test_causality():
    print("=" * 65)
    print("  CAUSALITY TEST")
    print("  Can the model see future tokens?")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build a small model
    field_size = 1024
    vocab_size = 100
    model = WaveFieldTransformer(
        vocab_size=vocab_size, embedding_dim=64, num_layers=2,
        num_heads=4, ffn_dim=128, field_size=field_size,
        max_seq_len=33, dropout=0.0, use_checkpoint=False,
        interference_interval=3, device=device,
    ).to(device)
    model.eval()

    seq_len = 10

    # Test 1: Change a FUTURE token, check if PAST logits change
    print(f"\n  Test 1: Change future token, check past logits")
    print(f"  Sequence length: {seq_len}, Field size: {field_size}")

    # Original input
    input_a = torch.randint(0, vocab_size, (1, seq_len), device=device)

    # Modified input: change ONLY the last token
    input_b = input_a.clone()
    input_b[0, -1] = (input_a[0, -1] + 50) % vocab_size

    print(f"\n  Input A: {input_a[0].tolist()}")
    print(f"  Input B: {input_b[0].tolist()} (only last token changed)")

    with torch.no_grad():
        logits_a, _ = model(input_a)
        logits_b, _ = model(input_b)

    # Check: do logits at positions 0 to seq_len-2 change?
    print(f"\n  Position-by-position logit difference:")
    print(f"  {'Position':<10} {'Max Diff':>12} {'Causal?':>10}")
    print(f"  {'-'*10} {'-'*12} {'-'*10}")

    violation_found = False
    for pos in range(seq_len):
        diff = (logits_a[0, pos] - logits_b[0, pos]).abs().max().item()
        if pos < seq_len - 1:
            causal = "YES" if diff < 1e-5 else "NO — LEAK!"
            if diff >= 1e-5:
                violation_found = True
        else:
            causal = "(changed)" if diff > 0 else "unchanged"
        print(f"  {pos:<10} {diff:>12.6f} {causal:>10}")

    if violation_found:
        print(f"\n  *** CAUSALITY VIOLATION DETECTED ***")
        print(f"  Changing the LAST token affected EARLIER positions.")
        print(f"  The model can see the future through FFT circular wraparound!")
    else:
        print(f"\n  CAUSALITY OK — past positions unaffected by future changes.")

    # Test 2: Quantify the leakage per position
    print(f"\n  Test 2: Leakage strength by position")
    max_diffs = []
    for pos in range(seq_len - 1):
        diff = (logits_a[0, pos] - logits_b[0, pos]).abs().max().item()
        max_diffs.append(diff)
        bar_len = min(int(diff * 100), 50)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"  Pos {pos}: {bar} {diff:.6f}")

    # Test 3: Try with different sequence lengths
    print(f"\n  Test 3: Leakage vs sequence length")
    print(f"  {'Seq Len':<10} {'Max Leakage':>15} {'Causal?':>10}")
    for test_len in [5, 10, 20, 50, 100, 128]:
        ia = torch.randint(0, vocab_size, (1, test_len), device=device)
        ib = ia.clone()
        ib[0, -1] = (ia[0, -1] + 50) % vocab_size

        with torch.no_grad():
            la, _ = model(ia)
            lb, _ = model(ib)

        max_leak = 0
        for p in range(test_len - 1):
            diff = (la[0, p] - lb[0, p]).abs().max().item()
            max_leak = max(max_leak, diff)

        causal = "OK" if max_leak < 1e-5 else f"LEAK ({max_leak:.4f})"
        print(f"  {test_len:<10} {max_leak:>15.6f} {causal:>10}")

    # Test 4: What happens with contiguous scatter (the fix)?
    print(f"\n  Test 4: Diagnosis — Token field positions")
    for test_len in [5, 10, 50, 128]:
        seq_pos = torch.arange(test_len, dtype=torch.float32)
        field_pos = (seq_pos / max(test_len - 1, 1)) * (field_size - 1)
        field_idx = field_pos.long().clamp(0, field_size - 1)
        print(f"  N={test_len:3d}: tokens at field positions {field_idx[0].item()}, "
              f"{field_idx[1].item()}, ..., {field_idx[-2].item()}, {field_idx[-1].item()} "
              f"(spread across 0-{field_size-1})")

    print(f"\n  CONCLUSION:")
    if violation_found:
        print(f"  The scatter mapping spreads tokens across the ENTIRE field (0 to {field_size-1}).")
        print(f"  FFT convolution is CIRCULAR — position 0 wraps to position {field_size-1}.")
        print(f"  This means token 0 can see token N-1 through circular wraparound.")
        print(f"  FIX: Place tokens contiguously at positions 0 to N-1 (not spread).")
        print(f"  With N << G/2, the wraparound reaches only empty positions.")
    else:
        print(f"  No causality violation detected. Architecture is correctly causal.")

    print(f"\n{'='*65}")


if __name__ == "__main__":
    test_causality()
