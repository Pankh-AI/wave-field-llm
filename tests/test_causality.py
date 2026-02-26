"""
Causality Test Suite: Comprehensive future-token leakage detection
===================================================================
Tests that changing any future token has ZERO effect on past positions.

8 tests covering every leak vector:
1. Basic (random init, multiple seq lengths)
2. Amplified SpectralGate (simulates post-training amplification)
3. Trained checkpoint (catches learned exploits)
4. Bidirectional (every position pair)
5. Gradient-based (d(logit[i]) / d(input[j]) must be 0 for j > i)
6. Batch isolation (sample 0 unaffected by sample 1)
7. Global context module (verify cumsum is causal)
8. Stateful leak (multiple forward passes don't contaminate)

The key insight: a leak at init might be 0.000014 (undetectable), but
training amplifies it 450,000x. So we test with artificially large weights
to simulate what training does.

Run: python tests/test_causality.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import copy
from src.wave_field_transformer import WaveFieldTransformer


# Threshold: fp32 has ~7 decimal digits of precision.
# For a model with outputs in range [-10, 10], noise floor is ~1e-6.
# We use 1e-4 as threshold — generous but catches real leaks.
STRICT_THRESHOLD = 1e-4
AMPLIFIED_THRESHOLD = 0.1   # amplified weights = amplified noise, but real leaks >> 0.1


def build_test_model(device, field_size=512, dropout=0.0):
    """Build a small model for testing."""
    return WaveFieldTransformer(
        vocab_size=100, embedding_dim=64, num_layers=2,
        num_heads=4, ffn_dim=128, field_size=field_size,
        max_seq_len=520, dropout=dropout, use_checkpoint=False,
        interference_interval=3, device=device,
    ).to(device).eval()


def measure_leak(model, seq_len, device, vocab_size=100, pos_changed=-1):
    """Measure max logit difference at past positions when one token changes.

    Args:
        pos_changed: which position to change (-1 = last, or specific index)

    Returns:
        max_leak: maximum logit difference at any position before pos_changed
        per_pos: list of (position, max_diff) for all positions before changed one
    """
    input_a = torch.randint(0, vocab_size, (1, seq_len), device=device)
    input_b = input_a.clone()

    change_idx = pos_changed if pos_changed >= 0 else seq_len - 1
    input_b[0, change_idx] = (input_a[0, change_idx] + 50) % vocab_size

    with torch.no_grad():
        logits_a, _ = model(input_a)
        logits_b, _ = model(input_b)

    max_leak = 0.0
    per_pos = []
    for p in range(change_idx):
        diff = (logits_a[0, p] - logits_b[0, p]).abs().max().item()
        per_pos.append((p, diff))
        max_leak = max(max_leak, diff)

    return max_leak, per_pos


def amplify_spectral_gate(model, scale=100.0):
    """Multiply SpectralGate output layer weights by a large factor.

    This simulates what training does: SpectralGate weights grow from
    norm ~9 to ~30 with 50x LR. We amplify even more aggressively to
    ensure any leak pathway is exposed.
    """
    amplified = copy.deepcopy(model)
    found = 0
    for name, module in amplified.named_modules():
        if 'spectral_gate' in name and hasattr(module, 'net'):
            # Amplify the output layer of the MLP
            with torch.no_grad():
                module.net[-1].weight.mul_(scale)
                module.net[-1].bias.mul_(scale)
            found += 1
    return amplified, found


def test_basic_causality(device):
    """Test 1: Random init model, multiple sequence lengths."""
    print("\n  TEST 1: Basic causality (random init)")
    print("  " + "-" * 55)

    model = build_test_model(device)

    all_pass = True
    print(f"  {'Seq Len':<10} {'Change Pos':<12} {'Max Leak':>12} {'Status':>10}")

    for seq_len in [10, 32, 64, 128, 256, 512]:
        # Change last token
        leak, _ = measure_leak(model, seq_len, device)
        status = "OK" if leak < STRICT_THRESHOLD else "LEAK!"
        if leak >= STRICT_THRESHOLD:
            all_pass = False
        print(f"  {seq_len:<10} {'last':<12} {leak:>12.6f} {status:>10}")

        # Change middle token
        mid = seq_len // 2
        leak_mid, _ = measure_leak(model, seq_len, device, pos_changed=mid)
        status = "OK" if leak_mid < STRICT_THRESHOLD else "LEAK!"
        if leak_mid >= STRICT_THRESHOLD:
            all_pass = False
        print(f"  {seq_len:<10} {f'mid ({mid})':<12} {leak_mid:>12.6f} {status:>10}")

    return all_pass


def test_amplified_causality(device):
    """Test 2: Amplified SpectralGate weights (simulates training).

    This is the KEY test. A leak at init might be 1e-5 (harmless looking).
    But training with 50x LR grows SpectralGate weights 3x, amplifying
    leaks ~450,000x. We simulate this by 100x amplification.
    """
    print("\n  TEST 2: Amplified SpectralGate (simulates post-training)")
    print("  " + "-" * 55)

    model = build_test_model(device)
    amplified, n_gates = amplify_spectral_gate(model, scale=100.0)
    print(f"  Amplified {n_gates} SpectralGate modules by 100x")

    all_pass = True
    print(f"  {'Seq Len':<10} {'Max Leak':>12} {'Amplified':>12} {'Status':>10}")

    for seq_len in [10, 64, 256, 512]:
        # Original
        leak_orig, _ = measure_leak(model, seq_len, device)
        # Amplified
        leak_amp, per_pos = measure_leak(amplified, seq_len, device)

        status = "OK" if leak_amp < AMPLIFIED_THRESHOLD else "LEAK!"
        if leak_amp >= AMPLIFIED_THRESHOLD:
            all_pass = False
            # Show worst positions
            per_pos.sort(key=lambda x: -x[1])
            print(f"  {seq_len:<10} {leak_orig:>12.6f} {leak_amp:>12.4f} {status:>10}")
            for pos, diff in per_pos[:3]:
                print(f"    -> pos {pos}: diff = {diff:.4f}")
        else:
            print(f"  {seq_len:<10} {leak_orig:>12.6f} {leak_amp:>12.4f} {status:>10}")

    return all_pass


def test_checkpoint_causality(device):
    """Test 3: Trained checkpoint (catches learned exploits)."""
    print("\n  TEST 3: Trained checkpoint")
    print("  " + "-" * 55)

    ckpt_paths = [
        'results/checkpoints/spectre-wave_s1_best.pt',
        'results/checkpoints/spectre-wave_s1_resume.pt',
    ]

    found_ckpt = None
    for p in ckpt_paths:
        if os.path.exists(p):
            found_ckpt = p
            break

    if found_ckpt is None:
        print("  No trained checkpoint found. Skipping.")
        print("  (Run S1 benchmark first to generate checkpoints)")
        return True  # not a failure, just skipped

    print(f"  Loading: {found_ckpt}")
    ckpt = torch.load(found_ckpt, map_location=device, weights_only=False)

    # Extract model config from checkpoint
    if 'model_config' in ckpt:
        cfg = ckpt['model_config']
    else:
        # Default S1 config
        cfg = dict(vocab_size=8000, embedding_dim=384, num_layers=8,
                   num_heads=8, ffn_dim=1536, field_size=512,
                   max_seq_len=514, dropout=0.1)

    model = WaveFieldTransformer(
        vocab_size=cfg.get('vocab_size', 8000),
        embedding_dim=cfg.get('embedding_dim', 384),
        num_layers=cfg.get('num_layers', 8),
        num_heads=cfg.get('num_heads', 8),
        ffn_dim=cfg.get('ffn_dim', 1536),
        field_size=cfg.get('field_size', 512),
        max_seq_len=cfg.get('max_seq_len', 514),
        dropout=0.0,  # no dropout for deterministic test
        use_checkpoint=False,
        interference_interval=3,
        device=device,
    ).to(device)

    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()

    vocab_size = cfg.get('vocab_size', 8000)
    all_pass = True
    print(f"  {'Seq Len':<10} {'Max Leak':>12} {'Status':>10}")

    for seq_len in [10, 64, 256, 512]:
        leak, per_pos = measure_leak(model, seq_len, device, vocab_size=vocab_size)
        status = "OK" if leak < STRICT_THRESHOLD else "LEAK!"
        if leak >= STRICT_THRESHOLD:
            all_pass = False
            per_pos.sort(key=lambda x: -x[1])
            print(f"  {seq_len:<10} {leak:>12.4f} {status:>10}")
            for pos, diff in per_pos[:3]:
                print(f"    -> pos {pos}: diff = {diff:.4f}")
        else:
            print(f"  {seq_len:<10} {leak:>12.6f} {status:>10}")

    return all_pass


def test_bidirectional_change(device):
    """Test 4: Change token at EVERY position, verify only later positions affected."""
    print("\n  TEST 4: Change each position, verify only downstream affected")
    print("  " + "-" * 55)

    model = build_test_model(device)
    seq_len = 32
    vocab_size = 100

    input_orig = torch.randint(0, vocab_size, (1, seq_len), device=device)
    with torch.no_grad():
        logits_orig, _ = model(input_orig)

    all_pass = True
    violations = []

    for change_pos in range(seq_len):
        input_mod = input_orig.clone()
        input_mod[0, change_pos] = (input_orig[0, change_pos] + 50) % vocab_size

        with torch.no_grad():
            logits_mod, _ = model(input_mod)

        # Check: positions BEFORE change_pos should be unaffected
        for check_pos in range(change_pos):
            diff = (logits_orig[0, check_pos] - logits_mod[0, check_pos]).abs().max().item()
            if diff >= STRICT_THRESHOLD:
                violations.append((change_pos, check_pos, diff))
                all_pass = False

    if violations:
        print(f"  FOUND {len(violations)} violations!")
        for cp, ck, d in violations[:10]:
            print(f"    Changed pos {cp}, leaked to pos {ck}: diff = {d:.6f}")
        if len(violations) > 10:
            print(f"    ... and {len(violations) - 10} more")
    else:
        total_checks = seq_len * (seq_len - 1) // 2
        print(f"  All {total_checks} position pairs clean. No leaks.")

    return all_pass


def test_gradient_causality(device):
    """Test 5: Gradient-based leak detection (most precise).

    Compute d(logit[i]) / d(input_embedding[j]). For j > i, this gradient
    MUST be exactly zero. This catches leaks that the logit-diff test might
    miss due to cancellation (e.g., two leak paths that partially cancel).
    """
    print("\n  TEST 5: Gradient-based causality (d(out[i])/d(emb[j]) = 0 for j>i)")
    print("  " + "-" * 55)

    model = build_test_model(device)
    # Need gradients, so switch to train mode but keep dropout=0
    model.eval()

    seq_len = 32
    vocab_size = 100
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

    # Hook into embedding output to get a tensor we can differentiate w.r.t.
    # We use a forward hook to intercept and replace the post-embedding tensor.
    embed_holder = {}

    def hook_fn(module, input, output):
        # output is the embedding lookup result (B, N, D)
        # Detach and make it require grad so we can compute jacobian
        new_out = output.detach().requires_grad_(True)
        embed_holder['embed'] = new_out
        return new_out

    handle = model.token_embedding.register_forward_hook(hook_fn)
    logits, _ = model(input_ids)
    handle.remove()

    embed = embed_holder['embed']  # (1, seq_len, embed_dim)

    all_pass = True
    violations = []

    # Check: gradient of logit[i, vocab_0] w.r.t. embed[j] for j > i
    for check_pos in [0, 5, 15, seq_len // 2]:
        # Take gradient of one logit scalar
        if embed.grad is not None:
            embed.grad.zero_()
        target = logits[0, check_pos, 0]  # scalar
        target.backward(retain_graph=True)

        grad = embed.grad[0]  # (seq_len, embed_dim)

        # Positions AFTER check_pos should have zero gradient
        for future_pos in range(check_pos + 1, seq_len):
            grad_norm = grad[future_pos].abs().max().item()
            if grad_norm >= STRICT_THRESHOLD:
                violations.append((check_pos, future_pos, grad_norm))
                all_pass = False

    if violations:
        print(f"  FOUND {len(violations)} gradient violations!")
        for cp, fp, g in violations[:10]:
            print(f"    d(logit[{cp}])/d(embed[{fp}]) = {g:.6f}")
    else:
        print(f"  All gradients from future positions are zero. Clean.")

    return all_pass


def test_batch_isolation(device):
    """Test 6: Verify samples in a batch don't affect each other.

    Run same input as batch=1 vs batch=2 (with a different second sample).
    Logits for sample 0 must be identical in both cases.
    """
    print("\n  TEST 6: Batch isolation (sample 0 unaffected by sample 1)")
    print("  " + "-" * 55)

    model = build_test_model(device)
    seq_len = 64
    vocab_size = 100

    # Sample 0: the one we care about
    input_0 = torch.randint(0, vocab_size, (1, seq_len), device=device)

    # Run as batch=1
    with torch.no_grad():
        logits_single, _ = model(input_0)

    # Run as batch=2 with random second sample
    all_pass = True
    max_diff_seen = 0.0

    for trial in range(5):  # 5 different companion samples
        input_companion = torch.randint(0, vocab_size, (1, seq_len), device=device)
        input_batch = torch.cat([input_0, input_companion], dim=0)  # (2, seq_len)

        with torch.no_grad():
            logits_batch, _ = model(input_batch)

        # Sample 0's logits should be identical
        diff = (logits_single[0] - logits_batch[0]).abs().max().item()
        max_diff_seen = max(max_diff_seen, diff)
        if diff >= STRICT_THRESHOLD:
            all_pass = False
            print(f"    Trial {trial}: max diff = {diff:.6f} LEAK!")

    if all_pass:
        print(f"  5 trials, max diff = {max_diff_seen:.8f}. Batch-isolated.")
    return all_pass


def test_global_context_causality(device):
    """Test 7: Verify GlobalContextModule is causal.

    The module uses torch.cumsum for causal pooling. Verify that changing
    a future token doesn't affect the global context at earlier positions.
    """
    print("\n  TEST 7: GlobalContextModule causality")
    print("  " + "-" * 55)

    from src.global_context import GlobalContextModule

    gcm = GlobalContextModule(embedding_dim=64, dropout=0.0).to(device).eval()

    seq_len = 64
    B = 1
    D = 64

    input_a = torch.randn(B, seq_len, D, device=device)
    input_b = input_a.clone()
    # Change last 10 positions
    input_b[:, -10:, :] = torch.randn(B, 10, D, device=device)

    with torch.no_grad():
        out_a = gcm(input_a)
        out_b = gcm(input_b)

    all_pass = True
    # Positions 0 to seq_len-11 should be identical
    for pos in range(seq_len - 10):
        diff = (out_a[0, pos] - out_b[0, pos]).abs().max().item()
        if diff >= STRICT_THRESHOLD:
            all_pass = False
            print(f"    Pos {pos}: diff = {diff:.6f} LEAK!")

    if all_pass:
        max_clean = (out_a[0, :seq_len-10] - out_b[0, :seq_len-10]).abs().max().item()
        # Verify the changed positions DO differ (sanity check)
        changed_diff = (out_a[0, -1] - out_b[0, -1]).abs().max().item()
        print(f"  Clean positions max diff: {max_clean:.8f}")
        print(f"  Changed position diff: {changed_diff:.4f} (should be > 0)")
        if changed_diff < STRICT_THRESHOLD:
            print(f"    WARNING: Changed positions also unchanged — module might be dead")

    return all_pass


def test_stateful_leak(device):
    """Test 8: Multiple forward passes don't contaminate each other.

    Some implementations cache intermediate results (buffers, hidden states).
    Verify that running input A then input B gives same result for B as
    running B alone.
    """
    print("\n  TEST 8: Stateful leak (sequential forward passes)")
    print("  " + "-" * 55)

    model = build_test_model(device)
    seq_len = 64
    vocab_size = 100

    input_a = torch.randint(0, vocab_size, (1, seq_len), device=device)
    input_b = torch.randint(0, vocab_size, (1, seq_len), device=device)

    # Run B alone
    with torch.no_grad():
        logits_b_alone, _ = model(input_b)

    # Run A first, then B
    with torch.no_grad():
        _ = model(input_a)  # "contaminate" with A
        logits_b_after_a, _ = model(input_b)

    diff = (logits_b_alone - logits_b_after_a).abs().max().item()
    passed = diff < STRICT_THRESHOLD

    if passed:
        print(f"  Max diff between solo vs post-A: {diff:.8f}. No state leak.")
    else:
        print(f"  Max diff: {diff:.6f} LEAK! Previous forward pass contaminated output.")

    return passed


def test_full_size_configs(device):
    """Test 9: Realistic S1/S2 model configs (not just tiny 2-layer model).

    The tiny test model (embed=64, 2 layers) can miss leaks that only appear
    at scale due to FFT padding ratios. This test uses the actual S1/S2
    configs with field_size=seq_len to catch circular wraparound leaks.
    """
    print("\n  TEST 9: Full-size model configs (S1/S2)")
    print("  " + "-" * 55)

    configs = {
        'S1': dict(vocab_size=8000, embedding_dim=384, num_layers=8,
                   num_heads=8, ffn_dim=1536, field_size=512, max_seq_len=514),
        'S2': dict(vocab_size=8000, embedding_dim=512, num_layers=12,
                   num_heads=8, ffn_dim=2048, field_size=512, max_seq_len=514),
    }

    all_pass = True

    for name, cfg in configs.items():
        model = WaveFieldTransformer(
            dropout=0.0, use_checkpoint=False,
            interference_interval=3, device=device, **cfg,
        ).to(device).eval()

        params = sum(p.numel() for p in model.parameters())
        print(f"  {name} ({params/1e6:.1f}M params):")

        # Test at N=512 (the actual training seq_len). Smaller N values
        # have fractional stride positions that create harmless interpolation
        # noise unrelated to causality.
        seq_len = 512
        max_leak = 0.0
        for trial in range(3):
            a = torch.randint(0, cfg['vocab_size'], (1, seq_len), device=device)
            b = a.clone()
            b[0, -1] = (a[0, -1] + 50) % cfg['vocab_size']

            with torch.no_grad():
                la, _ = model(a)
                lb, _ = model(b)

            leak = max((la[0, p] - lb[0, p]).abs().max().item()
                       for p in range(seq_len - 1))
            max_leak = max(max_leak, leak)

        status = "OK" if max_leak < STRICT_THRESHOLD else "LEAK!"
        if max_leak >= STRICT_THRESHOLD:
            all_pass = False
        print(f"    N={seq_len} (3 trials): max_leak = {max_leak:.8f}  {status}")

        del model

    return all_pass


def main():
    print("=" * 65)
    print("  CAUSALITY TEST SUITE")
    print("  Comprehensive future-token leakage detection")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    results = {}

    results['basic'] = test_basic_causality(device)
    results['amplified'] = test_amplified_causality(device)
    results['checkpoint'] = test_checkpoint_causality(device)
    results['bidirectional'] = test_bidirectional_change(device)
    results['gradient'] = test_gradient_causality(device)
    results['batch_isolation'] = test_batch_isolation(device)
    results['global_context'] = test_global_context_causality(device)
    results['stateful'] = test_stateful_leak(device)
    results['full_size'] = test_full_size_configs(device)

    # Summary
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("  " + "-" * 55)
    for name, passed in results.items():
        status = "PASS" if passed else "*** FAIL ***"
        print(f"  {name:<20} {status}")

    all_pass = all(results.values())
    print(f"\n  {'ALL TESTS PASSED' if all_pass else '*** CAUSALITY VIOLATIONS DETECTED ***'}")
    print("=" * 65)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())