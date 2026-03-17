"""
Diagnostic: WHERE does the 1.4x PPL gap between Standard and Wave live?

Compares per-token loss to find specific failure patterns:
1. Per-position gap (early vs late in sequence)
2. Per-token-type gap (punctuation, common, rare)
3. Copy/repetition detection (can Wave retrieve repeated tokens?)
4. Induction head test (A B ... A -> B?)
5. Local vs long-range dependency analysis

Usage:
    python diagnostics/diagnose_gap.py
"""

import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.wave_field_transformer import WaveFieldTransformer

# Import StandardTransformer from the benchmark (exact same class used to train the checkpoint)
# benchmarks/ has no __init__.py, so use importlib
import importlib.util
_bench_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'benchmarks', 'benchmark_scaling.py')
_spec = importlib.util.spec_from_file_location("benchmark_scaling", _bench_path)
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)
StandardTransformer = _bench.StandardTransformer

import torch.nn as nn


def load_models(device):
    """Load both S1 checkpoints."""
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

    # S1 config
    cfg = dict(embedding_dim=384, num_layers=8, num_heads=8, ffn_dim=1536,
               field_size=2048, seq_len=512)
    vocab_size = 8000

    # Wave model — try checkpoints/ first (V4.3.3), fall back to results/
    wave_path = os.path.join(results_dir, 'checkpoints', 'spectre-wave_s1.pt')
    if not os.path.exists(wave_path):
        wave_path = os.path.join(results_dir, 'spectre-wave_s1.pt')
        print(f"  WARNING: Using CTC checkpoint (not V4.3.3)")

    print(f"  Loading Wave: {wave_path}")
    wave_model = WaveFieldTransformer(
        vocab_size=vocab_size,
        embedding_dim=cfg['embedding_dim'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        ffn_dim=cfg['ffn_dim'],
        field_size=cfg['field_size'],
        max_seq_len=cfg['seq_len'] + 2,
        dropout=0.1,
        use_checkpoint=False,
        interference_interval=3,
        n_components=1,
        device=device,
    ).to(device)
    state = torch.load(wave_path, map_location=device, weights_only=True)
    info = wave_model.load_state_dict(state, strict=False)
    if info.missing_keys:
        print(f"  WARNING: Wave missing keys: {info.missing_keys}")
    if info.unexpected_keys:
        print(f"  WARNING: Wave unexpected keys: {info.unexpected_keys}")
    wave_model.eval()

    # Standard model (same class as benchmark — strict=True to catch mismatches)
    # Prefer checkpoints/ (V4.3.3 reference) over results/ (may be CTC run)
    std_path = os.path.join(results_dir, 'checkpoints', 'standard_s1.pt')
    if not os.path.exists(std_path):
        std_path = os.path.join(results_dir, 'standard_s1.pt')
    print(f"  Loading Standard: {std_path}")
    std_model = StandardTransformer(
        vocab_size=vocab_size,
        embedding_dim=cfg['embedding_dim'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        ffn_dim=cfg['ffn_dim'],
        max_seq_len=cfg['seq_len'] + 2,
        dropout=0.1,
    ).to(device)
    state = torch.load(std_path, map_location=device, weights_only=True)
    std_model.load_state_dict(state, strict=True)
    std_model.eval()

    return wave_model, std_model, vocab_size


def load_tokenizer_and_data():
    """Load cached tokenizer and val tokens."""
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    cache_dir = os.path.join(results_dir, 'cache')

    # Load tokenizer (optional — may not be available locally)
    tok = None
    tok_path = os.path.join(cache_dir, 'bpe_vocab8000.json')
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tok_path)
    except (ImportError, Exception) as e:
        print(f"  Tokenizer not available ({e}), using token IDs only")

    # Load val tokens
    val_path = os.path.join(cache_dir, 'wt2_val.npy')
    val_tokens = np.load(val_path)
    return tok, val_tokens


def per_token_loss(model, input_ids, device):
    """Compute per-position cross-entropy loss. Returns (N-1,) tensor."""
    with torch.no_grad():
        logits, _ = model(input_ids)
    # logits: (1, N, V), shift for next-token prediction
    logits = logits[0, :-1, :]  # (N-1, V)
    targets = input_ids[0, 1:]   # (N-1,)
    loss_per_token = F.cross_entropy(logits, targets, reduction='none')  # (N-1,)
    return loss_per_token


def diagnose_per_position(wave_losses_all, std_losses_all, seq_len):
    """Analyze gap by position in sequence."""
    print("\n" + "=" * 70)
    print("  1. PER-POSITION GAP ANALYSIS")
    print("     Where in the sequence does Wave fall behind?")
    print("=" * 70)

    # Average loss at each position across all chunks
    max_pos = min(seq_len - 1, wave_losses_all.shape[1])
    wave_by_pos = wave_losses_all[:, :max_pos].mean(dim=0).cpu().numpy()
    std_by_pos = std_losses_all[:, :max_pos].mean(dim=0).cpu().numpy()
    gap_by_pos = wave_by_pos - std_by_pos

    # Report in bins
    bins = [(0, 10, "First 10"), (10, 50, "Pos 10-50"), (50, 128, "Pos 50-128"),
            (128, 256, "Pos 128-256"), (256, 512, "Pos 256-512")]

    print(f"\n  {'Region':<15} {'Wave PPL':>10} {'Std PPL':>10} {'Gap':>8} {'Ratio':>8}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for start, end, label in bins:
        if start >= max_pos:
            break
        end = min(end, max_pos)
        w = wave_by_pos[start:end].mean()
        s = std_by_pos[start:end].mean()
        print(f"  {label:<15} {math.exp(w):>10.1f} {math.exp(s):>10.1f} {w-s:>8.3f} {math.exp(w)/math.exp(s):>8.2f}x")

    # Find worst positions
    worst_idx = np.argsort(gap_by_pos)[-10:][::-1]
    print(f"\n  Worst positions (biggest gap):")
    for idx in worst_idx:
        print(f"    Pos {idx:3d}: Wave={math.exp(wave_by_pos[idx]):.0f} Std={math.exp(std_by_pos[idx]):.0f} gap={gap_by_pos[idx]:.3f}")

    # Return structured data for programmatic use
    bins_data = []
    for start, end, label in bins:
        if start >= max_pos:
            break
        e = min(end, max_pos)
        w = float(wave_by_pos[start:e].mean())
        s = float(std_by_pos[start:e].mean())
        bins_data.append({"region": label, "start": start, "end": e,
                          "wave_loss": round(w, 4), "std_loss": round(s, 4),
                          "wave_ppl": round(math.exp(w), 1), "std_ppl": round(math.exp(s), 1),
                          "gap_ratio": round(math.exp(w) / max(math.exp(s), 1e-8), 3)})
    worst_data = [{"pos": int(idx), "wave_ppl": round(math.exp(wave_by_pos[idx]), 0),
                   "std_ppl": round(math.exp(std_by_pos[idx]), 0),
                   "gap": round(float(gap_by_pos[idx]), 4)} for idx in worst_idx]
    return {
        "bins": bins_data,
        "worst_positions": worst_data,
        "wave_by_pos": wave_by_pos.tolist(),
        "std_by_pos": std_by_pos.tolist(),
    }


def diagnose_token_types(wave_losses_all, std_losses_all, all_targets, tok, vocab_size):
    """Analyze gap by token type (punctuation, common, rare)."""
    print("\n" + "=" * 70)
    print("  2. PER-TOKEN-TYPE GAP ANALYSIS")
    print("     Which token types does Wave struggle with?")
    print("=" * 70)

    # Flatten
    wave_flat = wave_losses_all.flatten().cpu().numpy()
    std_flat = std_losses_all.flatten().cpu().numpy()
    target_flat = all_targets.flatten().cpu().numpy()

    # Compute per-token-id average loss
    token_wave_loss = np.zeros(vocab_size)
    token_std_loss = np.zeros(vocab_size)
    token_count = np.zeros(vocab_size)

    for i in range(len(target_flat)):
        tid = target_flat[i]
        token_wave_loss[tid] += wave_flat[i]
        token_std_loss[tid] += std_flat[i]
        token_count[tid] += 1

    # Mask out rare tokens (< 5 occurrences)
    valid = token_count >= 5
    token_wave_avg = np.where(valid, token_wave_loss / np.maximum(token_count, 1), 0)
    token_std_avg = np.where(valid, token_std_loss / np.maximum(token_count, 1), 0)
    token_gap = token_wave_avg - token_std_avg

    # Top tokens where Wave is worst relative to Standard
    worst_tokens = np.argsort(token_gap)[-20:][::-1]
    print(f"\n  Tokens where Wave is WORST vs Standard:")
    print(f"  {'Token':<20} {'Count':>6} {'Wave Loss':>10} {'Std Loss':>10} {'Gap':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
    for tid in worst_tokens:
        if not valid[tid]:
            continue
        if tok is not None:
            decoded = repr(tok.decode([int(tid)]))[:18]
        else:
            decoded = f"tok_{tid}"
        print(f"  {decoded:<20} {int(token_count[tid]):>6} {token_wave_avg[tid]:>10.3f} {token_std_avg[tid]:>10.3f} {token_gap[tid]:>8.3f}")

    # Best tokens (where Wave is closest or beats Standard)
    best_tokens = np.argsort(token_gap)[:10]
    print(f"\n  Tokens where Wave is BEST vs Standard:")
    print(f"  {'Token':<20} {'Count':>6} {'Wave Loss':>10} {'Std Loss':>10} {'Gap':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
    for tid in best_tokens:
        if not valid[tid]:
            continue
        if tok is not None:
            decoded = repr(tok.decode([int(tid)]))[:18]
        else:
            decoded = f"tok_{tid}"
        print(f"  {decoded:<20} {int(token_count[tid]):>6} {token_wave_avg[tid]:>10.3f} {token_std_avg[tid]:>10.3f} {token_gap[tid]:>8.3f}")

    # Frequency-based categorization (works without tokenizer)
    # Common tokens (top 500 by count) vs rare tokens
    sorted_by_count = np.argsort(token_count)[::-1]
    common_ids = set(sorted_by_count[:500])
    rare_ids = set(sorted_by_count[500:])
    common_gap = [token_gap[tid] for tid in common_ids if valid[tid]]
    rare_gap = [token_gap[tid] for tid in rare_ids if valid[tid]]

    print(f"\n  By frequency:")
    for name, gaps in [("Common (top 500)", common_gap), ("Rare (rest)", rare_gap)]:
        if gaps:
            print(f"    {name:<20}: mean gap = {np.mean(gaps):.3f} (n={len(gaps)})")

    # Return structured data
    def _token_entry(tid):
        decoded = repr(tok.decode([int(tid)])) if tok is not None else f"tok_{tid}"
        return {"token": decoded[:30], "id": int(tid), "count": int(token_count[tid]),
                "wave_loss": round(float(token_wave_avg[tid]), 4),
                "std_loss": round(float(token_std_avg[tid]), 4),
                "gap": round(float(token_gap[tid]), 4)}
    return {
        "worst_tokens": [_token_entry(tid) for tid in worst_tokens if valid[tid]][:15],
        "best_tokens": [_token_entry(tid) for tid in best_tokens if valid[tid]][:10],
        "common_mean_gap": round(float(np.mean(common_gap)), 4) if common_gap else 0.0,
        "rare_mean_gap": round(float(np.mean(rare_gap)), 4) if rare_gap else 0.0,
    }


def diagnose_repetition(wave_losses_all, std_losses_all, all_inputs, all_targets):
    """Check if Wave struggles specifically with repeated/copied tokens."""
    print("\n" + "=" * 70)
    print("  3. COPY/REPETITION ANALYSIS")
    print("     Does Wave fail when it needs to retrieve a repeated token?")
    print("=" * 70)

    # For each target token, check if it appeared earlier in the sequence
    wave_flat = wave_losses_all.cpu().numpy()
    std_flat = std_losses_all.cpu().numpy()

    repeated_wave = []
    repeated_std = []
    novel_wave = []
    novel_std = []

    n_chunks = all_inputs.shape[0]
    seq_len = all_inputs.shape[1]

    for chunk_idx in range(min(n_chunks, 200)):  # sample 200 chunks
        inputs = all_inputs[chunk_idx].cpu().numpy()
        for pos in range(1, min(seq_len - 1, wave_flat.shape[1])):
            target = inputs[pos + 1]  # what we're predicting
            # Check if this token appeared in the preceding context
            context = inputs[:pos + 1]
            is_repeated = target in context

            w = wave_flat[chunk_idx, pos]
            s = std_flat[chunk_idx, pos]

            if is_repeated:
                repeated_wave.append(w)
                repeated_std.append(s)
            else:
                novel_wave.append(w)
                novel_std.append(s)

    rep_w = np.mean(repeated_wave) if repeated_wave else 0
    rep_s = np.mean(repeated_std) if repeated_std else 0
    nov_w = np.mean(novel_wave) if novel_wave else 0
    nov_s = np.mean(novel_std) if novel_std else 0

    print(f"\n  {'Context':<20} {'Wave Loss':>10} {'Std Loss':>10} {'Gap':>8} {'Ratio':>8} {'Count':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Repeated token':<20} {rep_w:>10.3f} {rep_s:>10.3f} {rep_w-rep_s:>8.3f} {math.exp(rep_w)/math.exp(rep_s):>8.2f}x {len(repeated_wave):>8}")
    print(f"  {'Novel token':<20} {nov_w:>10.3f} {nov_s:>10.3f} {nov_w-nov_s:>8.3f} {math.exp(nov_w)/math.exp(nov_s):>8.2f}x {len(novel_wave):>8}")

    # Break down by distance to nearest repetition
    print(f"\n  Repeated tokens by distance to last occurrence:")
    print(f"  {'Distance':<15} {'Wave Loss':>10} {'Std Loss':>10} {'Gap':>8} {'Count':>8}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    dist_data = []
    dist_bins = [(1, 5), (5, 20), (20, 50), (50, 128), (128, 512)]
    for d_lo, d_hi in dist_bins:
        bin_w, bin_s = [], []
        for chunk_idx in range(min(n_chunks, 200)):
            inputs = all_inputs[chunk_idx].cpu().numpy()
            for pos in range(1, min(seq_len - 1, wave_flat.shape[1])):
                target = inputs[pos + 1]
                context = inputs[:pos + 1]
                positions = np.where(context == target)[0]
                if len(positions) > 0:
                    dist = pos - positions[-1]
                    if d_lo <= dist < d_hi:
                        bin_w.append(wave_flat[chunk_idx, pos])
                        bin_s.append(std_flat[chunk_idx, pos])

        if bin_w:
            mw, ms = np.mean(bin_w), np.mean(bin_s)
            print(f"  {f'{d_lo}-{d_hi}':<15} {mw:>10.3f} {ms:>10.3f} {mw-ms:>8.3f} {len(bin_w):>8}")
            dist_data.append({"range": f"{d_lo}-{d_hi}", "wave_loss": round(float(mw), 4),
                              "std_loss": round(float(ms), 4),
                              "gap": round(float(mw - ms), 4), "count": len(bin_w)})

    return {
        "repeated": {"wave_loss": round(float(rep_w), 4), "std_loss": round(float(rep_s), 4),
                      "gap": round(float(rep_w - rep_s), 4),
                      "ratio": round(math.exp(rep_w) / max(math.exp(rep_s), 1e-8), 3),
                      "count": len(repeated_wave)},
        "novel": {"wave_loss": round(float(nov_w), 4), "std_loss": round(float(nov_s), 4),
                   "gap": round(float(nov_w - nov_s), 4),
                   "ratio": round(math.exp(nov_w) / max(math.exp(nov_s), 1e-8), 3),
                   "count": len(novel_wave)},
        "by_distance": dist_data,
    }


def diagnose_induction(wave_model, std_model, device, vocab_size):
    """Direct induction head test: A B ... A -> should predict B."""
    print("\n" + "=" * 70)
    print("  4. INDUCTION HEAD TEST")
    print("     Feed [A B ... padding ... A] -> does model predict B?")
    print("=" * 70)

    n_tests = 100
    wave_correct = 0
    std_correct = 0
    wave_rank_sum = 0
    std_rank_sum = 0

    for _ in range(n_tests):
        # Random A, B tokens (avoid special tokens)
        A = torch.randint(100, vocab_size, (1,)).item()
        B = torch.randint(100, vocab_size, (1,)).item()
        while B == A:
            B = torch.randint(100, vocab_size, (1,)).item()

        # Build sequence: [A, B, random..., A]
        # The model should predict B after the second A
        seq_len = 64
        padding = torch.randint(100, vocab_size, (seq_len - 3,))
        seq = torch.cat([
            torch.tensor([A, B]),
            padding,
            torch.tensor([A])
        ]).unsqueeze(0).to(device)  # (1, seq_len)

        with torch.no_grad():
            wave_logits, _ = wave_model(seq)
            std_logits, _ = std_model(seq)

        # Check prediction at last position (after second A)
        wave_pred = wave_logits[0, -1, :].argmax().item()
        std_pred = std_logits[0, -1, :].argmax().item()

        if wave_pred == B:
            wave_correct += 1
        if std_pred == B:
            std_correct += 1

        # Rank of B in predictions
        wave_sorted = wave_logits[0, -1, :].argsort(descending=True)
        std_sorted = std_logits[0, -1, :].argsort(descending=True)
        wave_rank = (wave_sorted == B).nonzero(as_tuple=True)[0].item() + 1
        std_rank = (std_sorted == B).nonzero(as_tuple=True)[0].item() + 1
        wave_rank_sum += wave_rank
        std_rank_sum += std_rank

    print(f"\n  Induction accuracy (A B ... A -> B?):")
    print(f"    Standard: {std_correct}/{n_tests} ({std_correct}%)")
    print(f"    Wave:     {wave_correct}/{n_tests} ({wave_correct}%)")
    print(f"\n  Average rank of correct token B:")
    print(f"    Standard: {std_rank_sum/n_tests:.1f}")
    print(f"    Wave:     {wave_rank_sum/n_tests:.1f}")

    # Also test with shorter distances
    print(f"\n  Induction by distance (A B ... [dist tokens] ... A -> B?):")
    print(f"  {'Distance':<12} {'Std Acc':>10} {'Wave Acc':>10} {'Std Rank':>10} {'Wave Rank':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    by_dist = []
    for dist in [5, 10, 20, 50, 100, 200]:
        w_corr, s_corr, w_rank, s_rank = 0, 0, 0, 0
        n = 50
        for _ in range(n):
            A = torch.randint(100, vocab_size, (1,)).item()
            B = torch.randint(100, vocab_size, (1,)).item()
            while B == A:
                B = torch.randint(100, vocab_size, (1,)).item()

            padding = torch.randint(100, vocab_size, (dist,))
            seq = torch.cat([
                torch.tensor([A, B]),
                padding,
                torch.tensor([A])
            ]).unsqueeze(0).to(device)

            with torch.no_grad():
                wl, _ = wave_model(seq)
                sl, _ = std_model(seq)

            if wl[0, -1, :].argmax().item() == B:
                w_corr += 1
            if sl[0, -1, :].argmax().item() == B:
                s_corr += 1

            wr = (wl[0, -1, :].argsort(descending=True) == B).nonzero(as_tuple=True)[0].item() + 1
            sr = (sl[0, -1, :].argsort(descending=True) == B).nonzero(as_tuple=True)[0].item() + 1
            w_rank += wr
            s_rank += sr

        print(f"  {dist:<12} {s_corr/n*100:>9.0f}% {w_corr/n*100:>9.0f}% {s_rank/n:>10.1f} {w_rank/n:>10.1f}")
        by_dist.append({"distance": dist,
                         "wave_acc": round(w_corr / n * 100, 1), "std_acc": round(s_corr / n * 100, 1),
                         "wave_rank": round(w_rank / n, 1), "std_rank": round(s_rank / n, 1)})

    return {
        "overall": {"wave_acc": wave_correct, "std_acc": std_correct, "n_tests": n_tests,
                     "wave_rank": round(wave_rank_sum / n_tests, 1),
                     "std_rank": round(std_rank_sum / n_tests, 1)},
        "by_distance": by_dist,
    }


def main():
    print("=" * 70)
    print("  GAP DIAGNOSTIC: Standard vs Wave Field")
    print("  Finding WHERE the 1.4x PPL gap comes from")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")

    # Load models
    print("\n  Loading models...")
    wave_model, std_model, vocab_size = load_models(device)

    wave_params = sum(p.numel() for p in wave_model.parameters())
    std_params = sum(p.numel() for p in std_model.parameters())
    print(f"  Wave params: {wave_params:,}")
    print(f"  Standard params: {std_params:,}")

    # Load data
    print("\n  Loading validation data...")
    tok, val_tokens = load_tokenizer_and_data()
    print(f"  Val tokens: {len(val_tokens):,}")

    # Create chunks
    seq_len = 512
    n_chunks = min(len(val_tokens) // (seq_len + 1), 300)  # cap at 300 chunks
    print(f"  Using {n_chunks} chunks of {seq_len} tokens")

    # Compute per-token loss for both models
    print("\n  Computing per-token losses...")
    wave_losses_list = []
    std_losses_list = []
    all_inputs_list = []
    all_targets_list = []

    for i in range(n_chunks):
        start = i * seq_len
        chunk = torch.tensor(val_tokens[start:start + seq_len + 1], dtype=torch.long).unsqueeze(0).to(device)
        input_ids = chunk[:, :-1]  # (1, seq_len)
        # targets not needed separately - per_token_loss handles shift internally

        w_loss = per_token_loss(wave_model, chunk, device)  # (seq_len,)
        s_loss = per_token_loss(std_model, chunk, device)   # (seq_len,)

        wave_losses_list.append(w_loss.unsqueeze(0))
        std_losses_list.append(s_loss.unsqueeze(0))
        all_inputs_list.append(chunk)
        all_targets_list.append(chunk[:, 1:])

        if (i + 1) % 50 == 0:
            avg_w = torch.cat(wave_losses_list, dim=0).mean().item()
            avg_s = torch.cat(std_losses_list, dim=0).mean().item()
            print(f"    {i+1}/{n_chunks} chunks | Wave PPL: {math.exp(avg_w):.1f} | Std PPL: {math.exp(avg_s):.1f}")

    wave_losses_all = torch.cat(wave_losses_list, dim=0)  # (n_chunks, seq_len)
    std_losses_all = torch.cat(std_losses_list, dim=0)
    all_inputs = torch.cat(all_inputs_list, dim=0)
    all_targets = torch.cat(all_targets_list, dim=0)

    # Overall PPL
    wave_ppl = math.exp(wave_losses_all.mean().item())
    std_ppl = math.exp(std_losses_all.mean().item())
    print(f"\n  Overall: Wave PPL={wave_ppl:.1f} | Std PPL={std_ppl:.1f} | Gap={wave_ppl/std_ppl:.2f}x")

    # Run diagnostics
    diagnose_per_position(wave_losses_all, std_losses_all, seq_len)
    diagnose_token_types(wave_losses_all, std_losses_all, all_targets, tok, vocab_size)
    diagnose_repetition(wave_losses_all, std_losses_all, all_inputs, all_targets)
    diagnose_induction(wave_model, std_model, device, vocab_size)

    print("\n" + "=" * 70)
    print("  DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
