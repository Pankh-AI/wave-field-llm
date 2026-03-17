"""
Full Post-Training Evaluation Pipeline
=======================================

Runs after training to produce a comprehensive evaluation report:
1. Gap analysis (per-position, per-token-type, copy/repetition, induction head)
2. Generation comparison (Wave vs Standard side-by-side with quality metrics)
3. 3x3 visualization dashboard

Usage:
    # Auto-discover from results/ (default after training):
    python scripts/full_eval.py

    # Explicit:
    python scripts/full_eval.py --scale S1 --results-dir results/

Output:
    results/eval/eval_report.json
    results/eval/eval_dashboard.png
"""

import os
import sys
import json
import math
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

# Import model classes
from src.wave_field_transformer import WaveFieldTransformer

# Import StandardTransformer + helpers from benchmark
import importlib.util
_bench_path = os.path.join(_project_root, 'benchmarks', 'benchmark_scaling.py')
_spec = importlib.util.spec_from_file_location("benchmark_scaling", _bench_path)
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)
StandardTransformer = _bench.StandardTransformer
SCALE_CONFIGS = _bench.SCALE_CONFIGS
train_bpe_tokenizer = _bench.train_bpe_tokenizer
BPEWrapper = _bench.BPEWrapper

# Import gap analysis functions
_diag_path = os.path.join(_project_root, 'diagnostics', 'diagnose_gap.py')
_dspec = importlib.util.spec_from_file_location("diagnose_gap", _diag_path)
_diag = importlib.util.module_from_spec(_dspec)
_dspec.loader.exec_module(_diag)


# ======================================================================
# PHASE 1: LOAD RESOURCES
# ======================================================================

def find_results_dir():
    """Auto-discover results directory."""
    candidates = [
        os.path.join(_project_root, 'results'),
        '/app/results',  # Docker
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return os.path.join(_project_root, 'results')


def load_training_results(results_dir, scale_key):
    """Load training curve JSON for both models."""
    data_dir = os.path.join(results_dir, 'data')
    # Try scale-specific file first, then combined
    for name in [f'scaling_{scale_key.lower()}.json', 'scaling_benchmark.json']:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            results = data.get('results', data) if isinstance(data, dict) else data
            if isinstance(results, list):
                wave_r = next((r for r in results if r.get('model_type') == 'wave'), None)
                std_r = next((r for r in results if r.get('model_type') == 'standard'), None)
                return wave_r, std_r
    return None, None


def load_models(results_dir, scale_key, device):
    """Load both Wave and Standard checkpoints."""
    cfg = SCALE_CONFIGS[scale_key]
    ckpt_dir = os.path.join(results_dir, 'checkpoints')
    vocab_size = 8000  # BPE default

    # Wave model
    wave_path = os.path.join(ckpt_dir, 'spectre-wave_s1.pt')
    if not os.path.exists(wave_path):
        # Try scale-specific naming
        for pattern in [f'spectre-wave_{scale_key.lower()}.pt', 'spectre-wave_s1.pt']:
            p = os.path.join(ckpt_dir, pattern)
            if os.path.exists(p):
                wave_path = p
                break

    # Detect hybrid/GLA from env (same as training)
    hybrid_str = os.environ.get('HYBRID_LAYERS', '').strip()
    hybrid_layers = [int(x) for x in hybrid_str.split(',') if x.strip().isdigit()] or None
    gla_str = os.environ.get('GLA_LAYERS', '').strip()
    gla_layers = [int(x) for x in gla_str.split(',') if x.strip().isdigit()] or None
    n_attn_heads = int(os.environ.get('ATTN_HEADS', '') or '0')

    print(f"  Loading Wave: {wave_path}")
    wave_model = WaveFieldTransformer(
        vocab_size=vocab_size,
        embedding_dim=cfg['embedding_dim'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        ffn_dim=cfg['ffn_dim'],
        field_size=cfg['field_size'],
        max_seq_len=cfg['seq_len'] + 2,
        dropout=0.0,
        use_checkpoint=False,
        interference_interval=3,
        n_components=1,
        hybrid_attention_layers=hybrid_layers,
        gla_layers=gla_layers,
        n_attn_heads=n_attn_heads,
        device=device,
    ).to(device)
    state = torch.load(wave_path, map_location=device, weights_only=True)
    info = wave_model.load_state_dict(state, strict=False)
    if info.missing_keys:
        print(f"    WARNING: missing keys: {info.missing_keys[:5]}")
    wave_model.eval()

    # Standard model
    std_path = os.path.join(ckpt_dir, 'standard_s1.pt')
    for pattern in [f'standard_{scale_key.lower()}.pt', 'standard_s1.pt']:
        p = os.path.join(ckpt_dir, pattern)
        if os.path.exists(p):
            std_path = p
            break

    print(f"  Loading Standard: {std_path}")
    std_model = StandardTransformer(
        vocab_size=vocab_size,
        embedding_dim=cfg['embedding_dim'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        ffn_dim=cfg['ffn_dim'],
        max_seq_len=cfg['seq_len'] + 2,
        dropout=0.0,
    ).to(device)
    state = torch.load(std_path, map_location=device, weights_only=True)
    std_model.load_state_dict(state, strict=True)
    std_model.eval()

    wave_params = sum(p.numel() for p in wave_model.parameters())
    std_params = sum(p.numel() for p in std_model.parameters())
    print(f"  Wave: {wave_params:,} params | Standard: {std_params:,} params")

    return wave_model, std_model, vocab_size, wave_params, std_params


def load_tokenizer_and_data(results_dir):
    """Load BPE tokenizer and validation tokens.

    Tries cached tokenizer first (results/cache/bpe_vocab8000.json),
    falls back to rebuilding from HuggingFace dataset.
    Returns (BPEWrapper, np.ndarray) — wrapper has .encode()/.decode().
    """
    cache_dir = os.path.join(results_dir, 'cache')
    val_path = os.path.join(cache_dir, 'wt2_val.npy')
    tok_path = os.path.join(cache_dir, 'bpe_vocab8000.json')

    # Try to load cached tokenizer
    raw_tok = None
    try:
        from tokenizers import Tokenizer
        if os.path.exists(tok_path):
            raw_tok = Tokenizer.from_file(tok_path)
            print(f"  Loaded cached tokenizer from {tok_path}")
    except (ImportError, Exception):
        pass

    # If no cached tokenizer, try rebuilding from dataset
    if raw_tok is None:
        try:
            from datasets import load_dataset as _ld
            print("  Tokenizer cache miss, rebuilding from dataset...")
            ds_choice = os.environ.get('DATASET', '2')
            ds_name = "wikitext-103-raw-v1" if ds_choice == '103' else "wikitext-2-raw-v1"
            ds = _ld("wikitext", ds_name)
            train_lines = [t['text'].strip() for t in ds['train']
                           if t['text'].strip() and not t['text'].strip().startswith('=')]
            raw_tok = train_bpe_tokenizer(train_lines, vocab_size=8000,
                                           cache_dir=cache_dir)
        except ImportError:
            print("  WARNING: tokenizers/datasets not installed. "
                  "Tokenizer unavailable (generation will be skipped).")

    tok = BPEWrapper(raw_tok) if raw_tok is not None else None

    # Try cached val tokens
    if os.path.exists(val_path):
        val_tokens = np.load(val_path)
        print(f"  Loaded {len(val_tokens):,} cached val tokens")
        return tok, val_tokens

    # Rebuild val tokens from dataset
    if tok is None:
        print("  ERROR: No cached val tokens and no tokenizer available.")
        print("  Run training in Docker first to generate caches.")
        return None, None

    try:
        from datasets import load_dataset as _ld
        print("  Val token cache miss, encoding from dataset...")
        ds_choice = os.environ.get('DATASET', '2')
        ds_name = "wikitext-103-raw-v1" if ds_choice == '103' else "wikitext-2-raw-v1"
        ds = _ld("wikitext", ds_name)
        val_lines = [t['text'].strip() for t in ds['validation']
                     if t['text'].strip() and not t['text'].strip().startswith('=')]
        val_text = ' '.join(val_lines)
        val_tokens = np.array(tok.encode(val_text), dtype=np.int64)
        os.makedirs(cache_dir, exist_ok=True)
        np.save(val_path, val_tokens)
        print(f"  Encoded {len(val_tokens):,} val tokens (cached to {val_path})")
        return tok, val_tokens
    except ImportError:
        print("  ERROR: datasets not installed and no cache. "
              "Run training in Docker first.")
        return tok, None


# ======================================================================
# PHASE 2: GAP ANALYSIS (reuses diagnose_gap.py functions)
# ======================================================================

def run_gap_analysis(wave_model, std_model, tok, val_tokens, vocab_size, device,
                     seq_len=512, max_chunks=100):
    """Run all gap diagnostics and return structured results."""
    print("\n  Running gap analysis...")

    n_chunks = min(len(val_tokens) // (seq_len + 1), max_chunks)
    print(f"  Using {n_chunks} chunks of {seq_len} tokens")

    wave_losses_list = []
    std_losses_list = []
    all_inputs_list = []
    all_targets_list = []

    for i in range(n_chunks):
        start = i * seq_len
        chunk = torch.tensor(val_tokens[start:start + seq_len + 1],
                             dtype=torch.long).unsqueeze(0).to(device)
        w_loss = _diag.per_token_loss(wave_model, chunk, device)
        s_loss = _diag.per_token_loss(std_model, chunk, device)

        wave_losses_list.append(w_loss.unsqueeze(0))
        std_losses_list.append(s_loss.unsqueeze(0))
        all_inputs_list.append(chunk)
        all_targets_list.append(chunk[:, 1:])

        if (i + 1) % 25 == 0:
            avg_w = torch.cat(wave_losses_list, dim=0).mean().item()
            avg_s = torch.cat(std_losses_list, dim=0).mean().item()
            print(f"    {i+1}/{n_chunks} | Wave PPL: {math.exp(avg_w):.1f} | Std PPL: {math.exp(avg_s):.1f}")

    wave_losses = torch.cat(wave_losses_list, dim=0)
    std_losses = torch.cat(std_losses_list, dim=0)
    all_inputs = torch.cat(all_inputs_list, dim=0)
    all_targets = torch.cat(all_targets_list, dim=0)

    wave_ppl = math.exp(wave_losses.mean().item())
    std_ppl = math.exp(std_losses.mean().item())
    print(f"  Overall: Wave PPL={wave_ppl:.1f} | Std PPL={std_ppl:.1f} | Gap={wave_ppl/std_ppl:.2f}x")

    # Run diagnostics (each prints AND returns dict)
    pos_result = _diag.diagnose_per_position(wave_losses, std_losses, seq_len)
    type_result = _diag.diagnose_token_types(wave_losses, std_losses, all_targets, tok, vocab_size)
    rep_result = _diag.diagnose_repetition(wave_losses, std_losses, all_inputs, all_targets)
    ind_result = _diag.diagnose_induction(wave_model, std_model, device, vocab_size)

    return {
        "overall_wave_ppl": round(wave_ppl, 2),
        "overall_std_ppl": round(std_ppl, 2),
        "overall_gap": round(wave_ppl / std_ppl, 3),
        "per_position": pos_result,
        "token_types": type_result,
        "repetition": rep_result,
        "induction": ind_result,
    }


# ======================================================================
# PHASE 3: GENERATION COMPARISON
# ======================================================================

EVAL_PROMPTS = [
    "The history of",
    "In recent years, scientists have",
    "The city of London",
    "During the war,",
    "The president announced",
    "The cat sat on the",
    "1, 2, 3, 4, 5,",
    "A B C A B",
]


@torch.no_grad()
def generate_with_metrics(model, input_ids, max_tokens, temperature=0.8, top_k=40):
    """Generate tokens and collect per-step metrics."""
    generated = input_ids[0].tolist()
    entropies = []
    top1_confs = []

    for _ in range(max_tokens):
        if input_ids.shape[1] > 512:
            input_ids = input_ids[:, -512:]
        logits, _ = model(input_ids)
        next_logits = logits[0, -1, :] / temperature

        # Metrics before sampling
        probs = F.softmax(next_logits, dim=-1)
        log_probs = F.log_softmax(next_logits, dim=-1)
        entropy = -(probs * log_probs).sum().item()
        entropies.append(entropy)

        # Top-k filtering
        if top_k > 0:
            topk_vals, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < topk_vals[-1]] = float('-inf')
        probs_filtered = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs_filtered, num_samples=1)

        top1_confs.append(probs[next_token.item()].item())
        generated.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    return generated, entropies, top1_confs


def compute_generation_metrics(token_ids):
    """Compute repetition rate and unique bigram ratio."""
    if len(token_ids) < 3:
        return {"repetition_rate": 0.0, "unique_bigram_ratio": 0.0}
    bigrams = [(token_ids[i], token_ids[i + 1]) for i in range(len(token_ids) - 1)]
    seen = set()
    repeated = 0
    for bg in bigrams:
        if bg in seen:
            repeated += 1
        seen.add(bg)
    return {
        "repetition_rate": round(repeated / max(len(bigrams), 1), 4),
        "unique_bigram_ratio": round(len(seen) / max(len(bigrams), 1), 4),
    }


def run_generation_comparison(wave_model, std_model, tok, device, max_tokens=100):
    """Generate from both models with quality metrics."""
    print("\n  Running generation comparison...")
    results = []

    # Build encode/decode functions that work with both tokenizers.Tokenizer and BPEWrapper
    if tok is None:
        print("    WARNING: tokenizer not available, skipping generation")
        return {"prompts": [], "aggregate": {}}

    def encode_fn(text):
        result = tok.encode(text)
        return result.ids if hasattr(result, 'ids') else result

    def decode_fn(ids):
        return tok.decode(ids)

    for prompt in EVAL_PROMPTS:
        ids = encode_fn(prompt)
        if not ids:
            continue
        input_ids = torch.tensor([ids], device=device)

        # Wave generation
        w_tokens, w_ent, w_conf = generate_with_metrics(
            wave_model, input_ids.clone(), max_tokens)
        # Standard generation
        s_tokens, s_ent, s_conf = generate_with_metrics(
            std_model, input_ids.clone(), max_tokens)

        w_text = decode_fn(w_tokens)
        s_text = decode_fn(s_tokens)

        w_metrics = compute_generation_metrics(w_tokens)
        w_metrics["mean_entropy"] = round(float(np.mean(w_ent)), 4) if w_ent else 0.0
        w_metrics["mean_top1_confidence"] = round(float(np.mean(w_conf)), 4) if w_conf else 0.0

        s_metrics = compute_generation_metrics(s_tokens)
        s_metrics["mean_entropy"] = round(float(np.mean(s_ent)), 4) if s_ent else 0.0
        s_metrics["mean_top1_confidence"] = round(float(np.mean(s_conf)), 4) if s_conf else 0.0

        results.append({
            "prompt": prompt,
            "wave_output": w_text[:500],
            "std_output": s_text[:500],
            "wave_metrics": w_metrics,
            "std_metrics": s_metrics,
        })
        print(f"    \"{prompt[:30]}...\" done")

    # Aggregate
    if results:
        agg_wave = {k: round(float(np.mean([r["wave_metrics"][k] for r in results])), 4)
                    for k in ["repetition_rate", "unique_bigram_ratio", "mean_entropy", "mean_top1_confidence"]}
        agg_std = {k: round(float(np.mean([r["std_metrics"][k] for r in results])), 4)
                   for k in ["repetition_rate", "unique_bigram_ratio", "mean_entropy", "mean_top1_confidence"]}
    else:
        agg_wave, agg_std = {}, {}

    return {"prompts": results, "aggregate": {"wave": agg_wave, "std": agg_std}}


# ======================================================================
# PHASE 4: VISUALIZATION DASHBOARD
# ======================================================================

def plot_dashboard(report, output_path):
    """Generate 3x3 comprehensive evaluation dashboard."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping dashboard")
        return

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f'Wave Field LLM v{report["metadata"]["version"]} -- Full Evaluation ({report["metadata"]["scale"]})',
                 fontsize=14, fontweight='bold')

    # ---- Row 0, Col 0: Training PPL curves ----
    ax = axes[0, 0]
    training = report.get("training", {})
    curve_w = training.get("curve_wave", [])
    curve_s = training.get("curve_std", [])
    if curve_w:
        ax.plot([p['tokens_M'] for p in curve_w], [p['ppl'] for p in curve_w],
                '-', color='#2196F3', linewidth=2, label='Wave')
    if curve_s:
        ax.plot([p['tokens_M'] for p in curve_s], [p['ppl'] for p in curve_s],
                '--', color='#FF5722', linewidth=2, label='Standard')
    ax.set_xlabel('Tokens (M)')
    ax.set_ylabel('PPL')
    ax.set_yscale('log')
    ax.set_title('Training PPL')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Row 0, Col 1: PPL ratio over time ----
    ax = axes[0, 1]
    if curve_w and curve_s and len(curve_w) == len(curve_s):
        tokens = [p['tokens_M'] for p in curve_w]
        ratios = [w['ppl'] / max(s['ppl'], 1e-8)
                  for w, s in zip(curve_w, curve_s)]
        ax.plot(tokens, ratios, '-o', color='#9C27B0', markersize=3, linewidth=2)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Tokens (M)')
        ax.set_ylabel('Wave PPL / Std PPL')
        ax.set_title('PPL Gap Ratio Over Training')
    else:
        ax.text(0.5, 0.5, 'Curves not aligned', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    # ---- Row 0, Col 2: Per-position loss gap ----
    ax = axes[0, 2]
    gap = report.get("gap_analysis", {})
    pos_data = gap.get("per_position", {})
    wave_by_pos = pos_data.get("wave_by_pos", [])
    std_by_pos = pos_data.get("std_by_pos", [])
    if wave_by_pos and std_by_pos:
        gap_by_pos = [w - s for w, s in zip(wave_by_pos, std_by_pos)]
        ax.plot(range(len(gap_by_pos)), gap_by_pos, '-', color='#E91E63', linewidth=1, alpha=0.7)
        # Smoothed
        if len(gap_by_pos) > 20:
            kernel = np.ones(20) / 20
            smoothed = np.convolve(gap_by_pos, kernel, mode='valid')
            ax.plot(range(10, 10 + len(smoothed)), smoothed, '-', color='#E91E63', linewidth=2.5)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Position')
        ax.set_ylabel('Loss Gap (Wave - Std)')
        ax.set_title('Per-Position Loss Gap')
    ax.grid(True, alpha=0.3)

    # ---- Row 1, Col 0: Induction accuracy by distance ----
    ax = axes[1, 0]
    ind_data = gap.get("induction", {})
    by_dist = ind_data.get("by_distance", [])
    if by_dist:
        dists = [d['distance'] for d in by_dist]
        w_acc = [d['wave_acc'] for d in by_dist]
        s_acc = [d['std_acc'] for d in by_dist]
        x = np.arange(len(dists))
        ax.bar(x - 0.17, s_acc, 0.34, label='Standard', color='#FF5722')
        ax.bar(x + 0.17, w_acc, 0.34, label='Wave', color='#2196F3')
        ax.set_xticks(x)
        ax.set_xticklabels(dists)
        ax.set_xlabel('Distance (tokens)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Induction Head: A B...A -> B?')
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # ---- Row 1, Col 1: Repetition gap by distance ----
    ax = axes[1, 1]
    rep_data = gap.get("repetition", {})
    rep_by_dist = rep_data.get("by_distance", [])
    if rep_by_dist:
        labels = [d['range'] for d in rep_by_dist]
        w_loss = [d['wave_loss'] for d in rep_by_dist]
        s_loss = [d['std_loss'] for d in rep_by_dist]
        x = np.arange(len(labels))
        ax.bar(x - 0.17, s_loss, 0.34, label='Standard', color='#FF5722')
        ax.bar(x + 0.17, w_loss, 0.34, label='Wave', color='#2196F3')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, fontsize=8)
        ax.set_xlabel('Distance to Last Occurrence')
        ax.set_ylabel('Loss')
        ax.set_title('Repeated Token Loss by Distance')
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # ---- Row 1, Col 2: Generation entropy comparison ----
    ax = axes[1, 2]
    gen_data = report.get("generation", {})
    agg = gen_data.get("aggregate", {})
    w_agg = agg.get("wave", {})
    s_agg = agg.get("std", {})
    if w_agg and s_agg:
        metrics = ['mean_entropy', 'mean_top1_confidence', 'repetition_rate']
        labels = ['Entropy', 'Top-1 Conf', 'Repetition']
        w_vals = [w_agg.get(m, 0) for m in metrics]
        s_vals = [s_agg.get(m, 0) for m in metrics]
        x = np.arange(len(labels))
        ax.bar(x - 0.17, s_vals, 0.34, label='Standard', color='#FF5722')
        ax.bar(x + 0.17, w_vals, 0.34, label='Wave', color='#2196F3')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title('Generation Quality Metrics')
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # ---- Row 2, Col 0: Best generation sample (Wave) ----
    ax = axes[2, 0]
    ax.axis('off')
    prompts = gen_data.get("prompts", [])
    if prompts:
        p = prompts[0]
        text = f"Prompt: {p['prompt']}\n\nWave:\n{p['wave_output'][:300]}"
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))
    ax.set_title('Wave Generation Sample', fontsize=10)

    # ---- Row 2, Col 1: Best generation sample (Standard) ----
    ax = axes[2, 1]
    ax.axis('off')
    if prompts:
        p = prompts[0]
        text = f"Prompt: {p['prompt']}\n\nStandard:\n{p['std_output'][:300]}"
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#FBE9E7', alpha=0.8))
    ax.set_title('Standard Generation Sample', fontsize=10)

    # ---- Row 2, Col 2: Summary metrics table ----
    ax = axes[2, 2]
    ax.axis('off')
    training = report.get("training", {})
    wave_t = training.get("wave", {})
    std_t = training.get("standard", {})
    ind_overall = ind_data.get("overall", {})

    rows = [
        ['Final PPL', f'{wave_t.get("best_ppl", "?"):.1f}' if isinstance(wave_t.get("best_ppl"), (int, float)) else '?',
                      f'{std_t.get("best_ppl", "?"):.1f}' if isinstance(std_t.get("best_ppl"), (int, float)) else '?',
                      f'{training.get("ppl_gap", "?")}x' if training.get("ppl_gap") else '?'],
        ['Final Acc', f'{wave_t.get("best_acc", "?")}%', f'{std_t.get("best_acc", "?")}%', ''],
        ['Params', f'{wave_t.get("params", 0)/1e6:.1f}M', f'{std_t.get("params", 0)/1e6:.1f}M', ''],
        ['Speed', f'{wave_t.get("tok_per_sec", 0)/1e3:.1f}k', f'{std_t.get("tok_per_sec", 0)/1e3:.1f}k', ''],
        ['Induction @20',
         f'{ind_overall.get("wave_acc", "?")}%' if ind_overall else '?',
         f'{ind_overall.get("std_acc", "?")}%' if ind_overall else '?', ''],
        ['Repetition', f'{w_agg.get("repetition_rate", 0):.2%}', f'{s_agg.get("repetition_rate", 0):.2%}', ''],
        ['Entropy', f'{w_agg.get("mean_entropy", 0):.2f}', f'{s_agg.get("mean_entropy", 0):.2f}', ''],
    ]
    table = ax.table(cellText=rows, colLabels=['Metric', 'Wave', 'Standard', 'Gap'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    # Color header
    for j in range(4):
        table[0, j].set_facecolor('#E0E0E0')
    ax.set_title('Summary', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Dashboard saved: {output_path}")


# ======================================================================
# MAIN
# ======================================================================

def run_full_eval(scale_key='S1', results_dir=None):
    """Main entry point. Can be called from benchmark_scaling.py or standalone."""
    t0 = time.time()
    print("\n" + "=" * 72)
    print("  FULL POST-TRAINING EVALUATION")
    print("=" * 72)

    if results_dir is None:
        results_dir = find_results_dir()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    print(f"  Scale: {scale_key}")
    print(f"  Results: {results_dir}")

    # Version
    try:
        from src import __version__
        version = __version__
    except Exception:
        version = "unknown"

    # Output directory
    eval_dir = os.path.join(results_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    # Phase 1: Load
    print("\n--- Phase 1: Loading resources ---")
    wave_r, std_r = load_training_results(results_dir, scale_key)

    try:
        wave_model, std_model, vocab_size, wave_params, std_params = load_models(
            results_dir, scale_key, device)
    except Exception as e:
        print(f"  ERROR loading models: {e}")
        print("  Cannot continue without checkpoints.")
        return

    tok, val_tokens = load_tokenizer_and_data(results_dir)

    # Phase 2: Gap Analysis
    if val_tokens is not None:
        print("\n--- Phase 2: Gap analysis ---")
        gap_results = run_gap_analysis(wave_model, std_model, tok, val_tokens,
                                       vocab_size, device, max_chunks=100)
    else:
        print("\n--- Phase 2: SKIPPED (no val tokens) ---")
        gap_results = {"overall_wave_ppl": 0, "overall_std_ppl": 0, "overall_gap": 0}

    # Phase 3: Generation Comparison
    if tok is not None:
        print("\n--- Phase 3: Generation comparison ---")
        gen_results = run_generation_comparison(wave_model, std_model, tok, device,
                                                max_tokens=100)
    else:
        print("\n--- Phase 3: SKIPPED (no tokenizer) ---")
        gen_results = {"prompts": [], "aggregate": {}}

    # Build report
    training_info = {
        "wave": {"best_ppl": wave_r.get('best_ppl') if wave_r else None,
                 "best_acc": wave_r.get('best_acc') if wave_r else None,
                 "params": wave_params,
                 "tok_per_sec": wave_r.get('tokens_per_sec', 0) if wave_r else 0},
        "standard": {"best_ppl": std_r.get('best_ppl') if std_r else None,
                     "best_acc": std_r.get('best_acc') if std_r else None,
                     "params": std_params,
                     "tok_per_sec": std_r.get('tokens_per_sec', 0) if std_r else 0},
        "ppl_gap": round(gap_results["overall_wave_ppl"] / max(gap_results["overall_std_ppl"], 1e-8), 3),
        "curve_wave": wave_r.get('curve', []) if wave_r else [],
        "curve_std": std_r.get('curve', []) if std_r else [],
    }

    eval_time = time.time() - t0
    report = {
        "metadata": {
            "version": version,
            "scale": scale_key,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "device": device,
            "eval_time_s": round(eval_time, 1),
        },
        "training": training_info,
        "gap_analysis": gap_results,
        "generation": gen_results,
        "summary": {
            "ppl_gap": training_info["ppl_gap"],
            "wave_ppl": gap_results["overall_wave_ppl"],
            "std_ppl": gap_results["overall_std_ppl"],
            "induction_wave_acc": gap_results.get("induction", {}).get("overall", {}).get("wave_acc"),
            "induction_std_acc": gap_results.get("induction", {}).get("overall", {}).get("std_acc"),
        },
    }

    # Save JSON
    json_path = os.path.join(eval_dir, 'eval_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {json_path}")

    # Phase 4: Visualization
    print("\n--- Phase 4: Visualization ---")
    png_path = os.path.join(eval_dir, 'eval_dashboard.png')
    plot_dashboard(report, png_path)

    print(f"\n  Total eval time: {eval_time:.0f}s")
    print("=" * 72)

    # Free GPU memory
    del wave_model, std_model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return report


def main():
    parser = argparse.ArgumentParser(description='Full post-training evaluation')
    parser.add_argument('--scale', type=str, default='S1', help='Scale config (S1/S2/S3)')
    parser.add_argument('--results-dir', type=str, default=None, help='Results directory')
    args = parser.parse_args()

    run_full_eval(scale_key=args.scale.upper(), results_dir=args.results_dir)


if __name__ == '__main__':
    main()
