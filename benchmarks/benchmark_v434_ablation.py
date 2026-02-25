"""
V4.3.4 Component Ablation Benchmark
=====================================
Isolates the impact of each V4.3.4 fix by testing them individually
against the V4.3.3 baseline at S1 scale (22M params, 20M tokens).

V4.3.4 made 3 simultaneous changes:
  A) NormalizedExp activation (was ELU+1) — fixes rank-2 collapse
  B) SpectralGate init 10x stronger (0.1 vs 0.01) — gate actually develops
  C) Kernel damping range expanded (-3.0 vs -1.4) — 20 vs 5 position reach

This benchmark tests:
  1) V4.3.3 baseline (all old values)
  2) V4.3.3 + NormalizedExp only
  3) V4.3.3 + Gate 10x only
  4) V4.3.3 + Kernel reach only
  5) V4.3.4 full (all three)
  6) Standard Transformer (reference)

Usage:
  python benchmarks/benchmark_v434_ablation.py
  SEED=42 python benchmarks/benchmark_v434_ablation.py
  WANDB=0 python benchmarks/benchmark_v434_ablation.py
"""

import os
import sys
import time
import json
import math
import gc

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse infrastructure from scaling benchmark
from benchmarks.benchmark_scaling import (
    set_seed, train_bpe_tokenizer, BPEWrapper, load_wikitext,
    tokenize_corpus, make_chunks, create_batches,
    StandardTransformer, count_params, evaluate, train_run,
    WarmupCosineScheduler, _wandb_available,
)
from src.wave_field_transformer import WaveFieldTransformer


# ======================================================================
# ABLATION CONFIGS
# ======================================================================

# V4.3.3 values (before the 3 fixes)
V433_DEFAULTS = {
    'feature_map_activation': 'elu_plus_1',
    'spectral_gate_init_scale': 0.01,
    'damping_range': (-1.4, 0.0),
}

# V4.3.4 values (production)
V434_DEFAULTS = {
    'feature_map_activation': 'normalized_exp',
    'spectral_gate_init_scale': 0.1,
    'damping_range': (-3.0, 0.0),
}

ABLATIONS = {
    'A_v433_baseline': {
        'name': 'V4.3.3 Baseline',
        'desc': 'All V4.3.3 values (ELU+1, gate 0.01, damp -1.4)',
        **V433_DEFAULTS,
    },
    'B_normalized_exp_only': {
        'name': '+NormalizedExp only',
        'desc': 'NormalizedExp activation, rest V4.3.3',
        **V433_DEFAULTS,
        'feature_map_activation': 'normalized_exp',
    },
    'C_gate_10x_only': {
        'name': '+Gate 10x only',
        'desc': 'Gate init 0.1, rest V4.3.3',
        **V433_DEFAULTS,
        'spectral_gate_init_scale': 0.1,
    },
    'D_kernel_reach_only': {
        'name': '+Kernel reach only',
        'desc': 'Damping -3.0, rest V4.3.3',
        **V433_DEFAULTS,
        'damping_range': (-3.0, 0.0),
    },
    'E_v434_full': {
        'name': 'V4.3.4 Full',
        'desc': 'All V4.3.4 values (NormalizedExp, gate 0.1, damp -3.0)',
        **V434_DEFAULTS,
    },
}

# S1 config
S1_CONFIG = {
    'embedding_dim': 384,
    'num_layers': 8,
    'num_heads': 8,
    'ffn_dim': 1536,
    'field_size': 2048,
    'seq_len': 512,
    'batch_size': 16,
    'token_budget': 20_000_000,
    'peak_lr': 3e-4,
}


def create_ablation_model(vocab_size, cfg, ablation_cfg, device):
    """Create SPECTRE-Wave model with specific ablation settings."""
    model = WaveFieldTransformer(
        vocab_size=vocab_size,
        embedding_dim=cfg['embedding_dim'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        ffn_dim=cfg['ffn_dim'],
        field_size=cfg['field_size'],
        max_seq_len=cfg['seq_len'] + 2,
        dropout=0.1,
        use_checkpoint=True,
        interference_interval=3,
        n_components=1,
        local_window=0,
        device=device,
        # Ablation knobs
        feature_map_activation=ablation_cfg['feature_map_activation'],
        spectral_gate_init_scale=ablation_cfg['spectral_gate_init_scale'],
        damping_range=ablation_cfg['damping_range'],
    ).to(device)
    return model


def main():
    print("=" * 72)
    print("  V4.3.4 COMPONENT ABLATION BENCHMARK")
    print("  Isolating each fix: NormalizedExp / Gate 10x / Kernel Reach")
    print("=" * 72)

    seed = int(os.environ.get('SEED', '42'))
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"\n  Device: {device}")
    print(f"  Seed: {seed}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({vram_gb:.1f} GB)")

    cfg = S1_CONFIG

    # Load data + tokenizer
    splits, dataset_name = load_wikitext()
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'cache')
    print(f"\n  Training BPE tokenizer (8K vocab)...")
    raw_tok = train_bpe_tokenizer(splits['train'], vocab_size=8000, cache_dir=cache_dir)
    tok = BPEWrapper(raw_tok)
    vocab_size = tok.vocab_size_actual()

    print(f"  Tokenizing corpus...")
    train_ids = tokenize_corpus(splits['train'], tok)
    val_ids = tokenize_corpus(splits['valid'], tok)
    print(f"  Train: {len(train_ids):,} tokens | Val: {len(val_ids):,} tokens")

    train_data = make_chunks(train_ids, cfg['seq_len'])
    val_data = make_chunks(val_ids, cfg['seq_len'])

    all_results = []
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # --- Run each ablation ---
    for key, abl_cfg in ABLATIONS.items():
        print(f"\n{'='*72}")
        print(f"  ABLATION: {abl_cfg['name']}")
        print(f"  {abl_cfg['desc']}")
        print(f"  activation={abl_cfg['feature_map_activation']}, "
              f"gate_init={abl_cfg['spectral_gate_init_scale']}, "
              f"damping={abl_cfg['damping_range']}")
        print(f"{'='*72}")

        try:
            model = create_ablation_model(vocab_size, cfg, abl_cfg, device)
            params = count_params(model)
            print(f"  Params: {params:,}")

            result = train_run(
                model, train_data, val_data, vocab_size, device,
                f"Ablation-{abl_cfg['name']}",
                cfg['token_budget'], cfg['seq_len'], cfg['batch_size'],
                cfg['peak_lr'], use_amp,
                scale_key='S1', model_type='wave', seed=seed,
            )
            result['ablation'] = key
            result['ablation_name'] = abl_cfg['name']
            result['ablation_config'] = {
                'feature_map_activation': abl_cfg['feature_map_activation'],
                'spectral_gate_init_scale': abl_cfg['spectral_gate_init_scale'],
                'damping_range': list(abl_cfg['damping_range']),
            }
            all_results.append(result)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            all_results.append({
                'run_name': f"Ablation-{abl_cfg['name']}",
                'ablation': key,
                'error': str(e),
            })
        finally:
            if 'model' in dir():
                del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    # --- Standard Transformer reference ---
    print(f"\n{'='*72}")
    print(f"  REFERENCE: Standard Transformer")
    print(f"{'='*72}")

    try:
        model = StandardTransformer(
            vocab_size=vocab_size,
            embedding_dim=cfg['embedding_dim'],
            num_layers=cfg['num_layers'],
            num_heads=cfg['num_heads'],
            ffn_dim=cfg['ffn_dim'],
            max_seq_len=cfg['seq_len'] + 2,
            dropout=0.1,
        ).to(device)
        params = count_params(model)
        print(f"  Params: {params:,}")

        result = train_run(
            model, train_data, val_data, vocab_size, device,
            "Standard Transformer",
            cfg['token_budget'], cfg['seq_len'], cfg['batch_size'],
            cfg['peak_lr'], use_amp,
            scale_key='S1', model_type='standard', seed=seed,
        )
        result['ablation'] = 'F_standard'
        result['ablation_name'] = 'Standard Transformer'
        all_results.append(result)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
    finally:
        if 'model' in dir():
            del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    # ============================================================
    # RESULTS TABLE
    # ============================================================
    print(f"\n{'='*72}")
    print("  V4.3.4 ABLATION RESULTS")
    print(f"{'='*72}")

    print(f"\n  {'Variant':<30} {'PPL':>8} {'Acc':>7} {'Params':>12} {'tok/s':>10}")
    print(f"  {'-'*30} {'-'*8} {'-'*7} {'-'*12} {'-'*10}")

    std_ppl = None
    for r in all_results:
        ppl = r.get('best_ppl', 'N/A')
        acc = r.get('best_acc', 'N/A')
        params = r.get('params', 'N/A')
        tps = r.get('tokens_per_sec', 'N/A')
        name = r.get('ablation_name', r.get('run_name', '?'))

        ppl_s = f"{ppl:>8.1f}" if isinstance(ppl, (int, float)) else f"{ppl:>8}"
        acc_s = f"{acc:>6.1f}%" if isinstance(acc, (int, float)) else f"{acc:>7}"
        params_s = f"{params:>12,}" if isinstance(params, (int, float)) else f"{params:>12}"
        tps_s = f"{tps:>10,}" if isinstance(tps, (int, float)) else f"{tps:>10}"
        print(f"  {name:<30} {ppl_s} {acc_s} {params_s} {tps_s}")

        if r.get('ablation') == 'F_standard' and isinstance(ppl, (int, float)):
            std_ppl = ppl

    # Gap analysis
    if std_ppl:
        print(f"\n  --- GAP ANALYSIS (vs Standard PPL {std_ppl:.1f}) ---")
        print(f"  {'Variant':<30} {'PPL':>8} {'Gap':>8} {'Delta':>8}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")

        baseline_ppl = None
        for r in all_results:
            ppl = r.get('best_ppl')
            if not isinstance(ppl, (int, float)):
                continue
            name = r.get('ablation_name', r.get('run_name', '?'))
            gap = ppl / std_ppl
            if r.get('ablation') == 'A_v433_baseline':
                baseline_ppl = ppl
            delta = f"{((ppl - baseline_ppl) / baseline_ppl * 100):>+7.1f}%" if baseline_ppl else "    base"
            print(f"  {name:<30} {ppl:>8.1f} {gap:>7.2f}x {delta}")

    # Save results
    output = {
        'metadata': {
            'benchmark': 'v434_ablation',
            'dataset': dataset_name,
            'vocab_size': vocab_size,
            'scale': 'S1',
            'seed': seed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'results': all_results,
    }
    results_path = os.path.join(results_dir, 'v434_ablation.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {results_path}")


if __name__ == '__main__':
    main()
