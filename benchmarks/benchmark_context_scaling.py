"""
Context Length Scaling Benchmark
=================================
Proves O(n log n) vs O(n^2) memory scaling.

Runs both SPECTRE-Wave and Standard Transformer for 5 training steps
at increasing context lengths. Measures:
  - Peak VRAM (MB)
  - Throughput (tokens/sec)
  - Whether each model OOMs

Context lengths: 512, 1024, 2048, 4096, 8192, 16384
(Standard should OOM somewhere around 4K-8K on 6GB VRAM)

Usage:
  python benchmarks/benchmark_context_scaling.py
  # or via Docker:
  docker compose run --rm v43 python benchmarks/benchmark_context_scaling.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import json
import gc
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.wave_field_transformer import WaveFieldTransformer

# ======================================================================
# STANDARD TRANSFORMER (same as benchmark_scaling.py)
# ======================================================================

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, num_layers=6,
                 num_heads=8, ffn_dim=1024, max_seq_len=65536, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, mask=None):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, N = input_ids.shape
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(input_ids) + self.positional_embedding(positions)
        x = self.dropout(x)
        causal_mask = torch.triu(
            torch.full((N, N), float('-inf'), device=input_ids.device), diagonal=1
        )
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.norm(x)
        logits = self.output_projection(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), labels.view(-1), ignore_index=-100
            )
        return logits, loss


# ======================================================================
# CONFIG
# ======================================================================

VOCAB_SIZE = 8000
EMBED_DIM = 384
NUM_LAYERS = 8
NUM_HEADS = 8
FFN_DIM = 1536
NUM_STEPS = 5  # just enough to measure stable memory
BATCH_SIZE_MAP = {
    512:   16,
    1024:  8,
    2048:  4,
    4096:  2,
    8192:  1,
    16384: 1,
    32768: 1,
    65536: 1,
}

CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_steps(model, seq_len, batch_size, device, num_steps, use_amp=True):
    """Run num_steps forward+backward passes, return peak VRAM and throughput."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Synthetic data (random tokens)
    x = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device=device)
    y = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device=device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    total_tokens = 0
    t0 = time.time()

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_tokens += batch_size * seq_len

    torch.cuda.synchronize()
    elapsed = time.time() - t0

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0

    return peak_mem_mb, tok_per_sec, elapsed


def test_model(model_name, create_fn, device):
    """Test a model across all context lengths, return results."""
    results = []

    for seq_len in CONTEXT_LENGTHS:
        batch_size = BATCH_SIZE_MAP[seq_len]
        field_size = max(seq_len * 2, 2048)  # field >= 2x seq for wave model
        # Round field_size up to power of 2 for FFT
        field_size = 2 ** math.ceil(math.log2(field_size))

        cleanup()

        print(f"    seq={seq_len:>6d} batch={batch_size:>2d} field={field_size:>6d} ... ",
              end='', flush=True)

        try:
            model = create_fn(seq_len, field_size, batch_size, device)
            params = sum(p.numel() for p in model.parameters())

            # Warmup step (not measured)
            try:
                _ = run_steps(model, seq_len, batch_size, device, num_steps=1)
            except torch.cuda.OutOfMemoryError:
                print(f"OOM (warmup)")
                results.append({
                    'seq_len': seq_len,
                    'batch_size': batch_size,
                    'status': 'OOM',
                    'peak_vram_mb': None,
                    'tok_per_sec': None,
                    'params': params,
                })
                del model
                cleanup()
                continue

            cleanup()
            torch.cuda.reset_peak_memory_stats(device)

            # Measured run
            model = create_fn(seq_len, field_size, batch_size, device)
            peak_mb, tok_s, elapsed = run_steps(
                model, seq_len, batch_size, device, num_steps=NUM_STEPS
            )

            print(f"VRAM {peak_mb:>7.0f} MB | {tok_s:>8,.0f} tok/s | {elapsed:.1f}s")
            results.append({
                'seq_len': seq_len,
                'batch_size': batch_size,
                'status': 'OK',
                'peak_vram_mb': round(peak_mb, 1),
                'tok_per_sec': round(tok_s),
                'params': params,
            })

        except torch.cuda.OutOfMemoryError:
            print(f"OOM")
            results.append({
                'seq_len': seq_len,
                'batch_size': batch_size,
                'status': 'OOM',
                'peak_vram_mb': None,
                'tok_per_sec': None,
                'params': None,
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'seq_len': seq_len,
                'batch_size': batch_size,
                'status': f'ERROR: {str(e)[:80]}',
                'peak_vram_mb': None,
                'tok_per_sec': None,
                'params': None,
            })

        finally:
            if 'model' in dir():
                try:
                    del model
                except:
                    pass
            cleanup()

    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 72)
    print("  CONTEXT LENGTH SCALING BENCHMARK")
    print("  O(n log n) Wave Field vs O(n^2) Standard Transformer")
    print("  5 training steps per context length, measuring peak VRAM")
    print("=" * 72)

    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n  GPU: {gpu_name}")
        print(f"  VRAM: {vram:.1f} GB")
    print(f"  Model: embed={EMBED_DIM}, layers={NUM_LAYERS}, heads={NUM_HEADS}")
    print(f"  Steps per test: {NUM_STEPS}")
    print(f"  Context lengths: {CONTEXT_LENGTHS}")
    print()

    # --- Wave Field ---
    def create_wave(seq_len, field_size, batch_size, dev):
        return WaveFieldTransformer(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBED_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            ffn_dim=FFN_DIM,
            field_size=field_size,
            max_seq_len=seq_len + 2,
            dropout=0.0,
            use_checkpoint=True,
            interference_interval=3,
            n_components=1,
            local_window=0,
            device=dev,
        ).to(dev)

    def create_standard(seq_len, field_size, batch_size, dev):
        return StandardTransformer(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBED_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            ffn_dim=FFN_DIM,
            max_seq_len=seq_len + 2,
            dropout=0.0,
        ).to(dev)

    print("  --- SPECTRE-Wave (V4.3.3) ---")
    wave_results = test_model("SPECTRE-Wave", create_wave, device)

    print()
    print("  --- Standard Transformer ---")
    std_results = test_model("Standard", create_standard, device)

    # --- Summary table ---
    print()
    print("=" * 72)
    print("  CONTEXT SCALING RESULTS")
    print(f"  {'Context':>8s}  {'Batch':>5s}  {'Wave VRAM':>10s}  {'Std VRAM':>10s}  "
          f"{'Wave tok/s':>10s}  {'Std tok/s':>10s}  {'VRAM Ratio':>10s}")
    print(f"  {'':->8s}  {'':->5s}  {'':->10s}  {'':->10s}  {'':->10s}  {'':->10s}  {'':->10s}")

    for w, s in zip(wave_results, std_results):
        seq = w['seq_len']
        batch = w['batch_size']
        w_vram = f"{w['peak_vram_mb']:.0f} MB" if w['status'] == 'OK' else w['status']
        s_vram = f"{s['peak_vram_mb']:.0f} MB" if s['status'] == 'OK' else s['status']
        w_tok = f"{w['tok_per_sec']:,}" if w['status'] == 'OK' else '-'
        s_tok = f"{s['tok_per_sec']:,}" if s['status'] == 'OK' else '-'

        if w['status'] == 'OK' and s['status'] == 'OK':
            ratio = f"{s['peak_vram_mb'] / w['peak_vram_mb']:.2f}x"
        elif w['status'] == 'OK' and s['status'] == 'OOM':
            ratio = "Wave wins!"
        else:
            ratio = '-'

        print(f"  {seq:>8d}  {batch:>5d}  {w_vram:>10s}  {s_vram:>10s}  "
              f"{w_tok:>10s}  {s_tok:>10s}  {ratio:>10s}")

    print("=" * 72)

    # --- Save results ---
    output = {
        'metadata': {
            'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
            'vram_gb': round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1) if device.type == 'cuda' else 0,
            'embed_dim': EMBED_DIM,
            'num_layers': NUM_LAYERS,
            'num_heads': NUM_HEADS,
            'steps_per_test': NUM_STEPS,
        },
        'wave_results': wave_results,
        'standard_results': std_results,
    }

    os.makedirs('results', exist_ok=True)
    with open('results/context_scaling.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to results/context_scaling.json")


if __name__ == '__main__':
    main()
