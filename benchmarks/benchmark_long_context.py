"""
Long-Context Benchmark: Wave Field vs Standard Transformer
==========================================================
The critical test: does Wave Field's O(n log n) actually beat
standard transformer's O(n^2) at long sequences?

Measures at each sequence length (256 -> 8192):
  1. Forward pass wall-clock time (ms)
  2. Forward+backward pass wall-clock time (ms)
  3. Peak GPU memory (MB)
  4. Throughput (tokens/sec)

Both models: ~6M params, 256 embed, 6 layers, 8 heads.
Small enough to fit RTX 3060 (6GB) at most lengths.
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
# STANDARD TRANSFORMER BASELINE
# ======================================================================

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, num_layers=6,
                 num_heads=8, ffn_dim=1024, max_seq_len=8192, dropout=0.1):
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
# BENCHMARKING
# ======================================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def safe_cuda_cleanup(device):
    """Aggressively clean CUDA state after OOM."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()


def benchmark_seq_len(model, vocab_size, seq_len, device, warmup=3, repeats=8):
    """
    Run all measurements for a single sequence length.
    Returns dict of results, or None if OOM.
    Handles OOM gracefully without corrupting CUDA state.
    """
    batch_size = 1

    try:
        safe_cuda_cleanup(device)

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # --- Forward pass timing ---
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(warmup):
                _ = model(input_ids)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            fwd_times = []
            for _ in range(repeats):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(input_ids)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                fwd_times.append((time.perf_counter() - t0) * 1000)

        fwd_avg = sum(fwd_times) / len(fwd_times)
        fwd_min = min(fwd_times)

        # --- Forward+backward timing ---
        model.train()
        # Warmup
        for _ in range(min(warmup, 2)):
            model.zero_grad(set_to_none=True)
            _, loss = model(input_ids, labels=labels)
            loss.backward()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        bwd_times = []
        for _ in range(min(repeats, 5)):
            model.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, loss = model(input_ids, labels=labels)
            loss.backward()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            bwd_times.append((time.perf_counter() - t0) * 1000)

        fwd_bwd_avg = sum(bwd_times) / len(bwd_times)

        # --- Peak memory ---
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
            model.zero_grad(set_to_none=True)
            _, loss = model(input_ids, labels=labels)
            loss.backward()
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated(device) / 1e6
        else:
            peak_mem = 0.0

        # --- Loss ---
        model.eval()
        with torch.no_grad():
            _, loss = model(input_ids, labels=labels)
            loss_val = loss.item()

        tokens_per_sec = (batch_size * seq_len) / (fwd_avg / 1000)

        # Cleanup
        del input_ids, labels, loss
        model.zero_grad(set_to_none=True)
        safe_cuda_cleanup(device)

        return {
            'fwd_ms': round(fwd_avg, 2),
            'fwd_min_ms': round(fwd_min, 2),
            'fwd_bwd_ms': round(fwd_bwd_avg, 2),
            'peak_mem_mb': round(peak_mem, 1),
            'tokens_per_sec': round(tokens_per_sec),
            'loss': round(loss_val, 4),
        }

    except RuntimeError as e:
        err_str = str(e).lower()
        if "out of memory" in err_str or "cuda" in err_str:
            # OOM or CUDA error — clean up and report
            model.zero_grad(set_to_none=True)
            safe_cuda_cleanup(device)
            return None
        raise


def print_row(seq_len, r):
    """Print a formatted result row."""
    if r is None:
        print(f"  {seq_len:>8} {'OOM':>10} {'OOM':>10} {'OOM':>10} {'OOM':>12} {'OOM':>8}")
    else:
        print(f"  {seq_len:>8} {r['fwd_ms']:>10.1f} {r['fwd_bwd_ms']:>10.1f} "
              f"{r['peak_mem_mb']:>10.0f} {r['tokens_per_sec']:>12,} {r['loss']:>8.4f}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 72)
    print("  LONG-CONTEXT BENCHMARK")
    print("  Wave Field O(n log n) vs Standard Transformer O(n^2)")
    print("  Sequence lengths: 256 -> 8192")
    print("=" * 72)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    vocab_size = 256
    embedding_dim = 256
    num_layers = 6
    num_heads = 8
    ffn_dim = 1024
    field_size = 1024
    max_seq_len = 8192

    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]

    results = {
        'config': {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'ffn_dim': ffn_dim,
            'field_size': field_size,
        },
        'standard': {},
        'wave': {},
    }

    header = (f"  {'SeqLen':>8} {'Fwd (ms)':>10} {'Fwd+Bwd':>10} "
              f"{'Mem (MB)':>10} {'Tok/s':>12} {'Loss':>8}")
    divider = f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*8}"

    # ============================================================
    # STANDARD TRANSFORMER
    # ============================================================
    print(f"\n{'='*72}")
    print("  STANDARD TRANSFORMER (O(n^2) attention)")
    print(f"{'='*72}")

    std_model = StandardTransformer(
        vocab_size=vocab_size, embedding_dim=embedding_dim,
        num_layers=num_layers, num_heads=num_heads,
        ffn_dim=ffn_dim, max_seq_len=max_seq_len, dropout=0.0,
    ).to(device)
    std_params = count_params(std_model)
    print(f"  Parameters: {std_params:,}")
    print(f"\n{header}")
    print(divider)

    std_oom = False
    for seq_len in seq_lengths:
        if std_oom:
            results['standard'][seq_len] = None
            print_row(seq_len, None)
            continue

        r = benchmark_seq_len(std_model, vocab_size, seq_len, device)
        results['standard'][seq_len] = r
        print_row(seq_len, r)
        sys.stdout.flush()

        if r is None:
            std_oom = True

    del std_model
    safe_cuda_cleanup(device)

    # ============================================================
    # WAVE FIELD
    # ============================================================
    print(f"\n{'='*72}")
    print("  WAVE FIELD V3.5 (O(n log n) wave convolution)")
    print(f"{'='*72}")

    wave_model = WaveFieldTransformer(
        vocab_size=vocab_size, embedding_dim=embedding_dim,
        num_layers=num_layers, num_heads=num_heads,
        ffn_dim=ffn_dim, field_size=field_size,
        max_seq_len=max_seq_len, dropout=0.0,
        use_checkpoint=False, interference_interval=3, device=device,
    ).to(device)
    wave_params = count_params(wave_model)
    print(f"  Parameters: {wave_params:,}")
    print(f"\n{header}")
    print(divider)

    wave_oom = False
    for seq_len in seq_lengths:
        if wave_oom:
            results['wave'][seq_len] = None
            print_row(seq_len, None)
            continue

        r = benchmark_seq_len(wave_model, vocab_size, seq_len, device)
        results['wave'][seq_len] = r
        print_row(seq_len, r)
        sys.stdout.flush()

        if r is None:
            wave_oom = True

    del wave_model
    safe_cuda_cleanup(device)

    # ============================================================
    # COMPARISON
    # ============================================================
    print(f"\n{'='*72}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*72}")
    print(f"\n  Standard: {std_params:,} params | O(n^2)")
    print(f"  Wave:     {wave_params:,} params | O(n log n)")

    # Forward pass comparison
    print(f"\n  --- Forward Pass (ms) ---")
    print(f"  {'SeqLen':>8} {'Standard':>12} {'Wave':>12} {'Speedup':>10} {'Winner':>10}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    crossover_len = None
    for seq_len in seq_lengths:
        sr = results['standard'].get(seq_len)
        wr = results['wave'].get(seq_len)

        s_fwd = sr['fwd_ms'] if sr else None
        w_fwd = wr['fwd_ms'] if wr else None

        if s_fwd is not None and w_fwd is not None:
            speedup = s_fwd / w_fwd
            winner = "WAVE" if speedup > 1.0 else "Std"
            if speedup > 1.0 and crossover_len is None:
                crossover_len = seq_len
            print(f"  {seq_len:>8} {s_fwd:>11.1f} {w_fwd:>11.1f} {speedup:>9.2f}x {winner:>10}")
        else:
            s_str = f"{s_fwd:.1f}" if s_fwd is not None else "OOM"
            w_str = f"{w_fwd:.1f}" if w_fwd is not None else "OOM"
            winner = "WAVE" if s_fwd is None and w_fwd is not None else "---"
            print(f"  {seq_len:>8} {s_str:>12} {w_str:>12} {'---':>10} {winner:>10}")

    # Memory comparison
    print(f"\n  --- Peak Memory (MB) ---")
    print(f"  {'SeqLen':>8} {'Standard':>12} {'Wave':>12} {'Savings':>10}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10}")

    for seq_len in seq_lengths:
        sr = results['standard'].get(seq_len)
        wr = results['wave'].get(seq_len)

        s_mem = sr['peak_mem_mb'] if sr else None
        w_mem = wr['peak_mem_mb'] if wr else None

        if s_mem is not None and w_mem is not None and w_mem > 0:
            ratio = s_mem / w_mem
            print(f"  {seq_len:>8} {s_mem:>11.0f} {w_mem:>11.0f} {ratio:>9.2f}x")
        else:
            s_str = f"{s_mem:.0f}" if s_mem is not None else "OOM"
            w_str = f"{w_mem:.0f}" if w_mem is not None else "OOM"
            print(f"  {seq_len:>8} {s_str:>12} {w_str:>12} {'---':>10}")

    # Scaling analysis
    print(f"\n  --- Scaling Analysis ---")

    std_times, wave_times, valid_lens = [], [], []
    for sl in seq_lengths:
        sr = results['standard'].get(sl)
        wr = results['wave'].get(sl)
        if sr and wr:
            std_times.append(sr['fwd_ms'])
            wave_times.append(wr['fwd_ms'])
            valid_lens.append(sl)

    if len(valid_lens) >= 3:
        import numpy as np
        log_n = np.log(valid_lens)
        log_std = np.log(std_times)
        log_wave = np.log(wave_times)
        std_slope = np.polyfit(log_n, log_std, 1)[0]
        wave_slope = np.polyfit(log_n, log_wave, 1)[0]
        print(f"  Standard: empirical O(n^{std_slope:.2f})  [expected ~O(n^2)]")
        print(f"  Wave:     empirical O(n^{wave_slope:.2f})  [expected ~O(n^1.x)]")

    # Max sequence length each can handle
    std_max = max((sl for sl in seq_lengths if results['standard'].get(sl)), default=0)
    wave_max = max((sl for sl in seq_lengths if results['wave'].get(sl)), default=0)

    # Verdict
    print(f"\n{'='*72}")
    print("  VERDICT")
    print(f"{'='*72}")

    if wave_max > std_max:
        print(f"\n  MEMORY WIN: Wave handles {wave_max} tokens, Standard OOMs at >{std_max}")
    if crossover_len:
        print(f"  SPEED WIN: Wave faster starting at {crossover_len} tokens")
    if not crossover_len and wave_max <= std_max:
        print(f"\n  Standard Transformer wins at all tested lengths on this hardware.")
        print(f"  Wave Field advantage likely requires longer sequences (>8K) or larger models.")

    # Extrapolate crossover if not found
    if not crossover_len and len(valid_lens) >= 3:
        import numpy as np
        log_n = np.log(valid_lens)
        log_std = np.log(std_times)
        log_wave = np.log(wave_times)
        std_coeffs = np.polyfit(log_n, log_std, 1)
        wave_coeffs = np.polyfit(log_n, log_wave, 1)
        # Solve: std_slope * log(n) + std_intercept = wave_slope * log(n) + wave_intercept
        if std_coeffs[0] > wave_coeffs[0]:  # Standard grows faster
            cross_log_n = (wave_coeffs[1] - std_coeffs[1]) / (std_coeffs[0] - wave_coeffs[0])
            cross_n = math.exp(cross_log_n)
            if cross_n > 0:
                print(f"\n  EXTRAPOLATED CROSSOVER: ~{int(cross_n):,} tokens")
                print(f"  (Based on empirical scaling slopes: Std O(n^{std_coeffs[0]:.2f}) vs Wave O(n^{wave_coeffs[0]:.2f}))")

    # Save
    os.makedirs("results", exist_ok=True)
    output = {k: v if not isinstance(v, dict) else
              {str(kk): vv for kk, vv in v.items()} for k, v in results.items()}
    with open("results/long_context_benchmark.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to results/long_context_benchmark.json")

    print(f"\n{'='*72}")
    print("  BENCHMARK COMPLETE")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
