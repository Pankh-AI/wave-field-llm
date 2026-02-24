"""
SPECTRE-Wave Scaling Benchmark
===============================
Tests whether the 3.9x efficiency advantage (PPL 117 vs 457 at 5M tokens)
holds as we scale parameters and data.

4 scaling stages:
  S1: 22M params / 20M tokens   (~25 min A100, ~25 min 3060)
  S2: 55M params / 50M tokens   (~55 min A100, ~2.3 hrs 3060)
  S3: 100M params / 100M tokens (~2 hrs A100, ~9.3 hrs 3060)
  S4: 150M params / 200M tokens (~3 hrs A100, OOM on 3060)

Usage:
  # Run all scales (default):
  python benchmarks/benchmark_scaling.py

  # Run specific scales:
  SCALE=S1 python benchmarks/benchmark_scaling.py
  SCALE=S1,S2 python benchmarks/benchmark_scaling.py

  # Run only wave or standard:
  MODEL=wave python benchmarks/benchmark_scaling.py
  MODEL=standard python benchmarks/benchmark_scaling.py

Data: WikiText-103 (103M tokens) — auto-downloads via HuggingFace datasets.
      WikiText-2 used as fallback if WikiText-103 fails.

References:
  - SPECTRE (arXiv:2502.18394): content-adaptive spectral gating
  - Hedgehog (ICLR 2024): identity-init learned feature maps
  - S4D (arXiv:2206.11893): HiPPO initialization
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
# STANDARD TRANSFORMER (baseline reference)
# ======================================================================

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, num_layers=6,
                 num_heads=8, ffn_dim=1024, max_seq_len=4096, dropout=0.1):
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
# SCALING CONFIGS
# ======================================================================

SCALE_CONFIGS = {
    'S1': {
        'name': 'S1 (22M / 20M tok)',
        'embedding_dim': 384,
        'num_layers': 8,
        'num_heads': 8,
        'ffn_dim': 1536,
        'field_size': 2048,  # must be power-of-2 for cuFFT half precision
        'seq_len': 512,
        'batch_size': 16,
        'token_budget': 20_000_000,
        'peak_lr': 3e-4,
    },
    'S2': {
        'name': 'S2 (55M / 50M tok)',
        'embedding_dim': 512,
        'num_layers': 12,
        'num_heads': 8,
        'ffn_dim': 2048,
        'field_size': 2048,
        'seq_len': 512,
        'batch_size': 12,
        'token_budget': 50_000_000,
        'peak_lr': 2e-4,
    },
    'S3': {
        'name': 'S3 (100M / 100M tok)',
        'embedding_dim': 768,
        'num_layers': 12,
        'num_heads': 12,
        'ffn_dim': 3072,
        'field_size': 2048,
        'seq_len': 512,
        'batch_size': 8,
        'token_budget': 100_000_000,
        'peak_lr': 1.5e-4,
    },
    'S4': {
        'name': 'S4 (150M / 200M tok)',
        'embedding_dim': 1024,
        'num_layers': 12,
        'num_heads': 16,
        'ffn_dim': 4096,
        'field_size': 2048,
        'seq_len': 512,
        'batch_size': 4,
        'token_budget': 200_000_000,
        'peak_lr': 1e-4,
    },
}


# ======================================================================
# BPE TOKENIZER
# ======================================================================

def train_bpe_tokenizer(train_texts, vocab_size=8000):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    return tokenizer


class BPEWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids)
    def vocab_size_actual(self):
        return self.tokenizer.get_vocab_size()


# ======================================================================
# DATA — WikiText-103 (with WikiText-2 fallback)
# ======================================================================

def load_wikitext():
    """Load WikiText. Use DATASET env var to choose: '103' or '2' (default: '2')."""
    from datasets import load_dataset
    choice = os.environ.get('DATASET', '2').strip()
    if choice == '103':
        try:
            print("  Loading WikiText-103 (103M tokens)...")
            ds = load_dataset("wikitext", "wikitext-103-raw-v1")
            dataset_name = "WikiText-103"
        except Exception as e:
            print(f"  WikiText-103 failed ({e}), falling back to WikiText-2...")
            ds = load_dataset("wikitext", "wikitext-2-raw-v1")
            dataset_name = "WikiText-2"
    else:
        print("  Loading WikiText-2 (2.6M tokens)...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        dataset_name = "WikiText-2"

    splits = {}
    for split_name, hf_split in [('train', 'train'), ('valid', 'validation'), ('test', 'test')]:
        lines = [item['text'].strip() for item in ds[hf_split]
                 if item['text'].strip() and not item['text'].strip().startswith('=')]
        splits[split_name] = lines
    print(f"  Dataset: {dataset_name} — {len(splits['train']):,} train lines")
    return splits, dataset_name


def tokenize_corpus(lines, tok):
    all_ids = []
    for line in lines:
        ids = tok.encode(line)
        if ids:
            all_ids.extend(ids)
    return all_ids


def make_chunks(token_ids, seq_len):
    data = []
    for i in range(0, len(token_ids) - seq_len, seq_len):
        chunk = token_ids[i:i + seq_len + 1]
        if len(chunk) == seq_len + 1:
            data.append((torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])))
    return data


def create_batches(data, batch_size, device, shuffle=True):
    if shuffle:
        indices = torch.randperm(len(data)).tolist()
    else:
        indices = list(range(len(data)))
    batches = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        bx = torch.stack([data[i][0] for i in batch_idx]).to(device)
        by = torch.stack([data[i][1] for i in batch_idx]).to(device)
        batches.append((bx, by))
    return batches


# ======================================================================
# MODEL CREATION
# ======================================================================

def create_wave_model(vocab_size, cfg, device):
    """Create SPECTRE-Wave model for a given scale config."""
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
    ).to(device)
    return model


def create_standard_model(vocab_size, cfg, device):
    """Create Standard Transformer for a given scale config."""
    model = StandardTransformer(
        vocab_size=vocab_size,
        embedding_dim=cfg['embedding_dim'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        ffn_dim=cfg['ffn_dim'],
        max_seq_len=cfg['seq_len'] + 2,
        dropout=0.1,
    ).to(device)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ======================================================================
# TRAINING
# ======================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.step_count <= self.warmup_steps:
                lr = base_lr * (self.step_count / self.warmup_steps)
            else:
                p = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * p))
            pg['lr'] = lr


@torch.no_grad()
def evaluate(model, val_data, batch_size, vocab_size, device, use_amp):
    model.eval()
    batches = create_batches(val_data, batch_size, device, shuffle=False)
    total_loss, total_correct, total_tokens, n = 0, 0, 0, 0
    for x, y in batches:
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        total_loss += loss.item()
        n += 1
        mask = y != -100
        total_correct += (logits.argmax(-1)[mask] == y[mask]).sum().item()
        total_tokens += mask.sum().item()
    model.train()
    avg_loss = total_loss / max(n, 1)
    ppl = math.exp(min(avg_loss, 20))
    acc = total_correct / max(total_tokens, 1) * 100
    return avg_loss, ppl, acc


def train_run(model, train_data, val_data, vocab_size, device, run_name,
              total_token_budget, seq_len, batch_size, peak_lr=3e-4,
              use_amp=True):
    """Train a model and return results dict with training curve."""
    params = count_params(model)
    tokens_per_step = batch_size * seq_len
    total_steps = total_token_budget // tokens_per_step

    print(f"\n  --- {run_name} ---")
    print(f"  Params: {params:,} | Context: {seq_len} | Batch: {batch_size}")
    print(f"  Token budget: {total_token_budget:,} | Steps: {total_steps:,}")
    print(f"  Train chunks: {len(train_data):,} | Val chunks: {len(val_data):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01, eps=1e-8)
    warmup = max(total_steps // 10, 100)
    scheduler = WarmupCosineScheduler(optimizer, warmup, total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val_loss = float('inf')
    best_ppl = float('inf')
    best_acc = 0
    tokens_seen = 0
    step = 0
    epoch = 0
    curve = []  # training curve data points

    t0 = time.time()
    eval_interval = max(total_steps // 20, 25)

    # Initial eval (step 0)
    vl, vp, va = evaluate(model, val_data, batch_size, vocab_size, device, use_amp)
    curve.append({'step': 0, 'tokens_M': 0, 'ppl': round(vp, 2), 'acc': round(va, 2), 'time_s': 0})
    print(f"    Step     0/{total_steps} | Tokens 0.0M | Val PPL {vp:>7.1f} Acc {va:>5.1f}% | init", flush=True)

    while tokens_seen < total_token_budget:
        epoch += 1
        batches = create_batches(train_data, batch_size, device, shuffle=True)

        for x, y in batches:
            if tokens_seen >= total_token_budget:
                break

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            tokens_seen += tokens_per_step
            step += 1

            if step % eval_interval == 0 or tokens_seen >= total_token_budget:
                vl, vp, va = evaluate(model, val_data, batch_size, vocab_size, device, use_amp)
                elapsed = time.time() - t0
                tps = tokens_seen / elapsed
                mark = ""
                if vl < best_val_loss:
                    best_val_loss = vl
                    best_ppl = vp
                    best_acc = va
                    mark = " *BEST"
                    # Save best checkpoint
                    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
                    os.makedirs(ckpt_dir, exist_ok=True)
                    safe_name = run_name.replace(' ', '_').lower()
                    ckpt_path = os.path.join(ckpt_dir, f'{safe_name}.pt')
                    torch.save(model.state_dict(), ckpt_path)
                print(f"    Step {step:>5}/{total_steps} | "
                      f"Tokens {tokens_seen/1e6:.1f}M | "
                      f"Val PPL {vp:>7.1f} Acc {va:>5.1f}% | "
                      f"{tps:,.0f} tok/s | "
                      f"{elapsed:.0f}s{mark}", flush=True)
                curve.append({
                    'step': step,
                    'tokens_M': round(tokens_seen / 1e6, 2),
                    'ppl': round(vp, 2),
                    'acc': round(va, 2),
                    'time_s': round(elapsed, 1),
                })

    total_time = time.time() - t0
    final_tps = tokens_seen / total_time

    return {
        'run_name': run_name,
        'params': params,
        'seq_len': seq_len,
        'batch_size': batch_size,
        'best_ppl': round(best_ppl, 2),
        'best_acc': round(best_acc, 2),
        'tokens_seen': tokens_seen,
        'total_time_s': round(total_time, 1),
        'tokens_per_sec': round(final_tps),
        'epochs': epoch,
        'curve': curve,
    }


# ======================================================================
# VRAM ESTIMATION
# ======================================================================

def estimate_vram_gb(params, batch_size, seq_len, field_size, model_type):
    """Rough VRAM estimate (GB) for a given config."""
    # Model weights (fp16 with AMP)
    weight_gb = params * 2 / 1e9  # fp16
    # Optimizer states (fp32 — 2x for Adam momentum + variance)
    optim_gb = params * 4 * 2 / 1e9
    # Activations (rough: batch * seq * hidden * layers * 2 bytes)
    act_gb = batch_size * seq_len * 768 * 12 * 2 / 1e9  # normalized estimate
    # FFT intermediates (wave only): batch * heads * field_size * 8 bytes * layers
    if model_type == 'wave':
        fft_gb = batch_size * 8 * field_size * 8 * 12 / 1e9
    else:
        fft_gb = 0
    return weight_gb + optim_gb + act_gb + fft_gb


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 72)
    print("  SPECTRE-WAVE SCALING BENCHMARK")
    print("  Testing O(n log n) efficiency advantage at scale")
    print("=" * 72)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {vram_gb:.1f} GB")

    # Parse scale filter from environment
    scale_filter = os.environ.get('SCALE', '').strip()
    if scale_filter:
        scale_keys = [s.strip().upper() for s in scale_filter.split(',')]
    else:
        scale_keys = list(SCALE_CONFIGS.keys())

    # Parse model filter from environment
    model_filter = os.environ.get('MODEL', '').strip().lower()
    run_wave = model_filter in ('', 'wave', 'both')
    run_std = model_filter in ('', 'standard', 'std', 'both')

    print(f"\n  Scales: {scale_keys}")
    print(f"  Models: {'SPECTRE-Wave' if run_wave else ''} {'Standard' if run_std else ''}")

    # Load data + tokenizer
    splits, dataset_name = load_wikitext()
    print(f"\n  Training BPE tokenizer (8K vocab)...")
    raw_tok = train_bpe_tokenizer(splits['train'], vocab_size=8000)
    tok = BPEWrapper(raw_tok)
    vocab_size = tok.vocab_size_actual()
    print(f"  Vocab: {vocab_size}")

    print(f"  Tokenizing corpus...")
    train_ids = tokenize_corpus(splits['train'], tok)
    val_ids = tokenize_corpus(splits['valid'], tok)
    print(f"  Train: {len(train_ids):,} tokens | Val: {len(val_ids):,} tokens")

    all_results = []
    run_metadata = {
        'dataset': dataset_name,
        'vocab_size': vocab_size,
        'train_tokens': len(train_ids),
        'val_tokens': len(val_ids),
        'device': str(device),
        'gpu': gpu_name if device.type == 'cuda' else 'cpu',
        'vram_gb': round(vram_gb, 1) if device.type == 'cuda' else 0,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    for scale_key in scale_keys:
        if scale_key not in SCALE_CONFIGS:
            print(f"\n  Unknown scale: {scale_key}, skipping")
            continue

        cfg = SCALE_CONFIGS[scale_key]
        seq_len = cfg['seq_len']
        batch_size = cfg['batch_size']

        print(f"\n{'='*72}")
        print(f"  SCALE: {cfg['name']}")
        print(f"  embed={cfg['embedding_dim']} layers={cfg['num_layers']} "
              f"heads={cfg['num_heads']} ffn={cfg['ffn_dim']}")
        print(f"  field={cfg['field_size']} seq={seq_len} batch={batch_size}")
        print(f"{'='*72}")

        # Prepare data for this scale
        train_data = make_chunks(train_ids, seq_len)
        val_data = make_chunks(val_ids, seq_len)

        # Check if we have enough data for the token budget
        avail_tokens = len(train_data) * seq_len
        if avail_tokens < cfg['token_budget']:
            epochs_needed = math.ceil(cfg['token_budget'] / avail_tokens)
            print(f"  Note: {avail_tokens:,} tokens available, need {cfg['token_budget']:,} "
                  f"({epochs_needed} epochs)")

        # --- SPECTRE-Wave ---
        if run_wave:
            try:
                model = create_wave_model(vocab_size, cfg, device)
                params = count_params(model)
                print(f"\n  SPECTRE-Wave params: {params:,}")

                result = train_run(
                    model, train_data, val_data, vocab_size, device,
                    f"SPECTRE-Wave {scale_key}",
                    cfg['token_budget'], seq_len, batch_size,
                    cfg['peak_lr'], use_amp,
                )
                result['scale'] = scale_key
                result['model_type'] = 'wave'
                all_results.append(result)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n  OOM: SPECTRE-Wave {scale_key} — skipping")
                    all_results.append({
                        'run_name': f'SPECTRE-Wave {scale_key}',
                        'scale': scale_key,
                        'model_type': 'wave',
                        'best_ppl': 'OOM',
                        'best_acc': 'OOM',
                        'error': 'OOM',
                    })
                else:
                    print(f"\n  ERROR: SPECTRE-Wave {scale_key}: {e}")
                    traceback.print_exc()
                    all_results.append({
                        'run_name': f'SPECTRE-Wave {scale_key}',
                        'scale': scale_key,
                        'model_type': 'wave',
                        'best_ppl': 'ERROR',
                        'best_acc': 'ERROR',
                        'error': str(e),
                    })
            finally:
                if 'model' in dir():
                    del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

        # --- Standard Transformer ---
        if run_std:
            try:
                model = create_standard_model(vocab_size, cfg, device)
                params = count_params(model)
                print(f"\n  Standard Transformer params: {params:,}")

                result = train_run(
                    model, train_data, val_data, vocab_size, device,
                    f"Standard {scale_key}",
                    cfg['token_budget'], seq_len, batch_size,
                    cfg['peak_lr'], use_amp,
                )
                result['scale'] = scale_key
                result['model_type'] = 'standard'
                all_results.append(result)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n  OOM: Standard {scale_key} — skipping")
                    all_results.append({
                        'run_name': f'Standard {scale_key}',
                        'scale': scale_key,
                        'model_type': 'standard',
                        'best_ppl': 'OOM',
                        'best_acc': 'OOM',
                        'error': 'OOM',
                    })
                else:
                    print(f"\n  ERROR: Standard {scale_key}: {e}")
                    traceback.print_exc()
                    all_results.append({
                        'run_name': f'Standard {scale_key}',
                        'scale': scale_key,
                        'model_type': 'standard',
                        'best_ppl': 'ERROR',
                        'best_acc': 'ERROR',
                        'error': str(e),
                    })
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
    print("  SCALING BENCHMARK RESULTS")
    print(f"{'='*72}")

    # Group by scale
    for scale_key in scale_keys:
        if scale_key not in SCALE_CONFIGS:
            continue
        scale_results = [r for r in all_results if r.get('scale') == scale_key]
        if not scale_results:
            continue

        cfg = SCALE_CONFIGS[scale_key]
        print(f"\n  === {cfg['name']} ===")
        print(f"  {'Model':<25} {'PPL':>8} {'Acc':>7} {'Params':>12} {'tok/s':>10} {'Time':>8}")
        print(f"  {'-'*25} {'-'*8} {'-'*7} {'-'*12} {'-'*10} {'-'*8}")

        for r in scale_results:
            ppl = r.get('best_ppl', 'N/A')
            acc = r.get('best_acc', 'N/A')
            params = r.get('params', 'N/A')
            tps = r.get('tokens_per_sec', 'N/A')
            t = r.get('total_time_s', 'N/A')
            ppl_s = f"{ppl:>8.1f}" if isinstance(ppl, (int, float)) else f"{ppl:>8}"
            acc_s = f"{acc:>6.1f}%" if isinstance(acc, (int, float)) else f"{acc:>7}"
            params_s = f"{params:>12,}" if isinstance(params, (int, float)) else f"{params:>12}"
            tps_s = f"{tps:>10,}" if isinstance(tps, (int, float)) else f"{tps:>10}"
            t_s = f"{t/60:>7.1f}m" if isinstance(t, (int, float)) else f"{t:>8}"
            model_name = r['run_name'].replace(f' {scale_key}', '')
            print(f"  {model_name:<25} {ppl_s} {acc_s} {params_s} {tps_s} {t_s}")

        # Compute ratio
        wave_r = next((r for r in scale_results if r.get('model_type') == 'wave'), None)
        std_r = next((r for r in scale_results if r.get('model_type') == 'standard'), None)
        if (wave_r and std_r and
                isinstance(wave_r.get('best_ppl'), (int, float)) and
                isinstance(std_r.get('best_ppl'), (int, float))):
            ratio = std_r['best_ppl'] / wave_r['best_ppl']
            print(f"  >> Efficiency ratio: {ratio:.2f}x (SPECTRE/Standard PPL)")

    # Scaling trend
    print(f"\n  --- SCALING TREND ---")
    print(f"  {'Scale':<8} {'Wave PPL':>10} {'Std PPL':>10} {'Ratio':>8}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for scale_key in scale_keys:
        wave_r = next((r for r in all_results
                       if r.get('scale') == scale_key and r.get('model_type') == 'wave'), None)
        std_r = next((r for r in all_results
                      if r.get('scale') == scale_key and r.get('model_type') == 'standard'), None)
        w_ppl = wave_r.get('best_ppl', '-') if wave_r else '-'
        s_ppl = std_r.get('best_ppl', '-') if std_r else '-'
        if isinstance(w_ppl, (int, float)) and isinstance(s_ppl, (int, float)):
            ratio = s_ppl / w_ppl
            print(f"  {scale_key:<8} {w_ppl:>10.1f} {s_ppl:>10.1f} {ratio:>7.2f}x")
        else:
            print(f"  {scale_key:<8} {str(w_ppl):>10} {str(s_ppl):>10}     -")

    # V4.3 reference
    print(f"\n  --- REFERENCE ---")
    print(f"  V4.3 @ 5M tok (8.6M params): SPECTRE PPL 117.6, Standard PPL 457.2 (3.9x)")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    output = {
        'metadata': run_metadata,
        'results': all_results,
    }
    results_path = os.path.join(results_dir, 'scaling_benchmark.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # Also save per-scale for incremental runs
    for scale_key in scale_keys:
        scale_results = [r for r in all_results if r.get('scale') == scale_key]
        if scale_results:
            scale_path = os.path.join(results_dir, f'scaling_{scale_key.lower()}.json')
            with open(scale_path, 'w') as f:
                json.dump({'metadata': run_metadata, 'results': scale_results}, f, indent=2)
            print(f"  Scale {scale_key} saved: {scale_path}")


if __name__ == '__main__':
    main()
