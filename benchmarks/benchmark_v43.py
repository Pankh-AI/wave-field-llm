"""
V4.3 SPECTRE-Wave Benchmark
============================
Tests the content-adaptive spectral gate (SPECTRE-style) on Wave Field LLM.

3 configs at 5M tokens:
  A) V4.3 Base — Learned feature maps + HiPPO init, NO spectral gate
  B) V4.3 SPECTRE — Full: learned maps + HiPPO + content-adaptive spectral gate
  C) Standard Transformer — reference

Key question: Does content-adaptive kernel modulation close the gap to Standard?
V4.2 best was PPL 997 at 5M. Standard gets ~473. Target: PPL ≤ 500.

References:
  - SPECTRE (arXiv:2502.18394): content-adaptive spectral gating, PPL 39.0 vs Std 39.4
  - Hedgehog (ICLR 2024): identity-init learned feature maps
  - S4D (arXiv:2206.11893): HiPPO initialization for damped oscillators
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
# DATA
# ======================================================================

def load_wikitext2():
    from datasets import load_dataset
    print("  Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    splits = {}
    for split_name, hf_split in [('train', 'train'), ('valid', 'validation'), ('test', 'test')]:
        lines = [item['text'].strip() for item in ds[hf_split]
                 if item['text'].strip() and not item['text'].strip().startswith('=')]
        splits[split_name] = lines
    return splits


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
# WAVE MODEL CREATION
# ======================================================================

def create_wave_model(vocab_size, embedding_dim, num_layers, num_heads,
                      ffn_dim, field_size, seq_len, device,
                      disable_spectral_gate=False):
    """Create WaveFieldTransformer V4.3.

    By default, the model has all V4.3 features:
      - Learned feature maps (Hedgehog identity-init)
      - HiPPO kernel init
      - Content-adaptive spectral gate (SPECTRE-style)

    Set disable_spectral_gate=True to test without the spectral gate
    (isolates its contribution).
    """
    model = WaveFieldTransformer(
        vocab_size=vocab_size, embedding_dim=embedding_dim,
        num_layers=num_layers, num_heads=num_heads,
        ffn_dim=ffn_dim, field_size=field_size,
        max_seq_len=seq_len + 2, dropout=0.1,
        use_checkpoint=True, interference_interval=3,
        n_components=1, local_window=0, device=device,
    ).to(device)

    if disable_spectral_gate:
        # Replace spectral gate with identity: just returns base kernel
        # broadcast to batch dim (B, H, freq_bins)
        for layer in model.layers:
            attn = layer.attention
            # Monkey-patch: make spectral_gate a no-op that passes through base kernel
            class IdentityGate(nn.Module):
                def forward(self, q, base_kernel_fft):
                    return base_kernel_fft  # static kernel, no batch dim
            attn.spectral_gate = IdentityGate()

    return model


# ======================================================================
# TRAINING
# ======================================================================

class WarmupCosineScheduler:
    """Warmup + cosine decay."""

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
              total_token_budget, seq_len, batch_size, peak_lr=0.0003,
              use_amp=True):
    params = sum(p.numel() for p in model.parameters())
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

    t0 = time.time()
    eval_interval = max(total_steps // 20, 25)

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
                print(f"    Step {step:>5}/{total_steps} | "
                      f"Tokens {tokens_seen/1e6:.1f}M | "
                      f"Val PPL {vp:>7.1f} Acc {va:>5.1f}% | "
                      f"{tps:,.0f} tok/s | "
                      f"{elapsed:.0f}s{mark}", flush=True)

    total_time = time.time() - t0
    final_tps = tokens_seen / total_time

    return {
        'run_name': run_name,
        'params': params,
        'seq_len': seq_len,
        'best_ppl': round(best_ppl, 2),
        'best_acc': round(best_acc, 2),
        'tokens_seen': tokens_seen,
        'total_time_s': round(total_time, 1),
        'tokens_per_sec': round(final_tps),
        'epochs': epoch,
    }


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 72)
    print("  V4.3 SPECTRE-WAVE BENCHMARK")
    print("  Learned Feature Maps + HiPPO + Content-Adaptive Spectral Gate")
    print("=" * 72)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data + tokenizer
    splits = load_wikitext2()
    print(f"\n  Training BPE tokenizer (8K vocab)...")
    raw_tok = train_bpe_tokenizer(splits['train'], vocab_size=8000)
    tok = BPEWrapper(raw_tok)
    vocab_size = tok.vocab_size_actual()
    print(f"  Vocab: {vocab_size}")

    print(f"  Tokenizing corpus...")
    train_ids = tokenize_corpus(splits['train'], tok)
    val_ids = tokenize_corpus(splits['valid'], tok)
    print(f"  Train: {len(train_ids):,} tokens | Val: {len(val_ids):,} tokens")

    # Fixed config
    seq_len = 512
    batch_size = 16
    embedding_dim = 256
    num_layers = 6
    num_heads = 8
    ffn_dim = 1024
    field_size = 1024
    peak_lr = 0.0003
    total_token_budget = 5_000_000

    train_data = make_chunks(train_ids, seq_len)
    val_data = make_chunks(val_ids, seq_len)

    # 3-config experiment
    all_configs = [
        {
            'key': 'A',
            'name': 'A) V4.3 Base (no spectral gate)',
            'type': 'wave',
            'disable_spectral_gate': True,
        },
        {
            'key': 'B',
            'name': 'B) V4.3 SPECTRE (full)',
            'type': 'wave',
            'disable_spectral_gate': False,
        },
        {
            'key': 'C',
            'name': 'C) Standard Transformer',
            'type': 'standard',
        },
    ]

    # Optional: CONFIGS env var to run subset (e.g. CONFIGS=B,C)
    config_filter = os.environ.get('CONFIGS', '')
    if config_filter:
        keys = [k.strip().upper() for k in config_filter.split(',')]
        configs = [c for c in all_configs if c['key'] in keys]
        print(f"\n  Running configs: {[c['key'] for c in configs]} (filtered by CONFIGS env)")
    else:
        configs = all_configs

    all_results = []

    for cfg in configs:
        if cfg['type'] == 'standard':
            model = StandardTransformer(
                vocab_size=vocab_size, embedding_dim=embedding_dim,
                num_layers=num_layers, num_heads=num_heads,
                ffn_dim=ffn_dim, max_seq_len=seq_len + 2, dropout=0.1,
            ).to(device)
        else:
            model = create_wave_model(
                vocab_size=vocab_size, embedding_dim=embedding_dim,
                num_layers=num_layers, num_heads=num_heads,
                ffn_dim=ffn_dim, field_size=field_size,
                seq_len=seq_len, device=device,
                disable_spectral_gate=cfg.get('disable_spectral_gate', False),
            )

        try:
            result = train_run(
                model, train_data, val_data, vocab_size, device, cfg['name'],
                total_token_budget, seq_len, batch_size, peak_lr, use_amp,
            )
            all_results.append(result)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  OOM: {cfg['name']} -- skipping")
                all_results.append({
                    'run_name': cfg['name'], 'best_ppl': 'OOM', 'best_acc': 'OOM',
                })
            else:
                raise
        finally:
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    # ============================================================
    # RESULTS TABLE
    # ============================================================
    print(f"\n{'='*72}")
    print("  RESULTS: V4.3 SPECTRE-WAVE BENCHMARK")
    print(f"{'='*72}")
    print(f"\n  {'Config':<42} {'PPL':>8} {'Acc':>7} {'Params':>10} {'tok/s':>10}")
    print(f"  {'-'*42} {'-'*8} {'-'*7} {'-'*10} {'-'*10}")

    for r in all_results:
        ppl = r.get('best_ppl', 'N/A')
        acc = r.get('best_acc', 'N/A')
        params = r.get('params', 'N/A')
        tps = r.get('tokens_per_sec', 'N/A')
        ppl_s = f"{ppl:>8.1f}" if isinstance(ppl, (int, float)) else f"{ppl:>8}"
        acc_s = f"{acc:>6.1f}%" if isinstance(acc, (int, float)) else f"{acc:>7}"
        params_s = f"{params:>10,}" if isinstance(params, (int, float)) else f"{params:>10}"
        tps_s = f"{tps:>10,}" if isinstance(tps, (int, float)) else f"{tps:>10}"
        print(f"  {r['run_name']:<42} {ppl_s} {acc_s} {params_s} {tps_s}")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'v43_benchmark.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # Key comparison
    print(f"\n  --- KEY COMPARISON ---")
    std = next((r for r in all_results if 'Standard' in r.get('run_name', '')), None)
    base = next((r for r in all_results if 'Base' in r.get('run_name', '')), None)
    spectre = next((r for r in all_results if 'SPECTRE' in r.get('run_name', '')), None)

    if base and isinstance(base.get('best_ppl'), (int, float)):
        print(f"  V4.3 Base (5M):      PPL {base['best_ppl']:.1f}, Acc {base['best_acc']:.1f}%")
    if spectre and isinstance(spectre.get('best_ppl'), (int, float)):
        print(f"  V4.3 SPECTRE (5M):   PPL {spectre['best_ppl']:.1f}, Acc {spectre['best_acc']:.1f}%")
        if base and isinstance(base.get('best_ppl'), (int, float)):
            improvement = base['best_ppl'] - spectre['best_ppl']
            print(f"  Spectral gate gain:  {improvement:+.1f} PPL ({improvement/base['best_ppl']*100:.1f}% reduction)")
    if std and isinstance(std.get('best_ppl'), (int, float)):
        print(f"  Standard (5M):       PPL {std['best_ppl']:.1f}, Acc {std['best_acc']:.1f}%")
        if spectre and isinstance(spectre.get('best_ppl'), (int, float)):
            gap = (spectre['best_ppl'] - std['best_ppl']) / std['best_ppl'] * 100
            print(f"  V4.3 vs Standard:    {gap:+.1f}% gap")
    print(f"\n  Previous results:")
    print(f"  V4.1 at 15M tokens:  PPL 543, Acc 10.6%")
    print(f"  V4.2 at 5M tokens:   PPL 997 (best was 2x worse than Standard)")


if __name__ == '__main__':
    main()
