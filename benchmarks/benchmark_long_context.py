"""
Long-Context Showdown: The 10km Race
=====================================

Wave Field O(n log n) vs Standard Transformer O(n²) at long sequences.
20M tokens training budget per config.

"Comparing a truck to a sports car in a 100m sprint — of course
the sports car wins. The question is: What happens at 10km?"

Configs (ordered by importance — longest first):
  G) Wave Field   seq=32768 — OOM frontier
  H) Standard     seq=32768 — OOM frontier
  I) Wave Field   seq=65536 — extreme stress test
  J) Standard     seq=65536 — extreme stress test
  E) Wave Field   seq=8192  — pushing the limits
  F) Standard     seq=8192  — pushing the limits
  C) Wave Field   seq=4096  — the real test
  D) Standard     seq=4096  — the real test
  A) Wave Field   seq=512   — short-context reference
  B) Standard     seq=512   — short-context reference

All Wave configs: V4.3 best (analytic kernel, 2L FM, QK LR x3).
No write gate, no 3D interference.

Standard Transformer uses PyTorch SDPA with is_causal=True (no explicit
NxN mask — lets FlashAttention / memory-efficient attention handle it).

Quick run:   CONFIGS=C,D  (~2 hours on RTX 3060)
OOM hunt:    CONFIGS=G,H,I,J
Full run:    all configs
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
# STANDARD TRANSFORMER BASELINE
# ======================================================================

class StandardTransformer(nn.Module):
    """Standard transformer with O(n²) attention.
    Uses PyTorch SDPA (flash attention when available)."""

    def __init__(self, vocab_size, embedding_dim=256, num_layers=6,
                 num_heads=8, ffn_dim=1024, max_seq_len=65538, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
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
        # PyTorch 2.5 TransformerEncoder requires attn_mask with is_causal.
        # Bool mask (1 byte/elem) instead of float (4 bytes) — 4x savings.
        # SDPA ignores the mask when is_causal=True (uses efficient kernel).
        causal_mask = torch.triu(
            torch.ones(N, N, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
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
# DATA HELPERS
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

def load_wikitext2():
    from datasets import load_dataset
    print("  Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    splits = {}
    for split_name, hf_split in [('train', 'train'), ('valid', 'validation')]:
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


# ======================================================================
# DATA CACHING — avoids re-downloading and re-tokenizing every run
# ======================================================================

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '.cache', 'data')

def load_cached_data(vocab_size=8000):
    """Load tokenizer + tokenized IDs from cache, or build and cache them."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tok_path = os.path.join(CACHE_DIR, f'bpe_{vocab_size}.json')
    train_path = os.path.join(CACHE_DIR, f'wikitext2_train_{vocab_size}.pt')
    val_path = os.path.join(CACHE_DIR, f'wikitext2_val_{vocab_size}.pt')

    if os.path.exists(tok_path) and os.path.exists(train_path) and os.path.exists(val_path):
        print("  Loading from cache...")
        from tokenizers import Tokenizer
        raw_tok = Tokenizer.from_file(tok_path)
        tok = BPEWrapper(raw_tok)
        train_ids = torch.load(train_path, weights_only=True).tolist()
        val_ids = torch.load(val_path, weights_only=True).tolist()
        print(f"  Cached: Train: {len(train_ids):,} | Val: {len(val_ids):,} tokens "
              f"(vocab={tok.vocab_size_actual()})")
        return tok, train_ids, val_ids

    # Cache miss — build from scratch
    splits = load_wikitext2()
    print("  Training BPE tokenizer...")
    raw_tok = train_bpe_tokenizer(splits['train'], vocab_size=vocab_size)
    tok = BPEWrapper(raw_tok)
    train_ids = tokenize_corpus(splits['train'], tok)
    val_ids = tokenize_corpus(splits['valid'], tok)
    print(f"  Train: {len(train_ids):,} | Val: {len(val_ids):,} tokens")

    # Save to cache
    raw_tok.save(tok_path)
    torch.save(torch.tensor(train_ids, dtype=torch.int32), train_path)
    torch.save(torch.tensor(val_ids, dtype=torch.int32), val_path)
    print(f"  Saved to cache: {CACHE_DIR}")

    return tok, train_ids, val_ids

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
        bx = torch.stack([data[i][0] for i in batch_idx]).to(device, non_blocking=True)
        by = torch.stack([data[i][1] for i in batch_idx]).to(device, non_blocking=True)
        batches.append((bx, by))
    return batches


# ======================================================================
# LR SCHEDULE
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


# ======================================================================
# EVALUATE
# ======================================================================

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


# ======================================================================
# CONFIG
# ======================================================================

ARCH = {
    'embedding_dim': 384, 'num_layers': 8, 'num_heads': 8,
    'ffn_dim': 1536, 'peak_lr': 3e-4, 'token_budget': 20_000_000,
}

# Ordered by importance — longest context first (OOM frontier → baselines)
CONFIGS = [
    {'key': 'G', 'name': 'G) Wave Field seq=32768', 'type': 'wave',
     'seq_len': 32768, 'batch_size': 1, 'field_size': 65536,
     'token_budget': 32768 * 3},  # 3 steps — OOM probe only
    {'key': 'H', 'name': 'H) Standard seq=32768', 'type': 'standard',
     'seq_len': 32768, 'batch_size': 1,
     'token_budget': 32768 * 3},
    {'key': 'I', 'name': 'I) Wave Field seq=65536', 'type': 'wave',
     'seq_len': 65536, 'batch_size': 1, 'field_size': 131072,
     'token_budget': 65536 * 3},
    {'key': 'J', 'name': 'J) Standard seq=65536', 'type': 'standard',
     'seq_len': 65536, 'batch_size': 1,
     'token_budget': 65536 * 3},
    {'key': 'E', 'name': 'E) Wave Field seq=8192', 'type': 'wave',
     'seq_len': 8192, 'batch_size': 1, 'field_size': 16384},
    {'key': 'F', 'name': 'F) Standard seq=8192', 'type': 'standard',
     'seq_len': 8192, 'batch_size': 1},
    {'key': 'C', 'name': 'C) Wave Field seq=4096', 'type': 'wave',
     'seq_len': 4096, 'batch_size': 2, 'field_size': 8192},
    {'key': 'D', 'name': 'D) Standard seq=4096', 'type': 'standard',
     'seq_len': 4096, 'batch_size': 2},
    {'key': 'A', 'name': 'A) Wave Field seq=512', 'type': 'wave',
     'seq_len': 512, 'batch_size': 16, 'field_size': 2048},
    {'key': 'B', 'name': 'B) Standard seq=512', 'type': 'standard',
     'seq_len': 512, 'batch_size': 16},
]


# ======================================================================
# TRAIN
# ======================================================================

def train_run(model, train_ids, val_ids, vocab_size, device, cfg, use_amp):
    seq_len = cfg['seq_len']
    batch_size = cfg['batch_size']
    # Per-config budget > env var override > default
    env_budget = os.environ.get('TOKEN_BUDGET', '').strip()
    token_budget = cfg.get('token_budget',
                           int(env_budget) if env_budget else ARCH['token_budget'])
    tokens_per_step = batch_size * seq_len
    total_steps = token_budget // tokens_per_step
    params = sum(p.numel() for p in model.parameters())
    peak_lr = ARCH['peak_lr']

    # Create chunks for this seq_len
    train_data = make_chunks(train_ids, seq_len)
    val_data = make_chunks(val_ids, seq_len)

    print(f"\n  --- {cfg['name']} ---")
    print(f"  Params: {params:,} | Steps: {total_steps:,} | "
          f"seq={seq_len} batch={batch_size} | "
          f"chunks: train={len(train_data)} val={len(val_data)}", flush=True)

    if len(train_data) < 2:
        print(f"  SKIP: not enough training chunks for seq={seq_len}")
        return {
            'run_name': cfg['name'], 'best_ppl': 'SKIP', 'best_acc': 'SKIP',
            'seq_len': seq_len, 'reason': 'insufficient chunks',
        }

    # Optimizer — QK LR x3 for wave field
    if hasattr(model, 'configure_optimizer'):
        optimizer = model.configure_optimizer(base_lr=peak_lr, qk_lr_mult=3.0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                                      weight_decay=0.01, eps=1e-8)

    warmup = max(total_steps // 10, 50)
    scheduler = WarmupCosineScheduler(optimizer, warmup, total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_ppl = float('inf')
    best_acc = 0
    tokens_seen = 0
    step = 0
    epoch = 0
    curve = []

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    eval_interval = max(total_steps // 10, 10)

    # Initial eval
    _, vp, va = evaluate(model, val_data, batch_size, vocab_size, device, use_amp)
    curve.append({'tokens_M': 0, 'ppl': round(vp, 2), 'acc': round(va, 2)})
    print(f"    Step 0 | PPL {vp:>7.1f} Acc {va:>5.1f}% | init", flush=True)

    while tokens_seen < token_budget:
        epoch += 1
        batches = create_batches(train_data, batch_size, device, shuffle=True)
        for x, y in batches:
            if tokens_seen >= token_budget:
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

            if step % eval_interval == 0 or tokens_seen >= token_budget:
                vl, vp, va = evaluate(model, val_data, batch_size, vocab_size, device, use_amp)
                mark = ""
                if vp < best_ppl:
                    best_ppl = vp
                    best_acc = va
                    mark = " *BEST"
                elapsed = time.time() - t0
                tps = tokens_seen / elapsed
                eta_s = (token_budget - tokens_seen) / max(tps, 1)
                print(f"    Step {step:>5}/{total_steps} | "
                      f"Tok {tokens_seen/1e6:.1f}M | "
                      f"PPL {vp:>7.1f} Acc {va:>5.1f}% | "
                      f"{tps:,.0f} tok/s | "
                      f"ETA {eta_s/60:.0f}m{mark}", flush=True)
                curve.append({
                    'tokens_M': round(tokens_seen / 1e6, 2),
                    'ppl': round(vp, 2),
                    'acc': round(va, 2),
                })

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    total_time = time.time() - t0

    return {
        'run_name': cfg['name'],
        'params': params,
        'best_ppl': round(best_ppl, 2),
        'best_acc': round(best_acc, 2),
        'tokens_seen': tokens_seen,
        'time_s': round(total_time, 1),
        'tok_per_s': round(tokens_seen / total_time),
        'peak_vram_gb': round(peak_mem, 2),
        'seq_len': cfg['seq_len'],
        'batch_size': cfg['batch_size'],
        'curve': curve,
    }


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("  LONG-CONTEXT SHOWDOWN: THE 10km RACE")
    print("  Wave Field O(n log n) vs Standard O(n^2) at long sequences")
    print("  20M tokens training budget per config")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    # TF32 for Ampere+ GPUs: ~2x speedup on fp32 matmul/FFT operations
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  TF32: enabled | cudnn.benchmark: enabled")

    tok, train_ids, val_ids = load_cached_data(vocab_size=8000)
    vocab_size = tok.vocab_size_actual()

    # Config filter
    config_filter = os.environ.get('CONFIGS', '').strip().upper()
    if config_filter:
        keys = [k.strip() for k in config_filter.split(',')]
        run_configs = [c for c in CONFIGS if c['key'] in keys]
    else:
        run_configs = CONFIGS

    print(f"\n  Running {len(run_configs)} configs: "
          f"{', '.join(c['key'] for c in run_configs)}")

    all_results = []

    for cfg in run_configs:
        torch.cuda.empty_cache()
        gc.collect()

        model = None
        try:
            if cfg['type'] == 'standard':
                model = StandardTransformer(
                    vocab_size=vocab_size,
                    embedding_dim=ARCH['embedding_dim'],
                    num_layers=ARCH['num_layers'],
                    num_heads=ARCH['num_heads'],
                    ffn_dim=ARCH['ffn_dim'],
                    max_seq_len=cfg['seq_len'] + 2,
                    dropout=0.1,
                ).to(device)
            else:
                model = WaveFieldTransformer(
                    vocab_size=vocab_size,
                    embedding_dim=ARCH['embedding_dim'],
                    num_layers=ARCH['num_layers'],
                    num_heads=ARCH['num_heads'],
                    ffn_dim=ARCH['ffn_dim'],
                    field_size=cfg['field_size'],
                    max_seq_len=cfg['seq_len'] + 2,
                    dropout=0.1,
                    use_checkpoint=True,
                    interference_interval=3,
                    n_components=1,
                    local_window=0,
                    device=device,
                    use_analytic_kernel=True,
                    feature_map_depth=2,
                    use_write_gate=False,
                    use_3d_interference=False,
                ).to(device)

            # torch.compile: only beneficial at 50M+ params (overhead > gain below)
            n_params = sum(p.numel() for p in model.parameters())
            try:
                if n_params >= 50_000_000:
                    if cfg['type'] == 'wave' and hasattr(model, 'compile_model'):
                        model.compile_model(mode='default')
                        print(f"  [compile] Wave Field: torch.compile applied")
                    elif cfg['type'] == 'standard' and hasattr(torch, 'compile'):
                        model = torch.compile(model)
                        print(f"  [compile] Standard: torch.compile applied")
                else:
                    print(f"  [compile] skipped ({n_params/1e6:.0f}M params < 50M threshold)")
            except Exception as e:
                print(f"  [compile] skipped ({e})")

            result = train_run(model, train_ids, val_ids, vocab_size,
                               device, cfg, use_amp)
            all_results.append(result)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                peak_mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"\n  ** OOM: {cfg['name']} ** (peak VRAM: {peak_mem:.1f} GB)")
                all_results.append({
                    'run_name': cfg['name'], 'best_ppl': 'OOM', 'best_acc': 'OOM',
                    'seq_len': cfg['seq_len'], 'peak_vram_gb': round(peak_mem, 2),
                })
            else:
                raise
        finally:
            if model is not None:
                del model
            torch.cuda.empty_cache()
            gc.collect()

    # Save results (merge with existing to preserve prior runs)
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'long_context.json')
    existing = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)
    # Update existing results with new ones (keyed by run_name)
    existing_names = {r['run_name'] for r in existing}
    for r in all_results:
        if r['run_name'] in existing_names:
            existing = [e if e['run_name'] != r['run_name'] else r for e in existing]
        else:
            existing.append(r)
    with open(results_path, 'w') as f:
        json.dump(existing, f, indent=2)

    # Summary
    print(f"\n{'=' * 78}")
    print(f"  LONG-CONTEXT SHOWDOWN RESULTS (20M tokens, WikiText-2)")
    print(f"  {'Config':<30} {'Seq':>5} {'PPL':>8} {'Acc':>8} "
          f"{'tok/s':>10} {'VRAM':>7} {'Time':>8}")
    print(f"  {'-'*30} {'-'*5} {'-'*8} {'-'*8} {'-'*10} {'-'*7} {'-'*8}")
    for r in all_results:
        seq = r.get('seq_len', '?')
        ppl = r.get('best_ppl', '-')
        acc = r.get('best_acc', '-')
        tps = r.get('tok_per_s', '-')
        vram = r.get('peak_vram_gb', '-')
        t = r.get('time_s', '-')

        ppl_s = f"{ppl:>8.1f}" if isinstance(ppl, (int, float)) else f"{ppl:>8}"
        acc_s = f"{acc:>7.1f}%" if isinstance(acc, (int, float)) else f"{acc:>8}"
        tps_s = f"{tps:>10,}" if isinstance(tps, (int, float)) else f"{tps:>10}"
        vram_s = f"{vram:>6.1f}G" if isinstance(vram, (int, float)) else f"{vram:>7}"
        t_s = f"{t:>7.0f}s" if isinstance(t, (int, float)) else f"{t:>8}"

        print(f"  {r['run_name']:<30} {seq:>5} {ppl_s} {acc_s} {tps_s} {vram_s} {t_s}")
    print(f"{'=' * 78}")
    print("\n  Results saved to results/long_context.json")


if __name__ == '__main__':
    main()
