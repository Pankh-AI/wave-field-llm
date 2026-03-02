"""
Experiment A: PPL Gap vs Sequence Length
========================================

Does Wave Field's PPL gap shrink as sequence length increases?

Same S2-scale model (55M params), same data (WikiText-103), same token budget —
only sequence length changes.

Configs:
  W512)  Wave Field   seq=512   batch=12   — S2 baseline
  S512)  Standard     seq=512   batch=12   — S2 baseline
  W2K)   Wave Field   seq=2048  batch=3    — crossover zone
  S2K)   Standard     seq=2048  batch=3    — crossover zone
  W4K)   Wave Field   seq=4096  batch=2    — the real test
  S4K)   Standard     seq=4096  batch=2    — the real test

Usage:
  CONFIGS=W512,S512   — run one seq length pair
  CONFIGS=all         — run all 6 configs (~1.5 hrs on RTX 3060)

Env vars:
  TOKEN_BUDGET=20000000  — per-config token budget (default 20M)
  FROZEN_HEADS=4         — freeze first 4 heads' damping
  DATASET=103            — WikiText-103 (default)
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
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.wave_field_transformer import WaveFieldTransformer


# ======================================================================
# STANDARD TRANSFORMER BASELINE
# ======================================================================

class StandardTransformer(nn.Module):
    """Standard transformer with O(n^2) attention.
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
# DATA LOADING (WikiText-103 with caching)
# ======================================================================

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '.cache', 'data')


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


def load_data(vocab_size=8000):
    """Load WikiText-103 (default) or WikiText-2 with BPE tokenizer and caching."""
    from datasets import load_dataset as hf_load
    os.makedirs(CACHE_DIR, exist_ok=True)

    choice = os.environ.get('DATASET', '103').strip().lower()
    ds_tag = 'wt103' if choice == '103' else 'wt2'

    tok_path = os.path.join(CACHE_DIR, f'bpe_{vocab_size}.json')
    train_path = os.path.join(CACHE_DIR, f'{ds_tag}_train_{vocab_size}.npy')
    val_path = os.path.join(CACHE_DIR, f'{ds_tag}_val_{vocab_size}.npy')

    # Try loading from cache
    if os.path.exists(tok_path) and os.path.exists(train_path) and os.path.exists(val_path):
        print("  Loading from cache...")
        from tokenizers import Tokenizer
        raw_tok = Tokenizer.from_file(tok_path)
        tok = BPEWrapper(raw_tok)
        train_ids = np.load(train_path).tolist()
        val_ids = np.load(val_path).tolist()
        print(f"  Cached: Train: {len(train_ids):,} | Val: {len(val_ids):,} tokens "
              f"(vocab={tok.vocab_size_actual()})")
        return tok, train_ids, val_ids

    # Cache miss — load and tokenize
    if choice == '103':
        try:
            print("  Loading WikiText-103 (103M tokens)...")
            ds = hf_load("wikitext", "wikitext-103-raw-v1")
        except Exception as e:
            print(f"  WikiText-103 failed ({e}), falling back to WikiText-2...")
            ds = hf_load("wikitext", "wikitext-2-raw-v1")
            ds_tag = 'wt2'
    else:
        print("  Loading WikiText-2...")
        ds = hf_load("wikitext", "wikitext-2-raw-v1")

    splits = {}
    for split_name, hf_split in [('train', 'train'), ('valid', 'validation')]:
        lines = [item['text'].strip() for item in ds[hf_split]
                 if item['text'].strip() and not item['text'].strip().startswith('=')]
        splits[split_name] = lines
    print(f"  Dataset: {ds_tag} — {len(splits['train']):,} train lines")

    # Train BPE on training data
    print("  Training BPE tokenizer...")
    raw_tok = train_bpe_tokenizer(splits['train'], vocab_size=vocab_size)
    tok = BPEWrapper(raw_tok)

    # Tokenize with batch encoding
    def tokenize_lines(lines):
        all_ids = []
        if hasattr(raw_tok, 'encode_batch'):
            CHUNK = 10000
            for i in range(0, len(lines), CHUNK):
                batch = lines[i:i + CHUNK]
                encoded = raw_tok.encode_batch(batch)
                for enc in encoded:
                    if enc.ids:
                        all_ids.extend(enc.ids)
                if (i // CHUNK) % 10 == 0 and i > 0:
                    print(f"    Tokenized {i:,}/{len(lines):,} lines "
                          f"({len(all_ids):,} tokens)...")
        else:
            for line in lines:
                ids = tok.encode(line)
                if ids:
                    all_ids.extend(ids)
        return all_ids

    print("  Tokenizing train split...")
    train_ids = tokenize_lines(splits['train'])
    print("  Tokenizing val split...")
    val_ids = tokenize_lines(splits['valid'])
    print(f"  Train: {len(train_ids):,} | Val: {len(val_ids):,} tokens")

    # Save cache
    raw_tok.save(tok_path)
    np.save(train_path, np.array(train_ids, dtype=np.int32))
    np.save(val_path, np.array(val_ids, dtype=np.int32))
    # Also save .npy without vocab suffix for backwards compat
    train_compat = os.path.join(CACHE_DIR, f'{ds_tag}_train.npy')
    val_compat = os.path.join(CACHE_DIR, f'{ds_tag}_val.npy')
    if not os.path.exists(train_compat):
        np.save(train_compat, np.array(train_ids, dtype=np.int32))
    if not os.path.exists(val_compat):
        np.save(val_compat, np.array(val_ids, dtype=np.int32))
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
# PROGRESS FILE — readable even when stdout is buffered
# ======================================================================

def write_progress(results_dir, config_name, step, total_steps, tokens_M, ppl, acc, tps, eta_min):
    """Write progress to a file so we can monitor via docker exec."""
    progress_path = os.path.join(results_dir, 'progress.txt')
    with open(progress_path, 'w') as f:
        f.write(f"Config: {config_name}\n")
        f.write(f"Step: {step}/{total_steps}\n")
        f.write(f"Tokens: {tokens_M:.1f}M\n")
        f.write(f"PPL: {ppl:.1f}\n")
        f.write(f"Acc: {acc:.1f}%\n")
        f.write(f"Speed: {tps:,.0f} tok/s\n")
        f.write(f"ETA: {eta_min:.0f} min\n")
        f.write(f"Time: {time.strftime('%H:%M:%S')}\n")


# ======================================================================
# CONFIG — S2 scale architecture
# ======================================================================

ARCH = {
    'embedding_dim': 512,
    'num_layers': 12,
    'num_heads': 8,
    'ffn_dim': 2048,
    'peak_lr': 2e-4,
    'token_budget': 20_000_000,
}

# field_size = 2 * seq_len to ensure stride >= 1.0
CONFIGS = [
    {'key': 'W512', 'name': 'Wave seq=512', 'type': 'wave',
     'seq_len': 512, 'batch_size': 6, 'field_size': 1024},
    {'key': 'S512', 'name': 'Standard seq=512', 'type': 'standard',
     'seq_len': 512, 'batch_size': 12},
    {'key': 'W2K', 'name': 'Wave seq=2048', 'type': 'wave',
     'seq_len': 2048, 'batch_size': 2, 'field_size': 4096},
    {'key': 'S2K', 'name': 'Standard seq=2048', 'type': 'standard',
     'seq_len': 2048, 'batch_size': 3},
    {'key': 'W4K', 'name': 'Wave seq=4096', 'type': 'wave',
     'seq_len': 4096, 'batch_size': 1, 'field_size': 4100},
    {'key': 'S4K', 'name': 'Standard seq=4096', 'type': 'standard',
     'seq_len': 4096, 'batch_size': 2},
]


# ======================================================================
# TRAIN
# ======================================================================

def get_results_dir():
    """Get versioned results directory."""
    base = os.path.join(os.path.dirname(__file__), '..', 'results')
    version = os.environ.get('RESULTS_VERSION', '').strip()
    if version:
        vdir = os.path.join(base, version)
        os.makedirs(os.path.join(vdir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(vdir, 'data'), exist_ok=True)
        return vdir
    return base


def train_run(model, train_ids, val_ids, vocab_size, device, cfg, use_amp):
    seq_len = cfg['seq_len']
    batch_size = cfg['batch_size']
    env_budget = os.environ.get('TOKEN_BUDGET', '').strip()
    token_budget = cfg.get('token_budget',
                           int(env_budget) if env_budget else ARCH['token_budget'])
    tokens_per_step = batch_size * seq_len
    total_steps = token_budget // tokens_per_step
    params = sum(p.numel() for p in model.parameters())
    peak_lr = ARCH['peak_lr']

    train_data = make_chunks(train_ids, seq_len)
    val_data = make_chunks(val_ids, seq_len)

    results_dir = get_results_dir()
    run_tag = f"{cfg['key']}_{cfg['type']}_seq{seq_len}"

    print(f"\n  --- {cfg['name']} ---")
    print(f"  Params: {params:,} | Steps: {total_steps:,} | "
          f"Budget: {token_budget/1e6:.0f}M tok | "
          f"seq={seq_len} batch={batch_size} | "
          f"chunks: train={len(train_data)} val={len(val_data)}", flush=True)

    if len(train_data) < 2:
        print(f"  SKIP: not enough training chunks for seq={seq_len}")
        return {
            'run_name': cfg['name'], 'best_ppl': 'SKIP', 'best_acc': 'SKIP',
            'seq_len': seq_len, 'reason': 'insufficient chunks',
        }

    # Optimizer
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
    write_progress(results_dir, cfg['name'], 0, total_steps, 0, vp, va, 0, 0)

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

            if step % 50 == 0:
                elapsed = time.time() - t0
                tps = tokens_seen / elapsed
                print(f"      [{step}/{total_steps}] {tokens_seen/1e6:.1f}M tok | "
                      f"loss {loss.item():.2f} | {tps:,.0f} tok/s", flush=True)

            if step % eval_interval == 0 or tokens_seen >= token_budget:
                vl, vp, va = evaluate(model, val_data, batch_size, vocab_size, device, use_amp)
                mark = ""
                if vp < best_ppl:
                    best_ppl = vp
                    best_acc = va
                    mark = " *BEST"
                    ckpt_path = os.path.join(results_dir, 'checkpoints', f'{run_tag}_best.pt')
                    torch.save(model.state_dict(), ckpt_path)
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
                write_progress(results_dir, cfg['name'], step, total_steps,
                               tokens_seen / 1e6, vp, va, tps, eta_s / 60)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    total_time = time.time() - t0

    return {
        'run_name': cfg['name'],
        'key': cfg['key'],
        'type': cfg['type'],
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
    env_budget = os.environ.get('TOKEN_BUDGET', '').strip()
    budget_str = f"{int(env_budget)/1e6:.0f}M" if env_budget else f"{ARCH['token_budget']/1e6:.0f}M"

    print("=" * 70)
    print("  EXPERIMENT A: PPL Gap vs Sequence Length")
    print(f"  S2 scale (55M params) | {budget_str} tokens/config | WikiText-103")
    print("  Does the gap shrink as N grows?")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)

    tok, train_ids, val_ids = load_data(vocab_size=8000)
    vocab_size = tok.vocab_size_actual()

    # Config filter
    config_filter = os.environ.get('CONFIGS', '').strip().upper()
    if config_filter and config_filter != 'ALL':
        keys = [k.strip() for k in config_filter.split(',')]
        run_configs = [c for c in CONFIGS if c['key'] in keys]
    else:
        run_configs = CONFIGS

    print(f"\n  Running {len(run_configs)} configs: "
          f"{', '.join(c['key'] for c in run_configs)}")

    results_dir = get_results_dir()
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
                local_window = int(os.environ.get('LOCAL_WINDOW', '') or '0')
                n_frozen_heads = int(os.environ.get('FROZEN_HEADS', '') or '0')
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
                    local_window=local_window,
                    n_frozen_heads=n_frozen_heads,
                    device=device,
                    use_analytic_kernel=True,
                    feature_map_depth=2,
                    use_write_gate=False,
                    use_3d_interference=False,
                ).to(device)
                if local_window > 0 or n_frozen_heads > 0:
                    print(f"  Wave config: local_window={local_window}, "
                          f"n_frozen_heads={n_frozen_heads}")

            n_params = sum(p.numel() for p in model.parameters())
            # torch.compile: skip for S2 scale — 10-20 min compile overhead
            # on consumer GPUs outweighs the ~10% training speedup.
            # Only enable at S3+ (100M+ params) where the gain justifies it.
            try:
                if n_params >= 100_000_000:
                    if cfg['type'] == 'wave' and hasattr(model, 'compile_model'):
                        model.compile_model(mode='default')
                        print(f"  [compile] Wave Field: torch.compile applied")
                    elif cfg['type'] == 'standard' and hasattr(torch, 'compile'):
                        model = torch.compile(model)
                        print(f"  [compile] Standard: torch.compile applied")
                else:
                    print(f"  [compile] skipped ({n_params/1e6:.0f}M params — "
                          f"compile overhead > gain below 100M)")
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
                    'run_name': cfg['name'], 'key': cfg['key'], 'type': cfg['type'],
                    'best_ppl': 'OOM', 'best_acc': 'OOM',
                    'seq_len': cfg['seq_len'], 'peak_vram_gb': round(peak_mem, 2),
                })
            else:
                raise
        finally:
            if model is not None:
                del model
            torch.cuda.empty_cache()
            gc.collect()

    # Save results — merge with any existing data from previous runs
    base_results = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(base_results, exist_ok=True)
    vdir = get_results_dir()

    import datetime
    meta = {
        'experiment': 'A — PPL Gap vs Sequence Length',
        'timestamp': datetime.datetime.now().isoformat(),
        'version': os.environ.get('RESULTS_VERSION', 'unknown'),
        'arch': ARCH,
        'frozen_heads': int(os.environ.get('FROZEN_HEADS', '') or '0'),
        'local_window': int(os.environ.get('LOCAL_WINDOW', '') or '0'),
        'token_budget': os.environ.get('TOKEN_BUDGET', str(ARCH['token_budget'])),
        'dataset': os.environ.get('DATASET', '103'),
    }
    for r in all_results:
        r['meta'] = meta

    # Merge: load existing results, update with new ones (by key)
    base_path = os.path.join(base_results, 'data', 'seq_scaling.json')
    existing = []
    if os.path.exists(base_path):
        try:
            with open(base_path, 'r') as f:
                existing = json.load(f)
            print(f"\n  [merge] Loaded {len(existing)} existing results from {base_path}")
        except (json.JSONDecodeError, IOError):
            existing = []

    # Build map: key → result (new results override existing)
    merged = {r['key']: r for r in existing}
    for r in all_results:
        merged[r['key']] = r
        print(f"  [merge] {'Updated' if r['key'] in {e['key'] for e in existing} else 'Added'}: {r['key']}")
    all_results = list(merged.values())

    vdir_data = os.path.join(vdir, 'data')
    os.makedirs(vdir_data, exist_ok=True)
    v_path = os.path.join(vdir_data, 'seq_scaling.json')
    with open(v_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Also save to base results/
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    with open(base_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Summary ──
    print(f"\n{'=' * 78}")
    version_str = os.environ.get('RESULTS_VERSION', '')
    print(f"  EXPERIMENT A: PPL Gap vs Sequence Length ({budget_str} tok/config) {version_str}")
    print(f"  {'Config':<25} {'Seq':>5} {'PPL':>8} {'Acc':>8} "
          f"{'tok/s':>10} {'VRAM':>7} {'Time':>8}")
    print(f"  {'-'*25} {'-'*5} {'-'*8} {'-'*8} {'-'*10} {'-'*7} {'-'*8}")
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

        print(f"  {r['run_name']:<25} {seq:>5} {ppl_s} {acc_s} {tps_s} {vram_s} {t_s}")

    # ── PPL Gap Analysis ──
    print(f"\n  {'─' * 50}")
    print(f"  PPL GAP ANALYSIS (Wave / Standard)")
    print(f"  {'─' * 50}")

    seq_lengths = sorted(set(r.get('seq_len') for r in all_results if r.get('seq_len')))
    for seq in seq_lengths:
        wave_r = [r for r in all_results if r.get('seq_len') == seq and r.get('type') == 'wave']
        std_r = [r for r in all_results if r.get('seq_len') == seq and r.get('type') == 'standard']
        if wave_r and std_r:
            w_ppl = wave_r[0].get('best_ppl', '-')
            s_ppl = std_r[0].get('best_ppl', '-')
            if isinstance(w_ppl, (int, float)) and isinstance(s_ppl, (int, float)):
                gap = w_ppl / s_ppl
                speed_w = wave_r[0].get('tok_per_s', 0)
                speed_s = std_r[0].get('tok_per_s', 0)
                speed_ratio = speed_w / speed_s if speed_s > 0 else 0
                vram_w = wave_r[0].get('peak_vram_gb', 0)
                vram_s = std_r[0].get('peak_vram_gb', 0)
                print(f"  seq={seq:>5}: Wave PPL {w_ppl:>7.1f} / Std PPL {s_ppl:>7.1f} "
                      f"= {gap:.2f}x gap | "
                      f"Speed {speed_ratio:.2f}x | "
                      f"VRAM {vram_w:.1f}G vs {vram_s:.1f}G")
            else:
                print(f"  seq={seq:>5}: Wave={w_ppl} / Std={s_ppl} (incomplete)")

    print(f"{'=' * 78}")
    print(f"\n  Results saved to {os.path.relpath(v_path)}")
    print(f"  Checkpoints in {os.path.relpath(os.path.join(vdir, 'checkpoints'))}")


if __name__ == '__main__':
    main()
