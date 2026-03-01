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

  # Reproducibility + tracking:
  SEED=42 python benchmarks/benchmark_scaling.py    # set random seed (default: 42)
  WANDB=0 python benchmarks/benchmark_scaling.py    # disable wandb logging
  RESUME=1 python benchmarks/benchmark_scaling.py   # auto-resume from checkpoint

Data: WikiText-103 (default, 103M tokens), WikiText-2, or OpenWebText.
      Set DATASET=103 (default), DATASET=2, or DATASET=owt via env var.

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
import random
import traceback
import numpy as np

# Optional wandb (disable with WANDB=0 or if not installed)
_wandb_available = False
if os.environ.get('WANDB', '1') != '0':
    try:
        import wandb
        _wandb_available = True
    except ImportError:
        pass


def set_seed(seed: int, deterministic: bool = True):
    """Set all random seeds for reproducibility.

    Args:
        deterministic: If False, enables cudnn.benchmark for +5-10% speed
                       at the cost of ±1-2 PPL variance between runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.wave_field_transformer import WaveFieldTransformer

# Optional training monitor (disable with MONITOR=0)
_monitor_available = False
if os.environ.get('MONITOR', '1') != '0':
    try:
        from diagnostics.training_monitor import WaveFieldMonitor
        _monitor_available = True
    except ImportError:
        pass


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
        'field_size': 512,   # = seq_len: 4x smaller FFT (pad 1024 vs 4096), stride=1.0
        'seq_len': 512,
        'batch_size': 16,
        'token_budget': 20_000_000,
        'peak_lr': 3e-4,
        'use_checkpoint': True,   # 6GB laptop GPU needs checkpoint with monitor+saves
    },
    'S2': {
        'name': 'S2 (55M / 50M tok)',
        'embedding_dim': 512,
        'num_layers': 12,
        'num_heads': 8,
        'ffn_dim': 2048,
        'field_size': 512,   # = seq_len: 4x smaller FFT, stride=1.0
        'seq_len': 512,
        'batch_size': 12,
        'token_budget': 50_000_000,
        'peak_lr': 2e-4,
        'use_checkpoint': True,   # needs checkpoint on consumer GPUs
    },
    'S3': {
        'name': 'S3 (100M / 100M tok)',
        'embedding_dim': 768,
        'num_layers': 12,
        'num_heads': 12,
        'ffn_dim': 3072,
        'field_size': 512,   # = seq_len: 4x smaller FFT, stride=1.0
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
        'field_size': 512,   # = seq_len: 4x smaller FFT, stride=1.0
        'seq_len': 512,
        'batch_size': 4,
        'token_budget': 200_000_000,
        'peak_lr': 1e-4,
    },
}


# ======================================================================
# BPE TOKENIZER
# ======================================================================

def train_bpe_tokenizer(train_texts, vocab_size=8000, cache_dir=None):
    """Train BPE tokenizer with caching. Returns same tokenizer for same data+vocab."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    # Check cache first
    if cache_dir:
        cache_path = os.path.join(cache_dir, f'bpe_vocab{vocab_size}.json')
        if os.path.exists(cache_path):
            print(f"  Loading cached tokenizer: {cache_path}")
            return Tokenizer.from_file(cache_path)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(train_texts, trainer=trainer)

    # Save to cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        tokenizer.save(cache_path)
        print(f"  Tokenizer cached: {cache_path}")

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
# DATA — WikiText-103 (default), WikiText-2, or OpenWebText
# ======================================================================

def load_dataset_splits():
    """Load dataset. Use DATASET env var: '103' (default), '2', or 'owt'.

    WikiText-103: 103M tokens — enough for S1-S3 without repetition.
    WikiText-2:   2.6M tokens — only good for quick tests (S1 needs 8 epochs).
    OpenWebText:  ~8B tokens — subset loaded for S3/S4 scale.
    """
    from datasets import load_dataset
    choice = os.environ.get('DATASET', '103').strip().lower()

    if choice == 'owt' or choice == 'openwebtext':
        # OpenWebText: load a subset (100K docs ≈ 50-100M tokens)
        owt_size = os.environ.get('OWT_SIZE', '100000').strip()
        try:
            print(f"  Loading OpenWebText (first {owt_size} docs)...")
            ds = load_dataset("openwebtext", split=f"train[:{owt_size}]")
            train_lines = [item['text'].strip() for item in ds
                           if item['text'].strip()]
            # Use WikiText-2 for validation (OWT has no val split)
            val_ds = load_dataset("wikitext", "wikitext-2-raw-v1")
            val_lines = [item['text'].strip() for item in val_ds['validation']
                         if item['text'].strip() and not item['text'].strip().startswith('=')]
            test_lines = [item['text'].strip() for item in val_ds['test']
                          if item['text'].strip() and not item['text'].strip().startswith('=')]
            splits = {'train': train_lines, 'valid': val_lines, 'test': test_lines}
            dataset_name = f"OpenWebText-{owt_size}"
            print(f"  Dataset: {dataset_name} — {len(splits['train']):,} train docs")
            return splits, dataset_name
        except Exception as e:
            print(f"  OpenWebText failed ({e}), falling back to WikiText-103...")
            choice = '103'

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


def tokenize_corpus(lines, tok, cache_path=None):
    """Tokenize text lines, with optional disk caching. Uses batch encoding for speed."""
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached tokens: {cache_path}")
        return np.load(cache_path).tolist()

    # Use encode_batch for 10-20x faster tokenization (HuggingFace tokenizers)
    all_ids = []
    raw_tok = tok.tokenizer if hasattr(tok, 'tokenizer') else None
    if raw_tok and hasattr(raw_tok, 'encode_batch'):
        CHUNK = 10000
        for i in range(0, len(lines), CHUNK):
            batch = lines[i:i + CHUNK]
            encoded = raw_tok.encode_batch(batch)
            for enc in encoded:
                if enc.ids:
                    all_ids.extend(enc.ids)
            if (i // CHUNK) % 10 == 0 and i > 0:
                print(f"    Tokenized {i:,}/{len(lines):,} lines ({len(all_ids):,} tokens)...")
    else:
        for line in lines:
            ids = tok.encode(line)
            if ids:
                all_ids.extend(ids)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, np.array(all_ids, dtype=np.int32))
        print(f"  Tokens cached: {cache_path} ({len(all_ids):,} tokens)")

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
    use_ckpt = cfg.get('use_checkpoint', True)
    local_window = int(os.environ.get('LOCAL_WINDOW', '0'))
    n_frozen_heads = int(os.environ.get('FROZEN_HEADS', '0'))
    model = WaveFieldTransformer(
        vocab_size=vocab_size,
        embedding_dim=cfg['embedding_dim'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        ffn_dim=cfg['ffn_dim'],
        field_size=cfg['field_size'],
        max_seq_len=cfg['seq_len'] + 2,
        dropout=0.1,
        use_checkpoint=use_ckpt,
        interference_interval=3,
        n_components=1,
        local_window=local_window,
        n_frozen_heads=n_frozen_heads,
        device=device,
    ).to(device)
    if local_window > 0 or n_frozen_heads > 0:
        print(f"  Wave config: local_window={local_window}, "
              f"n_frozen_heads={n_frozen_heads}")
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
        # fp32 cross-entropy: bf16 logits can overflow at high confidence
        loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), y.reshape(-1))
        lv = loss.item()
        if not math.isnan(lv):
            total_loss += lv
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
              use_amp=True, scale_key='', model_type='', seed=42,
              resume_path=None):
    """Train a model and return results dict with training curve."""
    # Per-run seed + non-deterministic cuDNN for speed (+5-10%)
    set_seed(seed)  # deterministic=True: cudnn.benchmark=False avoids stall on variable batch sizes

    params = count_params(model)
    tokens_per_step = batch_size * seq_len
    total_steps = total_token_budget // tokens_per_step

    print(f"\n  --- {run_name} ---")
    print(f"  Params: {params:,} | Context: {seq_len} | Batch: {batch_size}")
    print(f"  Token budget: {total_token_budget:,} | Steps: {total_steps:,}")
    print(f"  Train chunks: {len(train_data):,} | Val chunks: {len(val_data):,}")
    print(f"  Seed: {seed}")

    # wandb run
    wb_run = None
    if _wandb_available:
        try:
            wb_run = wandb.init(
                project="wave-field-llm",
                name=run_name,
                config={
                    'model_type': model_type,
                    'scale': scale_key,
                    'params': params,
                    'seq_len': seq_len,
                    'batch_size': batch_size,
                    'token_budget': total_token_budget,
                    'peak_lr': peak_lr,
                    'seed': seed,
                },
                reinit=True,
                resume='allow' if resume_path else None,
            )
        except Exception as e:
            print(f"  wandb init failed: {e}")
            wb_run = None

    n_params = sum(p.numel() for p in model.parameters())
    # torch.compile: only for large models (>=50M) WITHOUT gradient checkpointing
    # (compiled submodules inside checkpointed layers cause recompilation storms/crashes)
    use_ckpt = getattr(model, 'use_checkpoint', False)
    if torch.cuda.is_available() and n_params >= 50_000_000 and hasattr(model, 'compile_model') and not use_ckpt:
        try:
            model.compile_model(mode='default')
            print("  torch.compile: enabled (default mode)")
        except Exception as e:
            print(f"  torch.compile: skipped ({e})")
    else:
        reason = f"{n_params/1e6:.0f}M params < 50M" if n_params < 50_000_000 else "checkpoint=True (incompatible)"
        print(f"  torch.compile: skipped ({reason})")

    # V4.3.2: Use per-group LR optimizer if available (kernel params at 50x LR)
    if hasattr(model, 'configure_optimizer'):
        optimizer = model.configure_optimizer(base_lr=peak_lr, kernel_lr_mult=50.0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01, eps=1e-8)
    warmup = max(total_steps // 10, 100)
    scheduler = WarmupCosineScheduler(optimizer, warmup, total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Training monitor (lightweight diagnostics)
    monitor = None
    if _monitor_available:
        try:
            safe_name = run_name.replace(' ', '_').replace('/', '-').lower()
            monitor = WaveFieldMonitor(model, log_dir=f'results/monitor/{safe_name}')
        except Exception:
            pass

    best_val_loss = float('inf')
    best_ppl = float('inf')
    best_acc = 0
    tokens_seen = 0
    step = 0
    epoch = 0
    curve = []  # training curve data points

    # Resume from checkpoint if available
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    safe_name = run_name.replace(' ', '_').lower()

    if resume_path and os.path.exists(resume_path):
        print(f"  Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        scheduler.step_count = ckpt['scheduler_step']
        step = ckpt['step']
        epoch = ckpt['epoch']
        tokens_seen = ckpt['tokens_seen']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        best_ppl = ckpt.get('best_ppl', float('inf'))
        best_acc = ckpt.get('best_acc', 0)
        curve = ckpt.get('curve', [])
        print(f"  Resumed: step={step}, tokens={tokens_seen/1e6:.1f}M, best_ppl={best_ppl:.1f}")

    t0 = time.time()
    eval_interval = max(total_steps // 20, 25)

    if step == 0:
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

            # Monitor: lightweight per-step logging
            if monitor:
                lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else peak_lr
                monitor.step(step, loss.item(), lr=lr_now)

            if step % eval_interval == 0 or tokens_seen >= total_token_budget:
                # Monitor: full snapshot (gradients still alive before next zero_grad)
                # Save incrementally so data survives crashes/stops
                if monitor:
                    monitor.snapshot(step, sample_input=x[:2])
                    try:
                        monitor.save_report()
                    except Exception:
                        pass
                vl, vp, va = evaluate(model, val_data, batch_size, vocab_size, device, use_amp)
                elapsed = time.time() - t0
                tps = tokens_seen / elapsed
                mark = ""
                if vl < best_val_loss:
                    best_val_loss = vl
                    best_ppl = vp
                    best_acc = va
                    mark = " *BEST"
                    # Save best weights (lightweight, for inference)
                    best_path = os.path.join(ckpt_dir, f'{safe_name}.pt')
                    torch.save(model.state_dict(), best_path)
                # Save resumable checkpoint (full training state) every eval
                resume_ckpt_path = os.path.join(ckpt_dir, f'{safe_name}_resume.pt')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'scheduler_step': scheduler.step_count,
                    'step': step,
                    'epoch': epoch,
                    'tokens_seen': tokens_seen,
                    'best_val_loss': best_val_loss,
                    'best_ppl': best_ppl,
                    'best_acc': best_acc,
                    'curve': curve,
                    'seed': seed,
                }, resume_ckpt_path)
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
                # wandb logging
                if wb_run:
                    wandb.log({
                        'val/ppl': vp,
                        'val/acc': va,
                        'val/loss': vl,
                        'train/tokens_M': tokens_seen / 1e6,
                        'train/tok_per_sec': tps,
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'step': step,
                    })

    total_time = time.time() - t0
    final_tps = tokens_seen / total_time

    # Save monitor report
    if monitor:
        try:
            monitor.save_report()
        except Exception as e:
            print(f"    [Monitor] save_report failed: {e}")

    # Close wandb run
    if wb_run:
        wandb.log({'final/best_ppl': best_ppl, 'final/best_acc': best_acc,
                   'final/tokens_per_sec': round(final_tps)})
        wandb.finish()

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
        'seed': seed,
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

    # Reproducibility
    seed = int(os.environ.get('SEED', '42'))
    set_seed(seed)
    print(f"\n  Seed: {seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"  Device: {device}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {vram_gb:.1f} GB")
    print(f"  wandb: {'enabled' if _wandb_available else 'disabled'}")

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

    # Resume support: RESUME=1 to auto-detect, RESUME=path for explicit
    resume_env = os.environ.get('RESUME', '').strip()
    resume_enabled = resume_env not in ('', '0', 'false')

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    data_dir = os.path.join(results_dir, 'data')
    plots_dir = os.path.join(results_dir, 'plots')
    ckpts_dir = os.path.join(results_dir, 'checkpoints')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(ckpts_dir, exist_ok=True)

    print(f"\n  Scales: {scale_keys}")
    print(f"  Models: {'SPECTRE-Wave' if run_wave else ''} {'Standard' if run_std else ''}")
    if resume_enabled:
        print(f"  Resume: enabled")

    # Load data + tokenizer
    splits, dataset_name = load_dataset_splits()
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'cache')
    print(f"\n  Training BPE tokenizer (8K vocab)...")
    raw_tok = train_bpe_tokenizer(splits['train'], vocab_size=8000, cache_dir=cache_dir)
    tok = BPEWrapper(raw_tok)
    vocab_size = tok.vocab_size_actual()
    print(f"  Vocab: {vocab_size}")

    print(f"  Tokenizing corpus...")
    ds_tag = 'wt103' if 'WikiText-103' in dataset_name else 'wt2'
    train_ids = tokenize_corpus(splits['train'], tok,
                                cache_path=os.path.join(cache_dir, f'{ds_tag}_train.npy'))
    val_ids = tokenize_corpus(splits['valid'], tok,
                              cache_path=os.path.join(cache_dir, f'{ds_tag}_val.npy'))
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
        'seed': seed,
    }

    for scale_key in scale_keys:
        if scale_key not in SCALE_CONFIGS:
            print(f"\n  Unknown scale: {scale_key}, skipping")
            continue

        cfg = SCALE_CONFIGS[scale_key]
        seq_len = cfg['seq_len']
        batch_size = cfg['batch_size']

        # Batch size override via env var
        batch_override = os.environ.get('BATCH_SIZE', '').strip()
        if batch_override:
            batch_size = int(batch_override)

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

                # Check for resumable checkpoint
                wave_resume = None
                if resume_enabled:
                    wave_safe = f"spectre-wave_{scale_key.lower()}_resume.pt"
                    wave_resume_path = os.path.join(ckpts_dir, wave_safe)
                    if os.path.exists(wave_resume_path):
                        wave_resume = wave_resume_path
                    elif resume_env not in ('1', 'true') and os.path.exists(resume_env):
                        wave_resume = resume_env

                result = train_run(
                    model, train_data, val_data, vocab_size, device,
                    f"SPECTRE-Wave {scale_key}",
                    cfg['token_budget'], seq_len, batch_size,
                    cfg['peak_lr'], use_amp,
                    scale_key=scale_key, model_type='wave', seed=seed,
                    resume_path=wave_resume,
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

                # Check for resumable checkpoint
                std_resume = None
                if resume_enabled:
                    std_safe = f"standard_{scale_key.lower()}_resume.pt"
                    std_resume_path = os.path.join(ckpts_dir, std_safe)
                    if os.path.exists(std_resume_path):
                        std_resume = std_resume_path

                result = train_run(
                    model, train_data, val_data, vocab_size, device,
                    f"Standard {scale_key}",
                    cfg['token_budget'], seq_len, batch_size,
                    cfg['peak_lr'], use_amp,
                    scale_key=scale_key, model_type='standard', seed=seed,
                    resume_path=std_resume,
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
    output = {
        'metadata': run_metadata,
        'results': all_results,
    }
    results_path = os.path.join(data_dir, 'scaling_benchmark.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # Also save per-scale for incremental runs
    for scale_key in scale_keys:
        scale_results = [r for r in all_results if r.get('scale') == scale_key]
        if scale_results:
            scale_path = os.path.join(data_dir, f'scaling_{scale_key.lower()}.json')
            with open(scale_path, 'w') as f:
                json.dump({'metadata': run_metadata, 'results': scale_results}, f, indent=2)
            print(f"  Scale {scale_key} saved: {scale_path}")

    # ============================================================
    # VISUALIZATIONS
    # ============================================================
    plot_results(all_results, scale_keys, plots_dir, dataset_name)

    # ============================================================
    # INFERENCE TEST (if wave checkpoint exists)
    # ============================================================
    run_inference_test(all_results, scale_keys, vocab_size, tok, device, ckpts_dir)


# ======================================================================
# VISUALIZATION
# ======================================================================

def plot_results(all_results, scale_keys, results_dir, dataset_name):
    """Generate training curve plots and comparison charts."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    # --- Plot 1: Training Curves (PPL over tokens) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PPL curves
    ax = axes[0]
    for r in all_results:
        curve = r.get('curve', [])
        if not curve:
            continue
        tokens = [p['tokens_M'] for p in curve]
        ppls = [p['ppl'] for p in curve]
        label = r['run_name']
        style = '-' if 'SPECTRE' in label else '--'
        color = '#2196F3' if 'SPECTRE' in label else '#FF5722'
        ax.plot(tokens, ppls, style, label=label, color=color, linewidth=2)
    ax.set_xlabel('Tokens (M)', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title(f'Training Curves — PPL ({dataset_name})', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Accuracy curves
    ax = axes[1]
    for r in all_results:
        curve = r.get('curve', [])
        if not curve:
            continue
        tokens = [p['tokens_M'] for p in curve]
        accs = [p['acc'] for p in curve]
        label = r['run_name']
        style = '-' if 'SPECTRE' in label else '--'
        color = '#2196F3' if 'SPECTRE' in label else '#FF5722'
        ax.plot(tokens, accs, style, label=label, color=color, linewidth=2)
    ax.set_xlabel('Tokens (M)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Training Curves — Accuracy ({dataset_name})', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {path}")

    # --- Plot 2: Final Results Bar Chart ---
    wave_results = [r for r in all_results if r.get('model_type') == 'wave'
                    and isinstance(r.get('best_ppl'), (int, float))]
    std_results = [r for r in all_results if r.get('model_type') == 'standard'
                   and isinstance(r.get('best_ppl'), (int, float))]

    if wave_results or std_results:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # PPL comparison
        ax = axes[0]
        labels, wave_ppls, std_ppls = [], [], []
        for sk in scale_keys:
            w = next((r for r in wave_results if r.get('scale') == sk), None)
            s = next((r for r in std_results if r.get('scale') == sk), None)
            if w or s:
                labels.append(sk)
                wave_ppls.append(w['best_ppl'] if w else 0)
                std_ppls.append(s['best_ppl'] if s else 0)
        x = range(len(labels))
        if wave_ppls:
            ax.bar([i - 0.2 for i in x], wave_ppls, 0.35, label='SPECTRE-Wave', color='#2196F3')
        if std_ppls:
            ax.bar([i + 0.2 for i in x], std_ppls, 0.35, label='Standard', color='#FF5722')
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Perplexity (lower = better)')
        ax.set_title('Best PPL by Scale')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Accuracy comparison
        ax = axes[1]
        wave_accs, std_accs = [], []
        for sk in scale_keys:
            w = next((r for r in wave_results if r.get('scale') == sk), None)
            s = next((r for r in std_results if r.get('scale') == sk), None)
            wave_accs.append(w['best_acc'] if w else 0)
            std_accs.append(s['best_acc'] if s else 0)
        if wave_accs:
            ax.bar([i - 0.2 for i in x], wave_accs, 0.35, label='SPECTRE-Wave', color='#2196F3')
        if std_accs:
            ax.bar([i + 0.2 for i in x], std_accs, 0.35, label='Standard', color='#FF5722')
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Best Accuracy by Scale')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Throughput comparison
        ax = axes[2]
        wave_tps, std_tps = [], []
        for sk in scale_keys:
            w = next((r for r in wave_results if r.get('scale') == sk), None)
            s = next((r for r in std_results if r.get('scale') == sk), None)
            wave_tps.append(w.get('tokens_per_sec', 0) if w else 0)
            std_tps.append(s.get('tokens_per_sec', 0) if s else 0)
        if wave_tps:
            ax.bar([i - 0.2 for i in x], wave_tps, 0.35, label='SPECTRE-Wave', color='#2196F3')
        if std_tps:
            ax.bar([i + 0.2 for i in x], std_tps, 0.35, label='Standard', color='#FF5722')
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Tokens/sec')
        ax.set_title('Throughput by Scale')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        path = os.path.join(results_dir, 'scaling_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved: {path}")

    # --- Plot 3: Step-by-step monitoring (PPL evolution per eval) ---
    for r in all_results:
        curve = r.get('curve', [])
        if len(curve) < 3:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        steps = [p['step'] for p in curve]
        ppls = [p['ppl'] for p in curve]
        accs = [p['acc'] for p in curve]

        color = '#2196F3' if 'SPECTRE' in r['run_name'] else '#FF5722'
        ax.plot(steps, ppls, '-o', color=color, markersize=4, label='PPL')
        ax.set_xlabel('Step')
        ax.set_ylabel('PPL', color=color)
        ax.set_yscale('log')
        ax.tick_params(axis='y', labelcolor=color)

        ax2 = ax.twinx()
        ax2.plot(steps, accs, '-s', color='#4CAF50', markersize=4, alpha=0.7, label='Acc')
        ax2.set_ylabel('Accuracy (%)', color='#4CAF50')
        ax2.tick_params(axis='y', labelcolor='#4CAF50')

        ax.set_title(f'{r["run_name"]} — Step Monitor ({dataset_name})')
        ax.grid(True, alpha=0.2)

        safe = r['run_name'].replace(' ', '_').lower()
        path = os.path.join(results_dir, f'monitor_{safe}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Monitor plot saved: {path}")


# ======================================================================
# INFERENCE TEST
# ======================================================================

def run_inference_test(all_results, scale_keys, vocab_size, tok, device, results_dir):
    """Load best wave checkpoint and generate sample text."""
    import torch.nn.functional as F

    # Find the wave checkpoint
    ckpt_path = None
    for r in all_results:
        if r.get('model_type') == 'wave' and isinstance(r.get('best_ppl'), (int, float)):
            safe_name = r['run_name'].replace(' ', '_').lower()
            candidate = os.path.join(results_dir, f'{safe_name}.pt')
            if os.path.exists(candidate):
                ckpt_path = candidate
                scale_key = r['scale']
                break

    if not ckpt_path:
        print("\n  No wave checkpoint found, skipping inference test")
        return

    print(f"\n{'='*72}")
    print(f"  INFERENCE TEST — {ckpt_path}")
    print(f"{'='*72}")

    cfg = SCALE_CONFIGS[scale_key]
    model = WaveFieldTransformer(
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
        local_window=0,
        device=device,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded {sum(p.numel() for p in model.parameters()):,} params")

    prompts = [
        "The history of",
        "In recent years, scientists have",
        "The city of London",
        "During the war,",
        "The president announced",
    ]

    for prompt in prompts:
        ids = tok.encode(prompt)
        input_ids = torch.tensor([ids], device=device)
        generated = list(ids)

        with torch.no_grad():
            for _ in range(80):
                if input_ids.shape[1] > cfg['seq_len']:
                    input_ids = input_ids[:, -cfg['seq_len']:]
                logits, _ = model(input_ids)
                next_logits = logits[0, -1, :] / 0.8
                # Top-k
                topk_vals, _ = torch.topk(next_logits, 40)
                next_logits[next_logits < topk_vals[-1]] = float('-inf')
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        output = tok.decode(generated)
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  Output: {output[:300]}")
        print(f"  {'─' * 60}")

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
