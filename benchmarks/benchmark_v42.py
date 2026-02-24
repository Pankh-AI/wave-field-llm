"""
V4.2 Data Efficiency Benchmark
================================
Tests three fixes for the ~3M token stuck phase in V4.1 linear-wave attention.

Root cause: elu(0)+1 = 1 everywhere at init → zero routing for ~3M tokens.
Additional bug: _init_weights() overwrites gate bias from 2.0 to 0.0.

Fixes (grounded in literature):
  Fix 1: Gate init bug — restore bias=2.0 (sigmoid=0.88, gates start open)
  Fix 2: Q/K bias diversity — per-head linspace biases break feature map symmetry
  Fix 3: Elevated QK LR — 3x LR for qkvg_proj accelerates feature map learning

6 configs at 5M tokens each:
  A) V4.1 Baseline (no fixes)       — expect PPL ~1188
  B) Gate Fix Only                   — isolate gate bug impact
  C) Gate + QK Bias                  — init diversity
  D) Gate + Elevated LR              — optimizer fix
  E) V4.2 Full (Gate+Bias+LR)       — all three
  F) Standard Transformer            — reference PPL ~459

References:
  - Plateau theory: arXiv:2501.16265 (escape time O(ln(1/w_init)))
  - Hedgehog (ICLR 2024): identity-init feature maps
  - GLA (ICML 2024): separate param groups for gate vs content
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
# STANDARD TRANSFORMER (baseline)
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
# MODEL CREATION WITH OPTIONAL FIXES
# ======================================================================

def create_wave_model(vocab_size, embedding_dim, num_layers, num_heads,
                      ffn_dim, field_size, seq_len, device,
                      fix_gate_init=False, qk_bias_diversity=False):
    """Create WaveFieldTransformer with optional V4.2 init fixes.

    The model's _init_weights() now includes both fixes by default (V4.2).
    To test WITHOUT fixes (V4.1 baseline), we undo them after construction.
    """
    model = WaveFieldTransformer(
        vocab_size=vocab_size, embedding_dim=embedding_dim,
        num_layers=num_layers, num_heads=num_heads,
        ffn_dim=ffn_dim, field_size=field_size,
        max_seq_len=seq_len + 2, dropout=0.1,
        use_checkpoint=True, interference_interval=3,
        n_components=1, local_window=0, device=device,
    ).to(device)

    # _init_weights() now applies both fixes by default.
    # If a config DOESN'T want a fix, undo it here.
    D = embedding_dim
    H = num_heads
    head_dim = D // H

    if not fix_gate_init:
        # Undo fix 1: reset gate to V4.1 behavior (bias=0, random weights)
        for layer in model.layers:
            attn = layer.attention
            with torch.no_grad():
                torch.nn.init.normal_(attn.qkvg_proj.weight[3 * D:], mean=0.0, std=0.02)
                attn.qkvg_proj.bias[3 * D:].zero_()

    if not qk_bias_diversity:
        # Undo fix 2: reset Q/K biases to zero (V4.1 behavior)
        for layer in model.layers:
            attn = layer.attention
            with torch.no_grad():
                attn.qkvg_proj.bias[:2 * D].zero_()

    return model


# ======================================================================
# TRAINING
# ======================================================================

class WarmupCosineScheduler:
    """Warmup + cosine decay, supports multiple param groups with different base LRs."""

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
                # Scale min_lr proportionally to base_lr
                min_lr_scaled = self.min_lr * (base_lr / self.base_lrs[0]) if self.base_lrs[0] > 0 else self.min_lr
                lr = min_lr_scaled + 0.5 * (base_lr - min_lr_scaled) * (1 + math.cos(math.pi * p))
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
              use_amp=True, qk_lr_mult=1.0):
    params = sum(p.numel() for p in model.parameters())
    tokens_per_step = batch_size * seq_len
    total_steps = total_token_budget // tokens_per_step

    print(f"\n  --- {run_name} ---")
    print(f"  Params: {params:,} | Context: {seq_len} | Batch: {batch_size}")
    print(f"  Token budget: {total_token_budget:,} | Steps: {total_steps:,}")
    print(f"  Train chunks: {len(train_data):,} | Val chunks: {len(val_data):,}")
    if qk_lr_mult != 1.0:
        print(f"  QK LR mult: {qk_lr_mult}x (qkvg_proj at {peak_lr * qk_lr_mult:.6f})")

    # Build optimizer — optionally with separate LR for qkvg_proj
    if qk_lr_mult != 1.0 and hasattr(model, 'layers'):
        qk_param_ids = set()
        qk_params = []
        for layer in model.layers:
            attn = layer.attention
            qk_params.append(attn.qkvg_proj.weight)
            qk_params.append(attn.qkvg_proj.bias)
            qk_param_ids.add(id(attn.qkvg_proj.weight))
            qk_param_ids.add(id(attn.qkvg_proj.bias))

        other_params = [p for p in model.parameters() if id(p) not in qk_param_ids]

        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': peak_lr},
            {'params': qk_params, 'lr': peak_lr * qk_lr_mult},
        ], weight_decay=0.01, eps=1e-8)
    else:
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
    eval_interval = max(total_steps // 20, 25)  # 20 eval points for finer curve

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
    print("  V4.2 DATA EFFICIENCY BENCHMARK")
    print("  Gate fix + QK bias diversity + elevated QK LR")
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

    # 6-config experiment matrix
    configs = [
        {
            'name': 'A) V4.1 Baseline (no fixes)',
            'type': 'wave',
            'fix_gate': False,
            'qk_bias': False,
            'qk_lr_mult': 1.0,
        },
        {
            'name': 'B) Gate Fix Only',
            'type': 'wave',
            'fix_gate': True,
            'qk_bias': False,
            'qk_lr_mult': 1.0,
        },
        {
            'name': 'C) Gate + QK Bias',
            'type': 'wave',
            'fix_gate': True,
            'qk_bias': True,
            'qk_lr_mult': 1.0,
        },
        {
            'name': 'D) Gate + Elevated LR',
            'type': 'wave',
            'fix_gate': True,
            'qk_bias': False,
            'qk_lr_mult': 3.0,
        },
        {
            'name': 'E) V4.2 Full (Gate+Bias+LR)',
            'type': 'wave',
            'fix_gate': True,
            'qk_bias': True,
            'qk_lr_mult': 3.0,
        },
        {
            'name': 'F) Standard Transformer',
            'type': 'standard',
        },
    ]

    all_results = []

    for cfg in configs:
        if cfg['type'] == 'standard':
            model = StandardTransformer(
                vocab_size=vocab_size, embedding_dim=embedding_dim,
                num_layers=num_layers, num_heads=num_heads,
                ffn_dim=ffn_dim, max_seq_len=seq_len + 2, dropout=0.1,
            ).to(device)
            qk_lr_mult = 1.0
        else:
            model = create_wave_model(
                vocab_size=vocab_size, embedding_dim=embedding_dim,
                num_layers=num_layers, num_heads=num_heads,
                ffn_dim=ffn_dim, field_size=field_size,
                seq_len=seq_len, device=device,
                fix_gate_init=cfg.get('fix_gate', False),
                qk_bias_diversity=cfg.get('qk_bias', False),
            )
            qk_lr_mult = cfg.get('qk_lr_mult', 1.0)

        try:
            result = train_run(
                model, train_data, val_data, vocab_size, device, cfg['name'],
                total_token_budget, seq_len, batch_size, peak_lr, use_amp,
                qk_lr_mult=qk_lr_mult,
            )
            if cfg['type'] == 'wave':
                result['fix_gate'] = cfg.get('fix_gate', False)
                result['qk_bias'] = cfg.get('qk_bias', False)
                result['qk_lr_mult'] = qk_lr_mult
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
    print("  RESULTS: V4.2 DATA EFFICIENCY BENCHMARK")
    print(f"{'='*72}")
    print(f"\n  {'Config':<42} {'PPL':>8} {'Acc':>7} {'tok/s':>10}")
    print(f"  {'-'*42} {'-'*8} {'-'*7} {'-'*10}")

    for r in all_results:
        ppl = r.get('best_ppl', 'N/A')
        acc = r.get('best_acc', 'N/A')
        tps = r.get('tokens_per_sec', 'N/A')
        ppl_s = f"{ppl:>8.1f}" if isinstance(ppl, (int, float)) else f"{ppl:>8}"
        acc_s = f"{acc:>6.1f}%" if isinstance(acc, (int, float)) else f"{acc:>7}"
        tps_s = f"{tps:>10,}" if isinstance(tps, (int, float)) else f"{tps:>10}"
        print(f"  {r['run_name']:<42} {ppl_s} {acc_s} {tps_s}")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'v42_benchmark.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # Key comparison
    print(f"\n  --- KEY COMPARISON ---")
    std = next((r for r in all_results if 'Standard' in r.get('run_name', '')), None)
    baseline = next((r for r in all_results if 'Baseline' in r.get('run_name', '')), None)
    v42 = next((r for r in all_results if 'V4.2 Full' in r.get('run_name', '')), None)

    if baseline and isinstance(baseline.get('best_ppl'), (int, float)):
        print(f"  V4.1 Baseline (5M):  PPL {baseline['best_ppl']:.1f}, Acc {baseline['best_acc']:.1f}%")
    if v42 and isinstance(v42.get('best_ppl'), (int, float)):
        print(f"  V4.2 Full (5M):      PPL {v42['best_ppl']:.1f}, Acc {v42['best_acc']:.1f}%")
        if baseline and isinstance(baseline.get('best_ppl'), (int, float)):
            improvement = baseline['best_ppl'] - v42['best_ppl']
            print(f"  Improvement:         {improvement:+.1f} PPL ({improvement/baseline['best_ppl']*100:.1f}% reduction)")
    if std and isinstance(std.get('best_ppl'), (int, float)):
        print(f"  Standard (5M):       PPL {std['best_ppl']:.1f}, Acc {std['best_acc']:.1f}%")
        if v42 and isinstance(v42.get('best_ppl'), (int, float)):
            gap = (v42['best_ppl'] - std['best_ppl']) / std['best_ppl'] * 100
            print(f"  V4.2 vs Standard:    {gap:+.1f}% gap")
    print(f"  V4.1 at 15M tokens:  PPL 543, Acc 10.6% (reference)")


if __name__ == '__main__':
    main()
