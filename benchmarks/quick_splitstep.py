"""
Minimal Split-Step benchmark — no checkpoint saves, no monitor, just train and eval.
Prints training loss every 50 steps and eval PPL every 500 steps.
"""
import torch
import torch.nn.functional as F
import time
import math
import os
import sys
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.wave_field_transformer import WaveFieldTransformer

# ── Config ──────────────────────────────────────────
BATCH = int(os.environ.get('BATCH_SIZE', '8'))
SEQ = 512
TOKEN_BUDGET = int(os.environ.get('TOKEN_BUDGET', '10000000'))
EMBED, LAYERS, HEADS, FFN = 384, 8, 8, 1536  # S1
FIELD_SIZE = 512
PEAK_LR = 3e-4
SEED = 42

torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Split-Step Quick Benchmark")
print(f"  batch={BATCH}, seq={SEQ}, budget={TOKEN_BUDGET/1e6:.0f}M tokens")
print(f"  Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Data ────────────────────────────────────────────
print("\n  Loading data...")
cache_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'cache')
train_path = os.path.join(cache_dir, 'wt2_train.npy')
val_path = os.path.join(cache_dir, 'wt2_val.npy')

import numpy as np
if os.path.exists(train_path):
    train_ids = np.load(train_path)
    val_ids = np.load(val_path)
    print(f"  Loaded cached: train={len(train_ids):,}, val={len(val_ids):,}")
else:
    print("  ERROR: cached tokens not found. Run benchmark_scaling.py first.")
    sys.exit(1)

VOCAB = 8000

def make_chunks(ids, seq_len):
    n = len(ids) // (seq_len + 1)
    chunks = []
    for i in range(n):
        chunk = ids[i * (seq_len + 1):(i + 1) * (seq_len + 1)]
        chunks.append(torch.tensor(chunk, dtype=torch.long))
    return chunks

train_data = make_chunks(train_ids, SEQ)
val_data = make_chunks(val_ids, SEQ)
print(f"  Chunks: train={len(train_data)}, val={len(val_data)}")

# ── Model ───────────────────────────────────────────
print("\n  Building model...")
model = WaveFieldTransformer(
    vocab_size=VOCAB, embedding_dim=EMBED, num_layers=LAYERS,
    num_heads=HEADS, ffn_dim=FFN, field_size=FIELD_SIZE,
    max_seq_len=SEQ + 2, dropout=0.1, use_checkpoint=True,
    interference_interval=3, n_components=1,
    local_window=0, n_frozen_heads=4,
    use_split_step=True, device=device,
).to(device)

params = sum(p.numel() for p in model.parameters())
print(f"  Params: {params:,}")

# ── Optimizer ───────────────────────────────────────
optimizer = model.configure_optimizer(base_lr=PEAK_LR, weight_decay=0.01)
print(f"  Optimizer: {len(optimizer.param_groups)} param groups")

use_amp = torch.cuda.is_available()
scaler = torch.amp.GradScaler('cuda', enabled=use_amp and not torch.cuda.is_bf16_supported())

# ── Training ────────────────────────────────────────
tokens_per_step = BATCH * SEQ
total_steps = TOKEN_BUDGET // tokens_per_step
eval_interval = max(total_steps // 10, 50)

print(f"\n  Steps: {total_steps}, eval every {eval_interval}")
print(f"  VRAM after model: {torch.cuda.memory_allocated()/1e6:.0f} MB")
print()

model.train()
step = 0
tokens_seen = 0
best_ppl = float('inf')
t0 = time.time()

import random
random.seed(SEED)

def create_batches(data, batch_size, shuffle=True):
    """Create proper batches from different chunks (not duplicated!)."""
    indices = list(range(len(data)))
    if shuffle:
        random.shuffle(indices)
    batches = []
    for i in range(0, len(indices) - batch_size + 1, batch_size):
        batch_idx = indices[i:i + batch_size]
        x_list, y_list = [], []
        for idx in batch_idx:
            chunk = data[idx]
            x_list.append(chunk[:-1])
            y_list.append(chunk[1:])
        batches.append((torch.stack(x_list), torch.stack(y_list)))
    return batches

while tokens_seen < TOKEN_BUDGET:
    batches = create_batches(train_data, BATCH, shuffle=True)
    for x, y in batches:
        if tokens_seen >= TOKEN_BUDGET:
            break

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, _ = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), y.reshape(-1))

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        tokens_seen += tokens_per_step
        step += 1

        # Print training loss
        if step % 50 == 0:
            elapsed = time.time() - t0
            tps = tokens_seen / elapsed
            vram = torch.cuda.memory_allocated() / 1e6
            print(f"  [{step:>5}/{total_steps}] loss={loss.item():.3f} | "
                  f"{tps:,.0f} tok/s | {vram:.0f}MB | {elapsed:.0f}s", flush=True)

        # Eval
        if step % eval_interval == 0:
            model.eval()
            total_loss, n = 0, 0
            with torch.no_grad():
                for vc in val_data[:100]:  # Only first 100 val chunks for speed
                    vc = vc.to(device)
                    vx, vy = vc[:-1].unsqueeze(0), vc[1:].unsqueeze(0)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        vlogits, _ = model(vx)
                    vl = F.cross_entropy(vlogits.float().reshape(-1, VOCAB), vy.reshape(-1))
                    if not math.isnan(vl.item()):
                        total_loss += vl.item()
                        n += 1
            model.train()
            del vc, vx, vy, vlogits, vl
            gc.collect()
            torch.cuda.empty_cache()

            avg_loss = total_loss / max(n, 1)
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t0
            tps = tokens_seen / elapsed
            mark = ""
            if ppl < best_ppl:
                best_ppl = ppl
                mark = " *BEST"
            print(f"  *** EVAL step {step} | PPL {ppl:.1f} | "
                  f"{tokens_seen/1e6:.1f}M tok | {tps:,.0f} tok/s | {elapsed:.0f}s{mark}",
                  flush=True)

total_time = time.time() - t0
print(f"\n{'='*60}")
print(f"  DONE: {tokens_seen/1e6:.1f}M tokens in {total_time:.0f}s")
print(f"  Best PPL: {best_ppl:.1f}")
print(f"  Avg tok/s: {tokens_seen/total_time:,.0f}")
print(f"{'='*60}")
