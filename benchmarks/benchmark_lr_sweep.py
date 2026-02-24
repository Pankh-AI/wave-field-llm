"""
QK LR Multiplier Sweep
======================
Finds optimal QK LR multiplier for qkvg_proj parameters.
All configs: V4.3 best (analytic kernel, 2L FM, no write gate, no 3D).

  A) QK LR x2
  B) QK LR x3  (current default)
  C) QK LR x4
  D) QK LR x5
  E) Standard Transformer reference
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


class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, num_layers=6,
                 num_heads=8, ffn_dim=1024, max_seq_len=4096, dropout=0.1):
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

    splits = load_wikitext2()
    print("  Training BPE tokenizer...")
    raw_tok = train_bpe_tokenizer(splits['train'], vocab_size=vocab_size)
    tok = BPEWrapper(raw_tok)
    train_ids = tokenize_corpus(splits['train'], tok)
    val_ids = tokenize_corpus(splits['valid'], tok)
    print(f"  Train: {len(train_ids):,} | Val: {len(val_ids):,} tokens")

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
        bx = torch.stack([data[i][0] for i in batch_idx]).to(device)
        by = torch.stack([data[i][1] for i in batch_idx]).to(device)
        batches.append((bx, by))
    return batches


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


# ======================================================================
# CONFIG
# ======================================================================

ARCH = {
    'embedding_dim': 384, 'num_layers': 8, 'num_heads': 8,
    'ffn_dim': 1536, 'field_size': 2048,
    'seq_len': 512, 'batch_size': 16,
    'token_budget': 5_000_000, 'peak_lr': 3e-4,
}

CONFIGS = [
    {'key': 'A', 'name': 'A) QK LR x2', 'type': 'wave', 'qk_lr_mult': 2.0},
    {'key': 'B', 'name': 'B) QK LR x3 (current)', 'type': 'wave', 'qk_lr_mult': 3.0},
    {'key': 'C', 'name': 'C) QK LR x4', 'type': 'wave', 'qk_lr_mult': 4.0},
    {'key': 'D', 'name': 'D) QK LR x5', 'type': 'wave', 'qk_lr_mult': 5.0},
    {'key': 'E', 'name': 'E) Standard Transformer', 'type': 'standard'},
]


# ======================================================================
# TRAIN
# ======================================================================

def train_run(model, train_data, val_data, vocab_size, device, run_name,
              use_amp, qk_lr_mult=3.0):
    seq_len = ARCH['seq_len']
    batch_size = ARCH['batch_size']
    token_budget = ARCH['token_budget']
    tokens_per_step = batch_size * seq_len
    total_steps = token_budget // tokens_per_step
    params = sum(p.numel() for p in model.parameters())
    peak_lr = ARCH['peak_lr']

    print(f"\n  --- {run_name} ---")
    print(f"  Params: {params:,} | Steps: {total_steps:,} | QK mult: {qk_lr_mult}", flush=True)

    if hasattr(model, 'configure_optimizer'):
        optimizer = model.configure_optimizer(base_lr=peak_lr, qk_lr_mult=qk_lr_mult)
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

    t0 = time.time()
    eval_interval = max(total_steps // 10, 10)

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
                tps = tokens_seen / (time.time() - t0)
                print(f"    Step {step:>4}/{total_steps} | "
                      f"Tok {tokens_seen/1e6:.1f}M | "
                      f"PPL {vp:>7.1f} Acc {va:>5.1f}% | "
                      f"{tps:,.0f} tok/s{mark}", flush=True)
                curve.append({
                    'tokens_M': round(tokens_seen / 1e6, 2),
                    'ppl': round(vp, 2),
                    'acc': round(va, 2),
                })

    return {
        'run_name': run_name,
        'params': params,
        'best_ppl': round(best_ppl, 2),
        'best_acc': round(best_acc, 2),
        'tokens_seen': tokens_seen,
        'time_s': round(time.time() - t0, 1),
        'curve': curve,
    }


def main():
    print("=" * 65)
    print("  QK LR MULTIPLIER SWEEP")
    print("  V4.3 best config, varying qkvg_proj LR multiplier")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    tok, train_ids, val_ids = load_cached_data(vocab_size=8000)
    vocab_size = tok.vocab_size_actual()

    train_data = make_chunks(train_ids, ARCH['seq_len'])
    val_data = make_chunks(val_ids, ARCH['seq_len'])

    config_filter = os.environ.get('CONFIGS', '').strip().upper()
    if config_filter:
        keys = [k.strip() for k in config_filter.split(',')]
        run_configs = [c for c in CONFIGS if c['key'] in keys]
    else:
        run_configs = CONFIGS

    all_results = []

    for cfg in run_configs:
        torch.cuda.empty_cache()
        gc.collect()

        if cfg['type'] == 'standard':
            model = StandardTransformer(
                vocab_size=vocab_size,
                embedding_dim=ARCH['embedding_dim'],
                num_layers=ARCH['num_layers'],
                num_heads=ARCH['num_heads'],
                ffn_dim=ARCH['ffn_dim'],
                max_seq_len=ARCH['seq_len'] + 2,
                dropout=0.1,
            ).to(device)
            qk_mult = 1.0
        else:
            model = WaveFieldTransformer(
                vocab_size=vocab_size,
                embedding_dim=ARCH['embedding_dim'],
                num_layers=ARCH['num_layers'],
                num_heads=ARCH['num_heads'],
                ffn_dim=ARCH['ffn_dim'],
                field_size=ARCH['field_size'],
                max_seq_len=ARCH['seq_len'] + 2,
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
            qk_mult = cfg.get('qk_lr_mult', 3.0)

        try:
            result = train_run(
                model, train_data, val_data, vocab_size, device,
                cfg['name'], use_amp, qk_lr_mult=qk_mult,
            )
            all_results.append(result)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  OOM: {cfg['name']} -- skipping")
                all_results.append({'run_name': cfg['name'], 'best_ppl': 'OOM', 'best_acc': 'OOM'})
            else:
                raise
        finally:
            del model
            torch.cuda.empty_cache()
            gc.collect()

    # Save
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'lr_sweep.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"  QK LR SWEEP RESULTS (5M tokens, WikiText-2)")
    print(f"  {'Config':<45} {'PPL':>8} {'Acc':>8} {'Time':>8}")
    print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8}")
    for r in all_results:
        ppl = r.get('best_ppl', '-')
        acc = r.get('best_acc', '-')
        t = r.get('time_s', '-')
        ppl_s = f"{ppl:>8.1f}" if isinstance(ppl, (int, float)) else f"{ppl:>8}"
        acc_s = f"{acc:>7.1f}%" if isinstance(acc, (int, float)) else f"{acc:>8}"
        t_s = f"{t:>7.0f}s" if isinstance(t, (int, float)) else f"{t:>8}"
        print(f"  {r['run_name']:<45} {ppl_s} {acc_s} {t_s}")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()
