"""
Kernel Mixture Benchmark: Content-Adaptive vs SpectralGate vs Standard
======================================================================

Smart benchmark: Standard first (set the bar), then K=4, then expand.
3M tokens per config — enough for 6 eval checkpoints to see the trend.
If a config is clearly losing by checkpoint 3, we know.

Configs (ordered: baseline first, then experimental):
  E) Standard Transformer  — the bar to beat
  A) V4.3 SPECTRE          — current best wave (SpectralGate)
  B) Kernel Mixture K=4    — the main experiment
  C) Kernel Mixture K=2    — ablation (less expressive)
  D) Kernel Mixture K=8    — ablation (more expressive)

Quick:  CONFIGS=E,B      (~5 min)
Full:   all configs       (~12 min)
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
    def __init__(self, vocab_size, embedding_dim=256, num_layers=6,
                 num_heads=8, ffn_dim=1024, max_seq_len=1024, dropout=0.1):
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
        causal_mask = nn.Transformer.generate_square_subsequent_mask(N, device=input_ids.device)
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
# DATA (with caching)
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
    def vocab_size_actual(self):
        return self.tokenizer.get_vocab_size()

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '.cache', 'data')

def load_cached_data(vocab_size=8000):
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
        print(f"  Cached: {len(train_ids):,} train | {len(val_ids):,} val tokens")
        return tok, train_ids, val_ids

    from datasets import load_dataset
    print("  Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    splits = {}
    for split_name, hf_split in [('train', 'train'), ('valid', 'validation')]:
        lines = [item['text'].strip() for item in ds[hf_split]
                 if item['text'].strip() and not item['text'].strip().startswith('=')]
        splits[split_name] = lines

    print("  Training BPE tokenizer...")
    raw_tok = train_bpe_tokenizer(splits['train'], vocab_size=vocab_size)
    tok = BPEWrapper(raw_tok)
    train_ids, val_ids = [], []
    for line in splits['train']:
        ids = tok.encode(line)
        if ids: train_ids.extend(ids)
    for line in splits['valid']:
        ids = tok.encode(line)
        if ids: val_ids.extend(ids)

    raw_tok.save(tok_path)
    torch.save(torch.tensor(train_ids, dtype=torch.int32), train_path)
    torch.save(torch.tensor(val_ids, dtype=torch.int32), val_path)
    print(f"  Built: {len(train_ids):,} train | {len(val_ids):,} val tokens")
    return tok, train_ids, val_ids


def make_chunks(token_ids, seq_len):
    data = []
    for i in range(0, len(token_ids) - seq_len, seq_len):
        chunk = token_ids[i:i + seq_len + 1]
        if len(chunk) == seq_len + 1:
            data.append((torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])))
    return data

def create_batches(data, batch_size, device, shuffle=True):
    indices = torch.randperm(len(data)).tolist() if shuffle else list(range(len(data)))
    batches = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        bx = torch.stack([data[i][0] for i in batch_idx]).to(device)
        by = torch.stack([data[i][1] for i in batch_idx]).to(device)
        batches.append((bx, by))
    return batches


# ======================================================================
# EVALUATE + LR SCHEDULE
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
        total_loss += loss.item(); n += 1
        mask = y != -100
        total_correct += (logits.argmax(-1)[mask] == y[mask]).sum().item()
        total_tokens += mask.sum().item()
    model.train()
    avg_loss = total_loss / max(n, 1)
    return math.exp(min(avg_loss, 20)), total_correct / max(total_tokens, 1) * 100

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
# CONFIG
# ======================================================================

SEQ_LEN = 512
BATCH_SIZE = 16
TOKEN_BUDGET = 3_000_000  # 3M tokens — 6 eval checkpoints at 500K each

ARCH = {
    'embedding_dim': 256, 'num_layers': 6, 'num_heads': 8,
    'ffn_dim': 1024, 'field_size': 1024, 'peak_lr': 3e-4,
}

# Standard first (set the bar), then SPECTRE baseline, then kernel mixture variants
CONFIGS = [
    {'key': 'E', 'name': 'E) Standard Transformer', 'type': 'standard'},
    {'key': 'A', 'name': 'A) V4.3 SPECTRE (baseline)', 'type': 'wave',
     'use_kernel_mixture': False},
    {'key': 'B', 'name': 'B) Kernel Mixture K=4', 'type': 'wave',
     'use_kernel_mixture': True, 'num_basis_kernels': 4},
    {'key': 'C', 'name': 'C) Kernel Mixture K=2', 'type': 'wave',
     'use_kernel_mixture': True, 'num_basis_kernels': 2},
    {'key': 'D', 'name': 'D) Kernel Mixture K=8', 'type': 'wave',
     'use_kernel_mixture': True, 'num_basis_kernels': 8},
]


# ======================================================================
# TRAIN
# ======================================================================

def train_run(model, train_ids, val_ids, vocab_size, device, cfg, use_amp):
    tokens_per_step = BATCH_SIZE * SEQ_LEN
    total_steps = TOKEN_BUDGET // tokens_per_step
    params = sum(p.numel() for p in model.parameters())

    train_data = make_chunks(train_ids, SEQ_LEN)
    val_data = make_chunks(val_ids, SEQ_LEN)

    print(f"\n  --- {cfg['name']} ---")
    print(f"  Params: {params:,} | Steps: {total_steps:,} | "
          f"seq={SEQ_LEN} batch={BATCH_SIZE}", flush=True)

    if hasattr(model, 'configure_optimizer'):
        optimizer = model.configure_optimizer(base_lr=ARCH['peak_lr'], qk_lr_mult=3.0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=ARCH['peak_lr'],
                                      weight_decay=0.01, eps=1e-8)

    warmup = max(total_steps // 10, 50)
    scheduler = WarmupCosineScheduler(optimizer, warmup, total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_ppl = float('inf')
    best_acc = 0
    tokens_seen = 0
    step = 0
    curve = []

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    eval_interval = max(total_steps // 6, 10)  # ~6 eval checkpoints

    # Initial eval
    ppl, acc = evaluate(model, val_data, BATCH_SIZE, vocab_size, device, use_amp)
    curve.append({'tokens_M': 0, 'ppl': round(ppl, 1), 'acc': round(acc, 1)})
    print(f"    Step     0 | PPL {ppl:>7.1f} Acc {acc:>5.1f}% | init", flush=True)

    while tokens_seen < TOKEN_BUDGET:
        batches = create_batches(train_data, BATCH_SIZE, device, shuffle=True)
        for x, y in batches:
            if tokens_seen >= TOKEN_BUDGET:
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

            if step % eval_interval == 0 or tokens_seen >= TOKEN_BUDGET:
                ppl, acc = evaluate(model, val_data, BATCH_SIZE, vocab_size, device, use_amp)
                mark = ""
                if ppl < best_ppl:
                    best_ppl = ppl; best_acc = acc; mark = " *BEST"
                elapsed = time.time() - t0
                tps = tokens_seen / elapsed
                print(f"    Step {step:>5} | Tok {tokens_seen/1e6:.1f}M | "
                      f"PPL {ppl:>7.1f} Acc {acc:>5.1f}% | "
                      f"{tps:,.0f} tok/s{mark}", flush=True)
                curve.append({
                    'tokens_M': round(tokens_seen / 1e6, 2),
                    'ppl': round(ppl, 1), 'acc': round(acc, 1),
                })

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    total_time = time.time() - t0

    return {
        'run_name': cfg['name'], 'params': params,
        'best_ppl': round(best_ppl, 1), 'best_acc': round(best_acc, 1),
        'tokens_seen': tokens_seen, 'time_s': round(total_time, 1),
        'tok_per_s': round(tokens_seen / total_time),
        'peak_vram_gb': round(peak_mem, 2), 'curve': curve,
    }


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("  KERNEL MIXTURE BENCHMARK")
    print("  Content-Adaptive Kernel Mixture vs SpectralGate vs Standard")
    print(f"  {TOKEN_BUDGET/1e6:.0f}M tokens, seq={SEQ_LEN}, WikiText-2")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    tok, train_ids, val_ids = load_cached_data(vocab_size=8000)
    vocab_size = tok.vocab_size_actual()

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
        torch.cuda.empty_cache(); gc.collect()

        if cfg['type'] == 'standard':
            model = StandardTransformer(
                vocab_size=vocab_size,
                embedding_dim=ARCH['embedding_dim'],
                num_layers=ARCH['num_layers'],
                num_heads=ARCH['num_heads'],
                ffn_dim=ARCH['ffn_dim'],
                max_seq_len=SEQ_LEN + 2,
                dropout=0.1,
            ).to(device)
        else:
            use_km = cfg.get('use_kernel_mixture', False)
            num_bk = cfg.get('num_basis_kernels', 4)
            model = WaveFieldTransformer(
                vocab_size=vocab_size,
                embedding_dim=ARCH['embedding_dim'],
                num_layers=ARCH['num_layers'],
                num_heads=ARCH['num_heads'],
                ffn_dim=ARCH['ffn_dim'],
                field_size=ARCH['field_size'],
                max_seq_len=SEQ_LEN + 2,
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
                use_kernel_mixture=use_km,
                num_basis_kernels=num_bk,
            ).to(device)

        try:
            result = train_run(model, train_ids, val_ids, vocab_size,
                               device, cfg, use_amp)
            all_results.append(result)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  ** OOM: {cfg['name']} **")
                all_results.append({
                    'run_name': cfg['name'], 'best_ppl': 'OOM', 'best_acc': 'OOM',
                })
            else:
                raise
        finally:
            del model; torch.cuda.empty_cache(); gc.collect()

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'kernel_mixture.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  KERNEL MIXTURE RESULTS ({TOKEN_BUDGET/1e6:.0f}M tokens, seq={SEQ_LEN})")
    print(f"  {'Config':<35} {'PPL':>7} {'Acc':>7} {'Params':>10} {'tok/s':>8}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*10} {'-'*8}")
    for r in all_results:
        ppl = r.get('best_ppl', '-')
        acc = r.get('best_acc', '-')
        par = r.get('params', '-')
        tps = r.get('tok_per_s', '-')
        ppl_s = f"{ppl:>7.1f}" if isinstance(ppl, (int, float)) else f"{ppl:>7}"
        acc_s = f"{acc:>6.1f}%" if isinstance(acc, (int, float)) else f"{acc:>7}"
        par_s = f"{par:>10,}" if isinstance(par, int) else f"{par:>10}"
        tps_s = f"{tps:>8,}" if isinstance(tps, int) else f"{tps:>8}"
        print(f"  {r['run_name']:<35} {ppl_s} {acc_s} {par_s} {tps_s}")
    print(f"{'=' * 70}")
    print("\n  Results saved to results/kernel_mixture.json")


if __name__ == '__main__':
    main()
