"""
Context-Length Quality Scaling Experiment
==========================================
THE critical experiment: does Wave Field + long context beat
Standard Transformer + short context?

Standard Transformer is capped by O(n^2) — beyond 1024 tokens,
training becomes impractically slow. Wave Field barely notices.
If longer context improves quality, Wave wins by AFFORDING more context.

Design:
  - Both models: ~8M params, BPE 8K, WikiText-2
  - Standard: train at context 256, 512, 1024
  - Wave: train at context 256, 512, 1024, 2048, 4096
  - Same total tokens seen per run (fair comparison)
  - Measure: validation perplexity, training throughput

Hypothesis: Wave @ 2048 will have LOWER perplexity than Standard @ 256,
despite Wave being architecturally "weaker" per-token, because it sees
8x more context per prediction.

Hardware: RTX 3060 (6GB). ~2-3 hours total.
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
# STANDARD TRANSFORMER
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
    """Tokenize entire corpus into one long token stream."""
    all_ids = []
    for line in lines:
        ids = tok.encode(line)
        if ids:
            all_ids.extend(ids)
    return all_ids


def make_chunks(token_ids, seq_len):
    """Slice token stream into (input, target) pairs of given context length."""
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
# TRAINING
# ======================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            p = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * p))
        for pg in self.optimizer.param_groups:
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
              total_token_budget, seq_len, batch_size, peak_lr=0.0003, use_amp=True):
    """
    Train for a fixed TOKEN BUDGET (not epochs).
    This ensures fair comparison across different context lengths.
    """
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
    eval_interval = max(total_steps // 10, 50)  # ~10 evals per run

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
    print("  CONTEXT-LENGTH QUALITY SCALING EXPERIMENT")
    print("  Does Wave Field + long context beat Standard + short context?")
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

    # Tokenize full corpus once
    print(f"  Tokenizing corpus...")
    train_ids = tokenize_corpus(splits['train'], tok)
    val_ids = tokenize_corpus(splits['valid'], tok)
    print(f"  Train: {len(train_ids):,} tokens | Val: {len(val_ids):,} tokens")

    # Fixed config
    embedding_dim = 256
    num_layers = 6
    num_heads = 8
    ffn_dim = 1024
    field_size = 1024
    peak_lr = 0.0003

    # CRITICAL: same token budget for ALL runs (fair comparison)
    # ~5M tokens — enough to learn, small enough to finish in ~15 min per run
    total_token_budget = 5_000_000

    # Batch sizes tuned for 6GB VRAM at each context length
    batch_sizes = {256: 32, 512: 16, 1024: 8, 2048: 4, 4096: 2}

    # Define experiment runs
    # Format: (model_type, seq_len, field_size_override_or_None)
    runs = [
        # Standard Transformer at increasing context
        ('standard', 256,  None),
        ('standard', 512,  None),
        ('standard', 1024, None),
        # Wave Field at increasing context — fixed field (current arch)
        ('wave', 256,  1024),
        ('wave', 512,  1024),
        ('wave', 1024, 1024),
        ('wave', 2048, 1024),   # stride=0.5, 2 tokens/cell — compressed
        # Wave Field — scaled field (field grows with context)
        ('wave', 2048, 2048),   # stride=1.0, no compression
        ('wave', 4096, 4096),   # stride=1.0, full resolution
    ]

    all_results = []

    for model_type, seq_len, fs_override in runs:
        # Prepare chunks for this context length
        train_data = make_chunks(train_ids, seq_len)
        val_data = make_chunks(val_ids, seq_len)
        bs = batch_sizes[seq_len]

        if len(train_data) < bs or len(val_data) < bs:
            print(f"\n  SKIP {model_type} @ {seq_len}: not enough data for batch_size={bs}")
            continue

        # Build model
        if model_type == 'standard':
            model = StandardTransformer(
                vocab_size=vocab_size, embedding_dim=embedding_dim,
                num_layers=num_layers, num_heads=num_heads,
                ffn_dim=ffn_dim, max_seq_len=seq_len + 2, dropout=0.1,
            ).to(device)
            name = f"Standard @ {seq_len}"
        else:
            fs = fs_override if fs_override else field_size
            tag = "fixed" if fs == 1024 and seq_len > 1024 else ""
            stride = (fs - 1) / max(seq_len, 1)
            model = WaveFieldTransformer(
                vocab_size=vocab_size, embedding_dim=embedding_dim,
                num_layers=num_layers, num_heads=num_heads,
                ffn_dim=ffn_dim, field_size=fs,
                max_seq_len=seq_len + 2, dropout=0.1,
                use_checkpoint=True, interference_interval=3, device=device,
            ).to(device)
            if tag:
                name = f"Wave @ {seq_len} (f={fs} compressed)"
            elif fs > 1024:
                name = f"Wave @ {seq_len} (f={fs} scaled)"
            else:
                name = f"Wave @ {seq_len}"

            print(f"  [field_size={fs}, stride={stride:.2f}, "
                  f"{'COMPRESSED' if stride < 1.0 else 'OK'}]")

        # Train
        try:
            result = train_run(
                model, train_data, val_data, vocab_size, device, name,
                total_token_budget, seq_len, bs, peak_lr, use_amp,
            )
            if fs_override:
                result['field_size'] = fs_override
            all_results.append(result)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  OOM: {name} — skipping")
                all_results.append({
                    'run_name': name, 'seq_len': seq_len,
                    'best_ppl': 'OOM', 'best_acc': 'OOM',
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
    print("  RESULTS: CONTEXT-LENGTH QUALITY SCALING")
    print(f"{'='*72}")
    print(f"  Token budget: {total_token_budget:,} per run (fair comparison)")
    print(f"  Vocab: BPE {vocab_size} | Embed: {embedding_dim} | Layers: {num_layers}")

    print(f"\n  {'Run':<20} {'Context':>8} {'Val PPL':>10} {'Acc':>8} {'Tok/s':>10} {'Time':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")

    for r in all_results:
        name = r['run_name']
        sl = r['seq_len']
        ppl = r.get('best_ppl', 'OOM')
        acc = r.get('best_acc', 'OOM')
        tps = r.get('tokens_per_sec', '-')
        t = r.get('total_time_s', '-')

        ppl_str = f"{ppl:.1f}" if isinstance(ppl, (int, float)) else str(ppl)
        acc_str = f"{acc:.1f}%" if isinstance(acc, (int, float)) else str(acc)
        tps_str = f"{tps:,}" if isinstance(tps, (int, float)) else str(tps)
        t_str = f"{t:.0f}s" if isinstance(t, (int, float)) else str(t)

        print(f"  {name:<20} {sl:>8} {ppl_str:>10} {acc_str:>8} {tps_str:>10} {t_str:>8}")

    # ============================================================
    # KEY COMPARISONS
    # ============================================================
    print(f"\n  --- KEY COMPARISONS ---")

    # Find Standard @ 256 as baseline
    std_256 = next((r for r in all_results
                    if r['run_name'] == 'Standard @ 256' and isinstance(r.get('best_ppl'), (int, float))),
                   None)

    if std_256:
        baseline_ppl = std_256['best_ppl']
        print(f"\n  Baseline: Standard @ 256 = PPL {baseline_ppl:.1f}")

        for r in all_results:
            if r['run_name'] == std_256['run_name']:
                continue
            if not isinstance(r.get('best_ppl'), (int, float)):
                continue
            ppl = r['best_ppl']
            if ppl < baseline_ppl:
                improvement = (baseline_ppl - ppl) / baseline_ppl * 100
                print(f"  {r['run_name']:<20} PPL {ppl:>7.1f} — {improvement:.1f}% BETTER than Standard @ 256")
            else:
                gap = (ppl - baseline_ppl) / baseline_ppl * 100
                print(f"  {r['run_name']:<20} PPL {ppl:>7.1f} — {gap:.1f}% worse than Standard @ 256")

    # ============================================================
    # THE VERDICT
    # ============================================================
    print(f"\n{'='*72}")
    print("  VERDICT")
    print(f"{'='*72}")

    # Find best Wave result
    wave_results = [r for r in all_results
                    if 'Wave' in r.get('run_name', '') and isinstance(r.get('best_ppl'), (int, float))]
    std_results = [r for r in all_results
                   if 'Standard' in r.get('run_name', '') and isinstance(r.get('best_ppl'), (int, float))]

    if wave_results and std_results:
        best_wave = min(wave_results, key=lambda r: r['best_ppl'])
        best_std = min(std_results, key=lambda r: r['best_ppl'])

        print(f"\n  Best Standard: {best_std['run_name']} — PPL {best_std['best_ppl']:.1f}")
        print(f"  Best Wave:     {best_wave['run_name']} — PPL {best_wave['best_ppl']:.1f}")

        if best_wave['best_ppl'] < best_std['best_ppl']:
            gap = (best_std['best_ppl'] - best_wave['best_ppl']) / best_std['best_ppl'] * 100
            print(f"\n  WAVE FIELD WINS by {gap:.1f}% when allowed longer context!")
            print(f"  Long context compensates for per-token expressiveness gap.")
            print(f"  -> This architecture is WORTH scaling to 100M params.")
        else:
            gap = (best_wave['best_ppl'] - best_std['best_ppl']) / best_std['best_ppl'] * 100
            print(f"\n  Standard still wins by {gap:.1f}% even at Wave's best context length.")
            if gap < 20:
                print(f"  Gap is small ({gap:.1f}%) — longer context is helping significantly.")
                print(f"  -> 100M scaling experiment is WORTH trying (gap may close further).")
            elif gap < 50:
                print(f"  Gap is moderate ({gap:.1f}%) — context helps but not enough alone.")
                print(f"  -> Consider hybrid approach before scaling.")
            else:
                print(f"  Gap is large ({gap:.1f}%) — context alone doesn't compensate.")
                print(f"  -> Architecture needs richer kernels before scaling is worthwhile.")

        # Check if Wave improves with context (slope)
        if len(wave_results) >= 2:
            wave_sorted = sorted(wave_results, key=lambda r: r['seq_len'])
            first_ppl = wave_sorted[0]['best_ppl']
            last_ppl = wave_sorted[-1]['best_ppl']
            if last_ppl < first_ppl:
                ctx_improvement = (first_ppl - last_ppl) / first_ppl * 100
                print(f"\n  Wave PPL improved {ctx_improvement:.1f}% from context "
                      f"{wave_sorted[0]['seq_len']} -> {wave_sorted[-1]['seq_len']}")
                print(f"  (Longer context IS helping Wave Field learn better)")
            else:
                print(f"\n  Wave PPL did NOT improve with longer context.")
                print(f"  (The wave kernel may not be capturing long-range dependencies)")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/context_scaling_experiment.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to results/context_scaling_experiment.json")

    print(f"\n{'='*72}")
    print("  EXPERIMENT COMPLETE")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
