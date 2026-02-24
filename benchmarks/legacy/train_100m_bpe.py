"""
Wave Field V3.5 — 100M Parameter Scale Test on OpenWebText
============================================================
Scale both Standard Transformer and Wave Field to ~100M params
with 32K BPE vocab on OpenWebText (real web text, ~8B tokens).

Previous results (6-8M params, BPE 8K vocab, WikiText-2):
  Standard Transformer: PPL 91.4
  Wave Field V3.5:      PPL 170.7  (87% gap)

Hypothesis: The gap is a capacity bottleneck. At 100M params with
32K vocab on a real dataset, the gap should narrow.

Config: embed=768, layers=12, heads=12, ffn=3072, BPE 32K
Data: OpenWebText (subset — ~50M tokens for tractable training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import math
import json
import gc

from src.wave_field_transformer import WaveFieldTransformer


# ======================================================================
# STANDARD TRANSFORMER BASELINE
# ======================================================================

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768, num_layers=12,
                 num_heads=12, ffn_dim=3072, max_seq_len=514, dropout=0.1):
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
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, input_ids, labels=None, mask=None):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, N = input_ids.shape
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(input_ids) + self.positional_embedding(positions)
        x = self.dropout(x)
        causal_mask = self._generate_causal_mask(N, input_ids.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.norm(x)
        logits = self.output_projection(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1), ignore_index=-100)
        return logits, loss


# ======================================================================
# BPE TOKENIZER
# ======================================================================

def train_bpe_tokenizer(train_texts, vocab_size=32000):
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
# UTILITIES
# ======================================================================

def load_openwebtext(max_docs=100000):
    """Load a subset of OpenWebText for tractable training."""
    from datasets import load_dataset
    print(f"Loading OpenWebText (streaming, up to {max_docs:,} docs)...")
    ds = load_dataset('openwebtext', split='train', streaming=True)

    texts = []
    for i, item in enumerate(ds):
        if i >= max_docs:
            break
        text = item['text'].strip()
        if len(text) > 50:
            texts.append(text)
        if (i + 1) % 20000 == 0:
            print(f"  Loaded {i+1:,} docs...")

    total_chars = sum(len(t) for t in texts)
    print(f"  Total: {len(texts):,} docs, {total_chars:,} chars (~{total_chars//4:,} tokens)")

    # Split: 95% train, 2.5% val, 2.5% test
    n = len(texts)
    train_end = int(n * 0.95)
    val_end = int(n * 0.975)

    return {
        'train': texts[:train_end],
        'valid': texts[train_end:val_end],
        'test': texts[val_end:],
    }


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
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
        return lr


def encode_lines(lines, tok, max_seq_len):
    data = []
    for line in lines:
        ids = tok.encode(line)
        if len(ids) < 2:
            continue
        for s in range(0, len(ids) - 1, max_seq_len):
            c = ids[s:s + max_seq_len + 1]
            if len(c) >= 2:
                data.append((torch.tensor(c[:-1]), torch.tensor(c[1:])))
    return data


def create_batches(data, batch_size, device, shuffle=True):
    if shuffle:
        indices = torch.randperm(len(data)).tolist()
    else:
        indices = list(range(len(data)))
    batches = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        bx = [data[i][0] for i in batch_idx]
        by = [data[i][1] for i in batch_idx]
        ml = max(x.size(0) for x in bx)
        px = torch.zeros(len(bx), ml, dtype=torch.long, device=device)
        py = torch.full((len(by), ml), -100, dtype=torch.long, device=device)
        for i, (x, y) in enumerate(zip(bx, by)):
            px[i, :x.size(0)] = x
            py[i, :y.size(0)] = y
        batches.append((px, py))
    return batches


@torch.no_grad()
def evaluate(model, batches, vocab_size, device, use_amp=False):
    model.eval()
    tl, tc, tt, n = 0, 0, 0, 0
    for x, y in batches:
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100)
        tl += loss.item(); n += 1
        mask = y != -100
        tc += (logits.argmax(-1)[mask] == y[mask]).sum().item()
        tt += mask.sum().item()
    model.train()
    al = tl / max(n, 1)
    return al, math.exp(min(al, 20)), tc / max(tt, 1) * 100


@torch.no_grad()
def generate_text(model, tok, seed, device, max_tokens=60,
                  temperature=0.8, top_k=50, top_p=0.92, rep_penalty=1.2):
    model.eval()
    ids = tok.encode(seed)
    if not ids:
        return seed + " [empty]"
    gen = ids.copy()
    inp = torch.tensor(ids, device=device).unsqueeze(0)
    for _ in range(max_tokens):
        logits, _ = model(inp)
        nl = logits[0, -1, :] / temperature
        for pid in set(gen[-30:]):
            if nl[pid] > 0: nl[pid] /= rep_penalty
            else: nl[pid] *= rep_penalty
        if top_k > 0:
            tv, ti = torch.topk(nl, min(top_k, nl.size(-1)))
            f = torch.full_like(nl, float('-inf'))
            f.scatter_(0, ti, tv)
        else:
            f = nl
        if top_p < 1.0:
            sl, si = torch.sort(f, descending=True)
            cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
            rm = cp > top_p; rm[1:] = rm[:-1].clone(); rm[0] = False
            f[si[rm]] = float('-inf')
        nid = torch.multinomial(F.softmax(f, dim=-1), 1).item()
        gen.append(nid)
        inp = torch.tensor([gen], device=device)
    model.train()
    return tok.decode(gen)


def train_model(model, train_data, val_data, tok, vocab_size, device,
                model_name, num_epochs=10, batch_size=8, grad_accum=4,
                peak_lr=0.0001, use_amp=True, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    params = sum(p.numel() for p in model.parameters())
    print(f"\n  {model_name}: {params:,} parameters")

    effective_batch = batch_size * grad_accum
    print(f"  Batch: {batch_size} x {grad_accum} accum = {effective_batch} effective")

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1,
                                   betas=(0.9, 0.95), eps=1e-8)
    spe = math.ceil(len(train_data) / batch_size)
    scheduler = WarmupCosineScheduler(optimizer, spe * 1, spe * num_epochs, min_lr=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_vl, best_vp, best_va, best_ep = float('inf'), float('inf'), 0, 0

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        et = time.time()
        model.train()
        batches = create_batches(train_data, batch_size, device)
        tl, nb = 0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(batches):
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100)
                loss = loss / grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            tl += loss.item() * grad_accum
            nb += 1

            if (step + 1) % grad_accum == 0 or (step + 1) == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % 500 == 0:
                print(f"    Step {step+1}/{len(batches)} | Loss {tl/nb:.4f}", flush=True)

        al = tl / max(nb, 1)
        et = time.time() - et

        # Evaluate every epoch (fewer epochs = every one matters)
        vb = create_batches(val_data, batch_size, device, shuffle=False)
        vl, vp, va = evaluate(model, vb, vocab_size, device, use_amp)
        if vl < best_vl:
            best_vl, best_vp, best_va, best_ep = vl, vp, va, epoch
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
            mk = " * BEST"
        else:
            mk = ""
        print(f"  Ep {epoch:3d}/{num_epochs} | Train {al:.4f} | Val {vl:.4f} PPL {vp:.1f} Acc {va:.1f}% | {et:.0f}s{mk}")

        if epoch % 2 == 0 or epoch == num_epochs:
            seeds = ["The president of the", "Scientists discovered that",
                     "In the year 2025,", "The city of New York"]
            for seed in seeds:
                text = generate_text(model, tok, seed, device, max_tokens=40)
                print(f"    [{seed}] -> {text}")

    total = time.time() - t0
    model.load_state_dict(torch.load(os.path.join(save_dir, "best.pt"), weights_only=True))

    return {
        'model_name': model_name,
        'params': params,
        'best_ppl': best_vp,
        'best_acc': best_va,
        'best_epoch': best_ep,
        'total_time': total,
    }


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("  WAVE FIELD V3.5 — 100M PARAMS / 32K VOCAB / OPENWEBTEXT")
    print("  The real test: does Wave Field scale?")
    print("=" * 70)
    print(f"\n  Previous (6-8M, WikiText-2): Std PPL 91.4 | Wave PPL 170.7 (87% gap)")
    print(f"  Now: 100M params, 32K BPE vocab, OpenWebText")

    # Load OpenWebText subset
    splits = load_openwebtext(max_docs=100000)
    print(f"\n  Train: {len(splits['train']):,} docs")
    print(f"  Val:   {len(splits['valid']):,} docs")
    print(f"  Test:  {len(splits['test']):,} docs")

    # Train 32K BPE tokenizer on training data
    bpe_vocab_size = 32000
    print(f"\nTraining Byte-Level BPE tokenizer (vocab={bpe_vocab_size})...")
    # Use subset for tokenizer training (faster)
    tok_train_texts = splits['train'][:50000]
    raw_tokenizer = train_bpe_tokenizer(tok_train_texts, vocab_size=bpe_vocab_size)
    tok = BPEWrapper(raw_tokenizer)
    vocab_size = tok.vocab_size_actual()
    print(f"  BPE vocab: {vocab_size} tokens")

    examples = [
        "The president of the United States announced today",
        "Scientists discovered that quantum computing can",
        "def fibonacci(n):\n    if n <= 1:\n        return n",
    ]
    for ex in examples:
        ids = tok.encode(ex)
        decoded = [tok.decode([i]) for i in ids[:8]]
        print(f"  \"{ex[:50]}...\" -> {len(ids)} tokens: {decoded}...")

    max_seq_len = 512
    print(f"\n  Encoding data (max_seq_len={max_seq_len})...")
    train_data = encode_lines(splits['train'], tok, max_seq_len)
    val_data = encode_lines(splits['valid'], tok, max_seq_len)
    test_data = encode_lines(splits['test'], tok, max_seq_len)

    print(f"  Train: {len(train_data):,} sequences")
    print(f"  Val:   {len(val_data):,} sequences")
    print(f"  Test:  {len(test_data):,} sequences")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"  Device: {device} | AMP: {use_amp}")

    # 100M config
    embed_dim = 768
    num_layers = 12
    num_heads = 12
    ffn_dim = 3072
    field_size = 1024
    num_epochs = 10
    batch_size = 4
    grad_accum = 8
    peak_lr = 6e-4
    results = []

    print(f"\n  CONFIG:")
    print(f"    embed={embed_dim}, layers={num_layers}, heads={num_heads}, ffn={ffn_dim}")
    print(f"    BPE vocab: {vocab_size}, max_seq_len: {max_seq_len}")
    print(f"    Batch: {batch_size} x {grad_accum} accum = {batch_size * grad_accum} effective")
    print(f"    LR: {peak_lr}, weight_decay: 0.1, epochs: {num_epochs}")
    print(f"    Vocab/embed ratio: {vocab_size/embed_dim:.1f}x")

    # ============================================================
    # MODEL 1: Standard Transformer 100M
    # ============================================================
    print(f"\n{'='*70}")
    print("  MODEL 1: STANDARD TRANSFORMER ~100M (O(n^2))")
    print(f"{'='*70}")

    std_model = StandardTransformer(
        vocab_size=vocab_size,
        embedding_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len + 2,
        dropout=0.1,
    ).to(device)

    std_result = train_model(
        std_model, train_data, val_data, tok, vocab_size, device,
        "Standard Transformer 100M", num_epochs=num_epochs,
        batch_size=batch_size, grad_accum=grad_accum,
        peak_lr=peak_lr, use_amp=use_amp, save_dir="100m_std_checkpoints",
    )
    results.append(std_result)

    print(f"\n  --- Standard Transformer 100M Generation ---")
    seeds = ["The president of the", "In the year", "Scientists discovered that",
             "The city of New York", "He was born in", "Music is",
             "The first time I saw", "It was a dark and stormy"]
    for seed in seeds:
        text = generate_text(std_model, tok, seed, device, max_tokens=50)
        print(f"  [{seed}] -> {text}")

    test_batches = create_batches(test_data, batch_size, device, shuffle=False)
    std_tl, std_tp, std_ta = evaluate(std_model, test_batches, vocab_size, device, use_amp)
    std_result['test_ppl'] = std_tp
    std_result['test_acc'] = std_ta
    print(f"\n  TEST: PPL {std_tp:.1f} | Acc {std_ta:.1f}%")

    del std_model
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # MODEL 2: Wave Field V3.5 100M
    # ============================================================
    print(f"\n{'='*70}")
    print("  MODEL 2: WAVE FIELD V3.5 ~100M (O(n log n))")
    print(f"{'='*70}")

    wave_model = WaveFieldTransformer(
        vocab_size=vocab_size,
        embedding_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        field_size=field_size,
        max_seq_len=max_seq_len + 2,
        dropout=0.1,
        use_checkpoint=True,
        interference_interval=3,
        device=device,
    ).to(device)

    wave_result = train_model(
        wave_model, train_data, val_data, tok, vocab_size, device,
        "Wave Field V3.5 100M", num_epochs=num_epochs,
        batch_size=batch_size, grad_accum=grad_accum,
        peak_lr=peak_lr, use_amp=use_amp, save_dir="100m_wave_checkpoints",
    )
    results.append(wave_result)

    print(f"\n  --- Wave Field V3.5 100M Generation ---")
    for seed in seeds:
        text = generate_text(wave_model, tok, seed, device, max_tokens=50)
        print(f"  [{seed}] -> {text}")

    test_batches = create_batches(test_data, batch_size, device, shuffle=False)
    wave_tl, wave_tp, wave_ta = evaluate(wave_model, test_batches, vocab_size, device, use_amp)
    wave_result['test_ppl'] = wave_tp
    wave_result['test_acc'] = wave_ta
    print(f"\n  TEST: PPL {wave_tp:.1f} | Acc {wave_ta:.1f}%")

    del wave_model
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # COMPARISON
    # ============================================================
    print(f"\n{'='*70}")
    print("  100M PARAMETER BENCHMARK — OPENWEBTEXT + 32K BPE")
    print(f"{'='*70}")

    print(f"\n  {'Metric':<25} {'Std 100M':>18} {'Wave 100M':>18} {'Winner':>10}")
    print(f"  {'-'*25} {'-'*18} {'-'*18} {'-'*10}")

    s_tp, w_tp = results[0]['test_ppl'], results[1]['test_ppl']
    s_ta, w_ta = results[0]['test_acc'], results[1]['test_acc']

    winner_p = "Wave V3.5" if w_tp < s_tp else "Standard"
    winner_a = "Wave V3.5" if w_ta > s_ta else "Standard"
    print(f"  {'Test PPL':<25} {s_tp:>18.1f} {w_tp:>18.1f} {winner_p:>10}")
    print(f"  {'Test Accuracy':<25} {s_ta:>17.1f}% {w_ta:>17.1f}% {winner_a:>10}")
    print(f"  {'Parameters':<25} {results[0]['params']:>18,} {results[1]['params']:>18,}")
    print(f"  {'Train Time':<25} {results[0]['total_time']/60:>17.1f}m {results[1]['total_time']/60:>17.1f}m")
    print(f"  {'Complexity':<25} {'O(n^2)':>18} {'O(n log n)':>18}")
    print(f"  {'BPE Vocab':<25} {vocab_size:>18,} {vocab_size:>18,}")
    print(f"  {'Embed Dim':<25} {embed_dim:>18} {embed_dim:>18}")
    print(f"  {'Seq Length':<25} {max_seq_len:>18} {max_seq_len:>18}")
    print(f"  {'Dataset':<25} {'OpenWebText':>18} {'OpenWebText':>18}")
    print(f"  {'Vocab/Embed Ratio':<25} {vocab_size/embed_dim:>17.1f}x {vocab_size/embed_dim:>17.1f}x")

    if w_tp < s_tp:
        pct = (s_tp - w_tp) / s_tp * 100
        print(f"\n  Wave V3.5 BEATS Standard Transformer by {pct:.1f}%!")
    else:
        pct = (w_tp - s_tp) / s_tp * 100
        print(f"\n  PPL Gap: {pct:.1f}% (Standard better)")

    prev_gap = 87.0
    print(f"\n  --- SCALING ANALYSIS ---")
    print(f"  Previous (6-8M, WikiText-2, 8K BPE):  87% gap")
    print(f"  Current (100M, OpenWebText, 32K BPE):  {pct:.1f}% gap")
    if pct < prev_gap:
        improvement = (prev_gap - pct) / prev_gap * 100
        print(f"  Gap reduced by {improvement:.1f}%!")
        if pct < 10:
            print(f"  BREAKTHROUGH: Within 10% — architecture validated for production scale!")
        elif pct < 20:
            print(f"  EXCELLENT: Within 20% — architecture is competitive at scale!")
        elif pct < 40:
            print(f"  GOOD: Significant improvement — scaling helps substantially")
        else:
            print(f"  MODERATE: Some improvement — more scaling may help")
    else:
        print(f"  Gap did not improve — fundamental bottleneck beyond capacity")

    with open("100m_benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("  100M BENCHMARK COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
