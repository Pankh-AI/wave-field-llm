"""
Wave Field V3.5 + BPE Tokenizer — WikiText-2 Benchmark
========================================================
Same V3.5 physics architecture, but with Byte-Level BPE tokenizer
instead of character-level FieldTokenizerV2.

BPE benefits:
- Proper word boundaries (no more "jackbourghumanism")
- Spaces as explicit tokens
- 128 tokens covers paragraphs instead of sentences
- Standard Transformer uses same BPE → fair comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import math
import json

from src.wave_field_transformer import WaveFieldTransformer


# ======================================================================
# STANDARD TRANSFORMER BASELINE
# ======================================================================

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, num_layers=6,
                 num_heads=8, ffn_dim=1024, max_seq_len=256, dropout=0.1):
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

def train_bpe_tokenizer(train_texts, vocab_size=8000):
    """Train a Byte-Level BPE tokenizer on the training data."""
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
    """Wrapper to give BPE tokenizer same interface as FieldTokenizerV2."""
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

def load_wikitext2():
    from datasets import load_dataset
    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    splits = {}
    for split_name, hf_split in [('train', 'train'), ('valid', 'validation'), ('test', 'test')]:
        lines = [item['text'].strip() for item in ds[hf_split]
                 if item['text'].strip() and not item['text'].strip().startswith('=')]
        splits[split_name] = lines
    return splits


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
        if len(ids) > max_seq_len:
            for s in range(0, len(ids) - 1, max_seq_len):
                c = ids[s:s + max_seq_len + 1]
                if len(c) >= 2:
                    data.append((torch.tensor(c[:-1]), torch.tensor(c[1:])))
        else:
            data.append((torch.tensor(ids[:-1]), torch.tensor(ids[1:])))
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
                model_name, num_epochs=30, batch_size=32, peak_lr=0.0003,
                use_amp=True, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    params = sum(p.numel() for p in model.parameters())
    print(f"\n  {model_name}: {params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01, eps=1e-8)
    spe = math.ceil(len(train_data) / batch_size)
    scheduler = WarmupCosineScheduler(optimizer, spe * 3, spe * num_epochs, min_lr=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_vl, best_vp, best_va, best_ep = float('inf'), float('inf'), 0, 0

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        et = time.time()
        model.train()
        batches = create_batches(train_data, batch_size, device)
        tl, nb = 0, 0
        for x, y in batches:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100)
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            tl += loss.item(); nb += 1
        al = tl / max(nb, 1)
        et = time.time() - et

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            vb = create_batches(val_data, batch_size, device, shuffle=False)
            vl, vp, va = evaluate(model, vb, vocab_size, device, use_amp)
            if vl < best_vl:
                best_vl, best_vp, best_va, best_ep = vl, vp, va, epoch
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
                mk = " * BEST"
            else:
                mk = ""
            print(f"  Ep {epoch:3d}/{num_epochs} | Train {al:.4f} | Val {vl:.4f} PPL {vp:.1f} Acc {va:.1f}% | {et:.1f}s{mk}")
            if epoch % 10 == 0:
                sample = generate_text(model, tok, "The president of the", device, max_tokens=30)
                print(f"    Gen: {sample}")
        else:
            print(f"  Ep {epoch:3d}/{num_epochs} | Train {al:.4f} | {et:.1f}s")

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
    print("=" * 65)
    print("  WAVE FIELD V3.5 + BPE TOKENIZER")
    print("  Standard Transformer vs Wave Field — fair BPE comparison")
    print("=" * 65)

    splits = load_wikitext2()

    # Train BPE tokenizer
    bpe_vocab_size = 8000
    print(f"\nTraining Byte-Level BPE tokenizer (vocab={bpe_vocab_size})...")
    raw_tokenizer = train_bpe_tokenizer(splits['train'], vocab_size=bpe_vocab_size)
    tok = BPEWrapper(raw_tokenizer)
    vocab_size = tok.vocab_size_actual()
    print(f"  BPE vocab: {vocab_size} tokens")

    # Show tokenization examples
    examples = ["The president of the united states", "Scientists discovered that"]
    for ex in examples:
        ids = tok.encode(ex)
        decoded = [tok.decode([i]) for i in ids]
        print(f"  \"{ex}\" → {len(ids)} tokens: {decoded[:10]}...")

    max_seq_len = 256
    train_data = encode_lines(splits['train'], tok, max_seq_len)
    val_data = encode_lines(splits['valid'], tok, max_seq_len)
    test_data = encode_lines(splits['test'], tok, max_seq_len)

    print(f"\n  max_seq_len: {max_seq_len}")
    print(f"  Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"  Device: {device} | AMP: {use_amp}")

    num_epochs = 30
    batch_size = 32
    peak_lr = 0.0003
    field_size = 1024
    results = []

    # ============================================================
    # MODEL 1: Standard Transformer
    # ============================================================
    print(f"\n{'='*65}")
    print("  MODEL 1: STANDARD TRANSFORMER (O(n^2))")
    print(f"{'='*65}")

    std_model = StandardTransformer(
        vocab_size=vocab_size,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        ffn_dim=1024,
        max_seq_len=max_seq_len + 1,
        dropout=0.1,
    ).to(device)

    std_result = train_model(
        std_model, train_data, val_data, tok, vocab_size, device,
        "Standard Transformer", num_epochs=num_epochs, batch_size=batch_size,
        peak_lr=peak_lr, use_amp=use_amp, save_dir="bpe_std_checkpoints",
    )
    results.append(std_result)

    print(f"\n  --- Standard Transformer Generation ---")
    for seed in ["The president of the", "In the year", "Scientists discovered that",
                 "The city of New York", "He was born in"]:
        text = generate_text(std_model, tok, seed, device, max_tokens=40)
        print(f"  [{seed}] -> {text}")

    test_batches = create_batches(test_data, batch_size, device, shuffle=False)
    std_tl, std_tp, std_ta = evaluate(std_model, test_batches, vocab_size, device, use_amp)
    std_result['test_ppl'] = std_tp
    std_result['test_acc'] = std_ta
    print(f"\n  TEST: PPL {std_tp:.1f} | Acc {std_ta:.1f}%")

    del std_model
    torch.cuda.empty_cache()

    # ============================================================
    # MODEL 2: Wave Field V3.5
    # ============================================================
    print(f"\n{'='*65}")
    print("  MODEL 2: WAVE FIELD V3.5 (O(n log n))")
    print(f"{'='*65}")

    wave_model = WaveFieldTransformer(
        vocab_size=vocab_size,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        ffn_dim=1024,
        field_size=field_size,
        max_seq_len=max_seq_len + 1,
        dropout=0.1,
        use_checkpoint=True,
        interference_interval=3,
        device=device,
    ).to(device)

    wave_result = train_model(
        wave_model, train_data, val_data, tok, vocab_size, device,
        "Wave Field V3.5", num_epochs=num_epochs, batch_size=batch_size,
        peak_lr=peak_lr, use_amp=use_amp, save_dir="bpe_wave_v35_checkpoints",
    )
    results.append(wave_result)

    print(f"\n  --- Wave Field V3.5 Generation ---")
    for seed in ["The president of the", "In the year", "Scientists discovered that",
                 "The city of New York", "He was born in"]:
        text = generate_text(wave_model, tok, seed, device, max_tokens=40)
        print(f"  [{seed}] -> {text}")

    test_batches = create_batches(test_data, batch_size, device, shuffle=False)
    wave_tl, wave_tp, wave_ta = evaluate(wave_model, test_batches, vocab_size, device, use_amp)
    wave_result['test_ppl'] = wave_tp
    wave_result['test_acc'] = wave_ta
    print(f"\n  TEST: PPL {wave_tp:.1f} | Acc {wave_ta:.1f}%")

    del wave_model
    torch.cuda.empty_cache()

    # ============================================================
    # COMPARISON
    # ============================================================
    print(f"\n{'='*65}")
    print("  BPE BENCHMARK RESULTS")
    print(f"{'='*65}")

    print(f"\n  {'Metric':<20} {'Std Transformer':>18} {'Wave V3.5':>18} {'Winner':>10}")
    print(f"  {'-'*20} {'-'*18} {'-'*18} {'-'*10}")

    s_tp, w_tp = results[0]['test_ppl'], results[1]['test_ppl']
    s_ta, w_ta = results[0]['test_acc'], results[1]['test_acc']

    winner_p = "Wave V3.5" if w_tp < s_tp else "Standard"
    winner_a = "Wave V3.5" if w_ta > s_ta else "Standard"
    print(f"  {'Test PPL':<20} {s_tp:>18.1f} {w_tp:>18.1f} {winner_p:>10}")
    print(f"  {'Test Accuracy':<20} {s_ta:>17.1f}% {w_ta:>17.1f}% {winner_a:>10}")
    print(f"  {'Parameters':<20} {results[0]['params']:>18,} {results[1]['params']:>18,}")
    print(f"  {'Train Time':<20} {results[0]['total_time']/60:>17.1f}m {results[1]['total_time']/60:>17.1f}m")
    print(f"  {'Complexity':<20} {'O(n^2)':>18} {'O(n log n)':>18}")
    print(f"  {'Tokenizer':<20} {'BPE '+str(vocab_size):>18} {'BPE '+str(vocab_size):>18}")
    print(f"  {'Seq Length':<20} {max_seq_len:>18} {max_seq_len:>18}")

    if w_tp < s_tp:
        pct = (s_tp - w_tp) / s_tp * 100
        print(f"\n  Wave V3.5 BEATS Standard Transformer by {pct:.1f}% on test PPL!")
    else:
        pct = (w_tp - s_tp) / s_tp * 100
        print(f"\n  Standard beats Wave V3.5 by {pct:.1f}% — but Wave uses O(n log n)")

    with open("bpe_benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*65}")
    print("  BENCHMARK COMPLETE")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
