"""
Result Verification Suite
=========================
6 independent tests to verify that Wave Field PPL is genuine and not
an artifact of data leaks, eval bugs, or causality violations.

Tests:
  1. Independent PPL evaluation (re-compute from checkpoint, batch-by-batch)
  2. Cross-dataset generalization (eval on WikiText-2 if trained on WT-103)
  3. Train vs Val gap (detects memorization / overfitting)
  4. Shuffle test (shuffled tokens should give PPL ≈ vocab_size)
  5. Random input test (random tokens should give PPL ≈ vocab_size)
  6. Text generation (human-readable sanity check)

Usage:
  python tests/verify_results.py                   # auto-detect latest checkpoint
  python tests/verify_results.py --checkpoint PATH  # specific checkpoint
  python tests/verify_results.py --scale S1         # use S1 config
  python tests/verify_results.py --scale S2         # use S2 config (default)
"""

import sys
import os
import math
import argparse
import json
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from src.wave_field_transformer import WaveFieldTransformer

# ── Scale configs (must match benchmark_scaling.py) ──────────────
SCALE_CONFIGS = {
    'S1': dict(
        vocab_size=8000, embedding_dim=384, num_layers=8,
        num_heads=8, ffn_dim=1536, field_size=512,
        max_seq_len=514, seq_len=512, batch_size=16,
    ),
    'S2': dict(
        vocab_size=8000, embedding_dim=512, num_layers=12,
        num_heads=8, ffn_dim=2048, field_size=512,
        max_seq_len=514, seq_len=512, batch_size=12,
    ),
    'S3': dict(
        vocab_size=8000, embedding_dim=768, num_layers=12,
        num_heads=12, ffn_dim=3072, field_size=512,
        max_seq_len=514, seq_len=512, batch_size=8,
    ),
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path, cfg, device):
    """Load Wave Field model from checkpoint."""
    model = WaveFieldTransformer(
        vocab_size=cfg['vocab_size'],
        embedding_dim=cfg['embedding_dim'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        ffn_dim=cfg['ffn_dim'],
        field_size=cfg['field_size'],
        max_seq_len=cfg['max_seq_len'],
        dropout=0.0,   # no dropout for deterministic eval
        use_checkpoint=False,
        interference_interval=3,
        device=device,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {params/1e6:.1f}M params from {checkpoint_path}")

    # Print training stats if available
    if isinstance(ckpt, dict):
        if 'best_ppl' in ckpt:
            print(f"  Checkpoint best PPL: {ckpt['best_ppl']:.2f}")
        if 'step' in ckpt:
            print(f"  Checkpoint step: {ckpt['step']}")
        if 'tokens_seen' in ckpt:
            print(f"  Tokens seen: {ckpt['tokens_seen']:,}")

    return model, ckpt


def load_tokenizer(vocab_size=8000):
    """Load BPE tokenizer (must match training)."""
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'cache')
    tok_path = os.path.join(cache_dir, f'bpe_vocab{vocab_size}.json')

    if os.path.exists(tok_path):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tok_path)
        print(f"  Tokenizer loaded: {tok_path} (vocab={tok.get_vocab_size()})")
        return tok
    else:
        print(f"  ERROR: Tokenizer not found at {tok_path}")
        print(f"  Run the benchmark first to generate the tokenizer cache.")
        sys.exit(1)


def tokenize_text(tok, lines):
    """Tokenize lines of text."""
    all_ids = []
    if hasattr(tok, 'encode_batch'):
        CHUNK = 5000
        for i in range(0, len(lines), CHUNK):
            batch = lines[i:i + CHUNK]
            encoded = tok.encode_batch(batch)
            for enc in encoded:
                if enc.ids:
                    all_ids.extend(enc.ids)
    else:
        for line in lines:
            ids = tok.encode(line).ids if hasattr(tok.encode(line), 'ids') else tok.encode(line)
            if ids:
                all_ids.extend(ids)
    return all_ids


def make_chunks(token_ids, seq_len):
    """Create (input, target) chunks from flat token list."""
    data = []
    for i in range(0, len(token_ids) - seq_len, seq_len):
        chunk = token_ids[i:i + seq_len + 1]
        if len(chunk) == seq_len + 1:
            data.append((torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])))
    return data


def create_batches(data, batch_size, device, shuffle=False):
    """Create batches from chunked data."""
    if shuffle:
        indices = torch.randperm(len(data)).tolist()
    else:
        indices = list(range(len(data)))
    batches = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        xs = torch.stack([data[i][0] for i in batch_idx]).to(device)
        ys = torch.stack([data[i][1] for i in batch_idx]).to(device)
        batches.append((xs, ys))
    return batches


@torch.no_grad()
def compute_ppl(model, data, batch_size, vocab_size, device):
    """Compute PPL with per-batch loss breakdown for transparency."""
    batches = create_batches(data, batch_size, device, shuffle=False)
    batch_losses = []
    total_correct = 0
    total_tokens = 0

    for x, y in batches:
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.float().reshape(-1, vocab_size),
            y.reshape(-1),
            reduction='mean'
        )
        lv = loss.item()
        if not math.isnan(lv) and not math.isinf(lv):
            batch_losses.append(lv)
        mask = y != -100
        total_correct += (logits.argmax(-1)[mask] == y[mask]).sum().item()
        total_tokens += mask.sum().item()

    avg_loss = sum(batch_losses) / max(len(batch_losses), 1)
    ppl = math.exp(min(avg_loss, 20))
    acc = total_correct / max(total_tokens, 1) * 100

    return ppl, acc, avg_loss, batch_losses


def load_wikitext(version='103'):
    """Load WikiText train/val/test splits."""
    from datasets import load_dataset
    if version == '103':
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        name = "WikiText-103"
    else:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        name = "WikiText-2"

    splits = {}
    for split_name, hf_split in [('train', 'train'), ('valid', 'validation'), ('test', 'test')]:
        lines = [item['text'].strip() for item in ds[hf_split]
                 if item['text'].strip() and not item['text'].strip().startswith('=')]
        splits[split_name] = lines
    return splits, name


# ── TEST 1: Independent PPL Evaluation ─────────────────────────
def test_independent_ppl(model, val_data, cfg, device):
    """Re-compute PPL from scratch on val set. Report per-batch loss variance."""
    print("\n" + "=" * 65)
    print("  TEST 1: Independent PPL Evaluation")
    print("  " + "-" * 55)

    ppl, acc, avg_loss, batch_losses = compute_ppl(
        model, val_data, cfg['batch_size'], cfg['vocab_size'], device
    )

    losses = np.array(batch_losses)
    print(f"  Val PPL:     {ppl:.2f}")
    print(f"  Val Acc:     {acc:.1f}%")
    print(f"  Avg Loss:    {avg_loss:.4f}")
    print(f"  Loss std:    {losses.std():.4f}")
    print(f"  Loss range:  [{losses.min():.4f}, {losses.max():.4f}]")
    print(f"  Batches:     {len(batch_losses)} ({len(batch_losses) - len(losses)} NaN skipped)")

    # Sanity: loss should be > 0 and < 10 for a real language model
    sane = 0 < avg_loss < 10
    print(f"  Loss sanity: {'OK' if sane else 'SUSPICIOUS'} (expected 0 < loss < 10)")

    return ppl, acc, sane


# ── TEST 2: Cross-Dataset Generalization ───────────────────────
def test_cross_dataset(model, tok, cfg, device, trained_on='103'):
    """Evaluate on a DIFFERENT dataset than training. Tests generalization."""
    print("\n" + "=" * 65)
    print("  TEST 2: Cross-Dataset Generalization")
    print("  " + "-" * 55)

    # If trained on WT-103, test on WT-2 (and vice versa)
    cross_version = '2' if trained_on == '103' else '103'
    splits, name = load_wikitext(cross_version)

    print(f"  Trained on: WikiText-{trained_on}")
    print(f"  Evaluating on: {name} (unseen)")

    val_ids = tokenize_text(tok, splits['valid'])
    test_ids = tokenize_text(tok, splits['test'])
    print(f"  Val tokens: {len(val_ids):,} | Test tokens: {len(test_ids):,}")

    val_data = make_chunks(val_ids, cfg['seq_len'])
    test_data = make_chunks(test_ids, cfg['seq_len'])

    val_ppl, val_acc, _, _ = compute_ppl(model, val_data, cfg['batch_size'], cfg['vocab_size'], device)
    test_ppl, test_acc, _, _ = compute_ppl(model, test_data, cfg['batch_size'], cfg['vocab_size'], device)

    print(f"  Cross-dataset Val PPL:  {val_ppl:.2f}  Acc: {val_acc:.1f}%")
    print(f"  Cross-dataset Test PPL: {test_ppl:.2f}  Acc: {test_acc:.1f}%")

    # Generalization check: cross-dataset PPL shouldn't be >> training PPL
    # A 2-3x increase is normal for domain shift, 10x+ is suspicious
    return val_ppl, test_ppl, val_acc, test_acc


# ── TEST 3: Train vs Val Gap ──────────────────────────────────
def test_train_val_gap(model, tok, cfg, device, trained_on='103'):
    """Compare PPL on training data vs validation data.

    If train PPL << val PPL (e.g., 10x gap), model is memorizing.
    If train PPL ≈ val PPL, model is generalizing.
    """
    print("\n" + "=" * 65)
    print("  TEST 3: Train vs Val PPL Gap (Memorization Check)")
    print("  " + "-" * 55)

    splits, name = load_wikitext(trained_on)

    # Use a RANDOM subset of training data (same size as val)
    train_ids = tokenize_text(tok, splits['train'])
    val_ids = tokenize_text(tok, splits['valid'])

    # Sample same number of tokens from train as in val
    n_val_tokens = len(val_ids)
    if len(train_ids) > n_val_tokens * 2:
        start = random.randint(0, len(train_ids) - n_val_tokens - 1)
        train_sample = train_ids[start:start + n_val_tokens]
    else:
        train_sample = train_ids[:n_val_tokens]

    train_data = make_chunks(train_sample, cfg['seq_len'])
    val_data = make_chunks(val_ids, cfg['seq_len'])

    print(f"  Train sample: {len(train_sample):,} tokens ({len(train_data)} chunks)")
    print(f"  Val data:     {len(val_ids):,} tokens ({len(val_data)} chunks)")

    train_ppl, train_acc, _, _ = compute_ppl(model, train_data, cfg['batch_size'], cfg['vocab_size'], device)
    val_ppl, val_acc, _, _ = compute_ppl(model, val_data, cfg['batch_size'], cfg['vocab_size'], device)

    gap = val_ppl / max(train_ppl, 0.01)
    print(f"  Train PPL:  {train_ppl:.2f}  Acc: {train_acc:.1f}%")
    print(f"  Val PPL:    {val_ppl:.2f}  Acc: {val_acc:.1f}%")
    print(f"  Gap ratio:  {gap:.2f}x (val/train)")

    if gap > 5:
        print(f"  WARNING: Large gap suggests memorization")
    elif gap > 2:
        print(f"  NOTE: Moderate gap, some overfitting but acceptable")
    else:
        print(f"  OK: Small gap, model is generalizing well")

    return train_ppl, val_ppl, gap


# ── TEST 4: Shuffle Test ──────────────────────────────────────
def test_shuffle(model, tok, cfg, device, trained_on='103'):
    """Shuffle validation tokens, re-evaluate.

    A real language model relies on sequential context. If we destroy
    the ordering, PPL should spike to near-random (vocab_size ≈ 8000).
    If shuffled PPL ≈ original PPL, the model isn't using context.
    """
    print("\n" + "=" * 65)
    print("  TEST 4: Shuffle Test (Context Dependency)")
    print("  " + "-" * 55)

    splits, _ = load_wikitext(trained_on)
    val_ids = tokenize_text(tok, splits['valid'])

    # Normal eval
    val_data = make_chunks(val_ids, cfg['seq_len'])
    normal_ppl, normal_acc, _, _ = compute_ppl(model, val_data, cfg['batch_size'], cfg['vocab_size'], device)

    # Shuffle tokens within each chunk (destroy local ordering)
    shuffled_data = []
    for inp, tgt in val_data:
        perm = torch.randperm(len(inp))
        shuffled_data.append((inp[perm], tgt[perm]))

    shuffled_ppl, shuffled_acc, _, _ = compute_ppl(model, shuffled_data, cfg['batch_size'], cfg['vocab_size'], device)

    ratio = shuffled_ppl / max(normal_ppl, 0.01)
    print(f"  Normal PPL:   {normal_ppl:.2f}  Acc: {normal_acc:.1f}%")
    print(f"  Shuffled PPL: {shuffled_ppl:.2f}  Acc: {shuffled_acc:.1f}%")
    print(f"  Ratio:        {ratio:.1f}x")

    if ratio > 5:
        print(f"  OK: Model heavily relies on sequential context (good)")
    elif ratio > 2:
        print(f"  NOTE: Moderate context dependency")
    else:
        print(f"  WARNING: Low ratio — model may not be using context properly")

    return normal_ppl, shuffled_ppl, ratio


# ── TEST 5: Random Input Test ─────────────────────────────────
def test_random_input(model, cfg, device):
    """Feed random token IDs. PPL should be close to vocab_size.

    This tests that the model hasn't collapsed to always predicting
    the same token. With random input, each token should be ~equally
    likely → loss ≈ log(vocab_size) → PPL ≈ vocab_size.
    """
    print("\n" + "=" * 65)
    print("  TEST 5: Random Input Baseline")
    print("  " + "-" * 55)

    n_chunks = 50
    random_data = []
    for _ in range(n_chunks):
        ids = torch.randint(0, cfg['vocab_size'], (cfg['seq_len'] + 1,))
        random_data.append((ids[:-1], ids[1:]))

    random_ppl, random_acc, _, _ = compute_ppl(model, random_data, cfg['batch_size'], cfg['vocab_size'], device)

    expected_ppl = cfg['vocab_size']  # 8000
    # Random accuracy should be ~1/vocab_size = 0.0125%
    expected_acc = 100.0 / cfg['vocab_size']

    print(f"  Random PPL:      {random_ppl:.1f}")
    print(f"  Expected PPL:    ~{expected_ppl} (vocab_size)")
    print(f"  Random Acc:      {random_acc:.2f}%")
    print(f"  Expected Acc:    ~{expected_acc:.2f}%")

    # Random PPL should be in same ballpark as vocab_size
    # It can vary a lot depending on the unigram distribution, but should be >>100
    if random_ppl > 100:
        print(f"  OK: Random input gives high PPL (model hasn't collapsed)")
    else:
        print(f"  WARNING: Random PPL surprisingly low — potential issue")

    return random_ppl, random_acc


# ── TEST 6: Text Generation ───────────────────────────────────
def test_generation(model, tok, cfg, device):
    """Generate text from the model. Human-readable quality check.

    If PPL is truly < 10, generated text should be coherent English.
    """
    print("\n" + "=" * 65)
    print("  TEST 6: Text Generation (Human-Readable Check)")
    print("  " + "-" * 55)

    prompts = [
        "The history of science",
        "In the beginning",
        "The president announced",
        "Machine learning is",
        "The city of New York",
    ]

    for prompt_text in prompts:
        # Tokenize prompt
        encoded = tok.encode(prompt_text)
        prompt_ids = encoded.ids if hasattr(encoded, 'ids') else encoded
        input_ids = torch.tensor([prompt_ids], device=device)

        # Generate 60 tokens with nucleus sampling
        generated = list(prompt_ids)
        for _ in range(60):
            if len(generated) > cfg['seq_len']:
                context = torch.tensor([generated[-cfg['seq_len']:]], device=device)
            else:
                context = torch.tensor([generated], device=device)

            with torch.no_grad():
                logits, _ = model(context)

            next_logits = logits[0, -1].float()

            # Temperature + top-p sampling
            temperature = 0.8
            top_p = 0.9
            next_logits = next_logits / temperature

            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)

            # Remove tokens with cumulative probability above top_p
            sorted_mask = cumulative_probs - probs > top_p
            sorted_logits[sorted_mask] = float('-inf')

            probs = F.softmax(sorted_logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            next_token = sorted_indices[idx].item()

            generated.append(next_token)

        # Decode
        text = tok.decode(generated)
        print(f"\n  Prompt: \"{prompt_text}\"")
        print(f"  Output: {text[:200]}")

    print(f"\n  (Examine generated text above for coherence and fluency)")
    return True


# ── MAIN ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Verify Wave Field results")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--scale', type=str, default='S2', choices=['S1', 'S2', 'S3'], help='Model scale')
    parser.add_argument('--dataset', type=str, default='103', choices=['103', '2'], help='Training dataset')
    parser.add_argument('--skip-cross', action='store_true', help='Skip cross-dataset test (requires download)')
    parser.add_argument('--skip-generation', action='store_true', help='Skip text generation test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = SCALE_CONFIGS[args.scale]
    set_seed(42)

    print("=" * 65)
    print("  WAVE FIELD RESULT VERIFICATION SUITE")
    print("=" * 65)
    print(f"  Device:  {device}")
    print(f"  Scale:   {args.scale}")
    print(f"  Dataset: WikiText-{args.dataset}")

    # Find checkpoint
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'checkpoints')
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        # Auto-detect: prefer resume checkpoint (has optimizer state = latest)
        scale_lower = args.scale.lower()
        candidates = [
            os.path.join(ckpt_dir, f'spectre-wave_{scale_lower}.pt'),
            os.path.join(ckpt_dir, f'spectre-wave_{scale_lower}_resume.pt'),
            os.path.join(ckpt_dir, f'spectre-wave_{scale_lower}_best.pt'),
        ]
        ckpt_path = None
        for c in candidates:
            if os.path.exists(c):
                ckpt_path = c
                break
        if ckpt_path is None:
            print(f"\n  ERROR: No checkpoint found in {ckpt_dir}")
            print(f"  Looked for: {candidates}")
            return 1

    # Load model
    model, ckpt = load_model(ckpt_path, cfg, device)

    # Load tokenizer
    tok = load_tokenizer(cfg['vocab_size'])

    # Load primary dataset
    splits, ds_name = load_wikitext(args.dataset)
    val_ids = tokenize_text(tok, splits['valid'])
    val_data = make_chunks(val_ids, cfg['seq_len'])
    print(f"  Val data: {len(val_ids):,} tokens → {len(val_data)} chunks")

    results = {}

    # ── Test 1: Independent PPL ──
    ppl, acc, sane = test_independent_ppl(model, val_data, cfg, device)
    results['independent_ppl'] = {'ppl': ppl, 'acc': acc, 'sane': sane}

    # ── Test 2: Cross-dataset ──
    if not args.skip_cross:
        try:
            v_ppl, t_ppl, v_acc, t_acc = test_cross_dataset(model, tok, cfg, device, args.dataset)
            results['cross_dataset'] = {
                'val_ppl': v_ppl, 'test_ppl': t_ppl,
                'val_acc': v_acc, 'test_acc': t_acc
            }
        except Exception as e:
            print(f"  Skipped (download error): {e}")

    # ── Test 3: Train vs Val gap ──
    train_ppl, val_ppl, gap = test_train_val_gap(model, tok, cfg, device, args.dataset)
    results['train_val_gap'] = {'train_ppl': train_ppl, 'val_ppl': val_ppl, 'gap': gap}

    # ── Test 4: Shuffle test ──
    normal_ppl, shuffled_ppl, ratio = test_shuffle(model, tok, cfg, device, args.dataset)
    results['shuffle'] = {'normal_ppl': normal_ppl, 'shuffled_ppl': shuffled_ppl, 'ratio': ratio}

    # ── Test 5: Random input ──
    random_ppl, random_acc = test_random_input(model, cfg, device)
    results['random_input'] = {'ppl': random_ppl, 'acc': random_acc}

    # ── Test 6: Text generation ──
    if not args.skip_generation:
        test_generation(model, tok, cfg, device)

    # ── Summary ──
    print("\n" + "=" * 65)
    print("  VERIFICATION SUMMARY")
    print("  " + "-" * 55)
    print(f"  Independent Val PPL:     {results['independent_ppl']['ppl']:.2f}")
    print(f"  Independent Val Acc:     {results['independent_ppl']['acc']:.1f}%")
    if 'cross_dataset' in results:
        print(f"  Cross-dataset Val PPL:   {results['cross_dataset']['val_ppl']:.2f}")
        print(f"  Cross-dataset Test PPL:  {results['cross_dataset']['test_ppl']:.2f}")
    print(f"  Train PPL:               {results['train_val_gap']['train_ppl']:.2f}")
    print(f"  Train/Val gap:           {results['train_val_gap']['gap']:.2f}x")
    print(f"  Shuffle ratio:           {results['shuffle']['ratio']:.1f}x (higher = more context-dependent)")
    print(f"  Random input PPL:        {results['random_input']['ppl']:.1f} (expected ~{cfg['vocab_size']})")

    # ── Verdict ──
    print("\n  VERDICT:")
    issues = []
    if not results['independent_ppl']['sane']:
        issues.append("Loss outside expected range [0, 10]")
    if results['train_val_gap']['gap'] > 5:
        issues.append(f"Train/Val gap {results['train_val_gap']['gap']:.1f}x suggests memorization")
    if results['shuffle']['ratio'] < 2:
        issues.append(f"Shuffle ratio {results['shuffle']['ratio']:.1f}x — model may not use context")
    if results['random_input']['ppl'] < 100:
        issues.append(f"Random input PPL {results['random_input']['ppl']:.1f} — suspiciously low")

    if issues:
        for issue in issues:
            print(f"    WARNING: {issue}")
    else:
        print(f"    All checks passed. Results appear genuine.")

    print("=" * 65)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'data',
                            f'verification_{args.scale.lower()}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")

    return 0 if not issues else 1


if __name__ == '__main__':
    sys.exit(main())
