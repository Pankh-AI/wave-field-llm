"""
Wave Field LLM — Interactive Text Generation
==============================================
Loads a trained checkpoint and generates text from prompts.

Usage:
  # Inside Docker (after training):
  python inference.py --checkpoint results/spectre-wave_s1.pt --scale S1

  # Interactive mode:
  python inference.py --checkpoint results/spectre-wave_s1.pt --scale S1 --interactive
"""

import torch
import torch.nn.functional as F
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.wave_field_transformer import WaveFieldTransformer
from benchmarks.benchmark_scaling import SCALE_CONFIGS, train_bpe_tokenizer, BPEWrapper


def load_model(checkpoint_path, scale_key, vocab_size, device):
    """Load SPECTRE-Wave model from checkpoint."""
    cfg = SCALE_CONFIGS[scale_key]
    model = WaveFieldTransformer(
        vocab_size=vocab_size,
        embedding_dim=cfg['embedding_dim'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        ffn_dim=cfg['ffn_dim'],
        field_size=cfg['field_size'],
        max_seq_len=cfg['seq_len'] + 2,
        dropout=0.0,  # no dropout at inference
        use_checkpoint=False,
        interference_interval=3,
        n_components=1,
        local_window=0,
        device=device,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded {sum(p.numel() for p in model.parameters()):,} params from {checkpoint_path}")
    return model


@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=100, temperature=0.8,
             top_k=50, top_p=0.9, device='cuda'):
    """Generate text from a prompt."""
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], device=device)
    seq_len = SCALE_CONFIGS.get('S1', {}).get('seq_len', 512)

    generated = list(ids)

    for _ in range(max_tokens):
        # Truncate to max seq_len
        if input_ids.shape[1] > seq_len:
            input_ids = input_ids[:, -seq_len:]

        logits, _ = model(input_ids)
        next_logits = logits[0, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
            next_logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = float('-inf')

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description='Wave Field LLM Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pt checkpoint')
    parser.add_argument('--scale', type=str, default='S1', help='Scale config (S1/S2/S3/S4)')
    parser.add_argument('--prompt', type=str, default=None, help='Text prompt')
    parser.add_argument('--max-tokens', type=int, default=100, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Build tokenizer (same as training)
    from datasets import load_dataset
    ds_choice = os.environ.get('DATASET', '2')
    if ds_choice == '103':
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    else:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_lines = [item['text'].strip() for item in ds['train']
                   if item['text'].strip() and not item['text'].strip().startswith('=')]
    print(f"  Building BPE tokenizer...")
    raw_tok = train_bpe_tokenizer(train_lines, vocab_size=8000)
    tok = BPEWrapper(raw_tok)
    vocab_size = tok.vocab_size_actual()
    print(f"  Vocab: {vocab_size}")

    # Load model
    model = load_model(args.checkpoint, args.scale.upper(), vocab_size, device)

    if args.interactive:
        print(f"\n  === Wave Field LLM Interactive Mode ===")
        print(f"  Type a prompt and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                prompt = input("  > ")
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.strip().lower() in ('quit', 'exit', 'q'):
                break
            if not prompt.strip():
                continue
            output = generate(model, tok, prompt, args.max_tokens,
                              args.temperature, args.top_k, args.top_p, device)
            print(f"\n  {output}\n")
    elif args.prompt:
        output = generate(model, tok, args.prompt, args.max_tokens,
                          args.temperature, args.top_k, args.top_p, device)
        print(f"\n  {output}")
    else:
        # Default demo prompts
        prompts = [
            "The history of",
            "In recent years, scientists have",
            "The city of New York",
        ]
        for p in prompts:
            output = generate(model, tok, p, args.max_tokens,
                              args.temperature, args.top_k, args.top_p, device)
            print(f"\n  Prompt: {p}")
            print(f"  Output: {output}")
            print(f"  {'─' * 60}")


if __name__ == '__main__':
    main()
