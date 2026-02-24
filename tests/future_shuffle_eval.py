"""
Future Shuffle Eval: Compare prefix PPL with and without randomized suffix.
===========================================================================
For a causal model, randomizing future tokens should NOT change prefix PPL.
"""

import torch
import torch.nn.functional as F
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.wave_field_transformer import WaveFieldTransformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VOCAB_SIZE = 8000
CFG = {
    'embedding_dim': 384, 'num_layers': 8, 'num_heads': 8,
    'ffn_dim': 1536, 'field_size': 2048, 'seq_len': 512,
}


def build_model():
    return WaveFieldTransformer(
        vocab_size=VOCAB_SIZE,
        embedding_dim=CFG['embedding_dim'],
        num_layers=CFG['num_layers'],
        num_heads=CFG['num_heads'],
        ffn_dim=CFG['ffn_dim'],
        field_size=CFG['field_size'],
        max_seq_len=CFG['seq_len'] + 2,
        dropout=0.0,
        use_checkpoint=False,
        interference_interval=3,
        n_components=1,
        local_window=0,
        device=DEVICE,
    ).to(DEVICE)


@torch.no_grad()
def future_shuffle_eval(model, n_seqs=20, seq_len=256, split_pos=128):
    """
    For each random sequence:
      1. Compute loss on prefix positions [0..split_pos-1] with ORIGINAL suffix
      2. Compute loss on prefix positions [0..split_pos-1] with RANDOM suffix
    If causal: losses should be identical.
    """
    model.eval()

    normal_losses = []
    shuffled_losses = []

    for _ in range(n_seqs):
        # x = input, y = shifted labels (x[1:])
        tokens = torch.randint(0, VOCAB_SIZE, (1, seq_len + 1), device=DEVICE)
        x = tokens[:, :-1]  # (1, seq_len)
        y = tokens[:, 1:]   # (1, seq_len)

        # Normal forward
        logits1, _ = model(x)
        # Loss on prefix only (positions 0..split_pos-1)
        prefix_logits1 = logits1[:, :split_pos, :].reshape(-1, VOCAB_SIZE)
        prefix_labels1 = y[:, :split_pos].reshape(-1)
        loss1 = F.cross_entropy(prefix_logits1, prefix_labels1).item()
        normal_losses.append(loss1)

        # Shuffle suffix
        x2 = x.clone()
        x2[:, split_pos:] = torch.randint(0, VOCAB_SIZE, (1, seq_len - split_pos), device=DEVICE)

        logits2, _ = model(x2)
        prefix_logits2 = logits2[:, :split_pos, :].reshape(-1, VOCAB_SIZE)
        # Same labels (we only care about prefix positions)
        loss2 = F.cross_entropy(prefix_logits2, prefix_labels1).item()
        shuffled_losses.append(loss2)

    avg_normal = sum(normal_losses) / len(normal_losses)
    avg_shuffled = sum(shuffled_losses) / len(shuffled_losses)
    ppl_normal = math.exp(min(avg_normal, 20))
    ppl_shuffled = math.exp(min(avg_shuffled, 20))

    return ppl_normal, ppl_shuffled


def main():
    print("=" * 65)
    print("  FUTURE SHUFFLE EVAL")
    print("  Causal model: prefix PPL unchanged when suffix is randomized")
    print("=" * 65)

    ckpt = os.path.join(os.path.dirname(__file__), '..', 'results', 'spectre-wave_s1.pt')
    if not os.path.exists(ckpt):
        print(f"  Checkpoint not found: {ckpt}")
        return

    model = build_model()
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded checkpoint: {ckpt}")

    for split in [64, 128, 192]:
        ppl_n, ppl_s = future_shuffle_eval(model, n_seqs=30, seq_len=256, split_pos=split)
        diff_pct = abs(ppl_n - ppl_s) / max(ppl_n, 1) * 100
        verdict = "CAUSAL (identical)" if diff_pct < 1.0 else "LEAK (prefix changed)"
        print(f"  split={split:>3} | Normal PPL={ppl_n:>8.2f} | Shuffled PPL={ppl_s:>8.2f} | diff={diff_pct:.2f}% | {verdict}")

    print(f"\n  INTERPRETATION:")
    print(f"    diff < 1%  -> Model is causal (prefix unaffected by future)")
    print(f"    diff > 5%  -> Future leakage (prefix depends on suffix)")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
