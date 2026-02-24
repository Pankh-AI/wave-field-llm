"""
Wave Field V3.5 + BPE — Deep Physics Diagnostics
==================================================
Diagnose WHY the PPL gap exists between Standard Transformer (91.4)
and Wave Field V3.5 (170.7) with BPE tokenizer.

Sections:
1. Wave Kernel Analysis — what each head learned
2. Field Coupling — how heads interact
3. Field Utilization — are we using the field efficiently?
4. Information Flow — energy through layers
5. Gate Analysis — content-dependent gating patterns
6. Interference Analysis — amplification vs suppression
7. Prediction Confidence — where the model struggles
8. GAP DIAGNOSIS — root cause analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np

from src.wave_field_transformer import WaveFieldTransformer


def train_bpe_tokenizer(train_texts, vocab_size=8000):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
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


def main():
    print("=" * 65)
    print("  WAVE FIELD V3.5 + BPE — DEEP PHYSICS DIAGNOSTICS")
    print("  Diagnosing PPL gap: Wave 170.7 vs Standard 91.4")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Rebuild BPE tokenizer
    from datasets import load_dataset
    print("\nLoading WikiText-2 + BPE tokenizer...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_lines = [item['text'].strip() for item in ds['train']
                   if item['text'].strip() and not item['text'].strip().startswith('=')]

    raw_tok = train_bpe_tokenizer(train_lines, vocab_size=8000)
    tok = BPEWrapper(raw_tok)
    vocab_size = tok.vocab_size_actual()
    print(f"  BPE vocab: {vocab_size}")

    field_size = 1024
    max_seq_len = 257

    model = WaveFieldTransformer(
        vocab_size=vocab_size, embedding_dim=256, num_layers=6,
        num_heads=8, ffn_dim=1024, field_size=field_size,
        max_seq_len=max_seq_len, dropout=0.1, use_checkpoint=False,
        interference_interval=3, device=device,
    ).to(device)

    ckpt = "bpe_wave_v35_checkpoints/best.pt"
    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
    model.eval()
    print(f"  Model loaded from {ckpt}")

    test_sentences = [
        "The president of the United States",
        "In the year of our Lord",
        "Scientists discovered that the",
        "The city of New York is",
        "He was born in London",
    ]

    # ============================================================
    # 1. WAVE KERNEL ANALYSIS
    # ============================================================
    print(f"\n{'='*65}")
    print("  1. WAVE KERNEL ANALYSIS")
    print("  What listening pattern each head learned")
    print(f"{'='*65}")

    all_ranges = []
    for layer_idx, layer in enumerate(model.layers):
        attn = layer.attention
        omega = attn.wave_frequency.detach().cpu()
        alpha_raw = attn.wave_damping.detach().cpu()
        alpha = F.softplus(alpha_raw)
        phi = attn.wave_phase.detach().cpu()

        print(f"\n  Layer {layer_idx + 1}:")
        print(f"  {'Head':<6} {'Freq(w)':>8} {'Damp(a)':>8} {'Phase(p)':>8} {'Range':>8}  Role")
        print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*25}")

        layer_ranges = []
        for h in range(8):
            w = omega[h].item()
            a = alpha[h].item()
            p = phi[h].item()

            if a > 0.01:
                eff_range = min(int(-math.log(0.1) / a), field_size)
            else:
                eff_range = field_size

            layer_ranges.append(eff_range)
            all_ranges.append(eff_range)

            if a > 0.5 and abs(w) < 1.5:
                role = "LOCAL (grammar/syntax)"
            elif a < 0.3 and abs(w) > 2.0:
                role = "GLOBAL (long patterns)"
            elif a < 0.3 and abs(w) < 1.5:
                role = "WIDE SMOOTH (context)"
            elif abs(w) > 3.0:
                role = "HIGH-FREQ (periodicity)"
            else:
                role = "MEDIUM (balanced)"

            bar_len = min(eff_range // 20, 30)
            bar = "█" * bar_len + "░" * max(0, 30 - bar_len)

            print(f"  H{h+1:<4} {w:>8.3f} {a:>8.3f} {p:>8.3f} {eff_range:>7}  {role}")
            print(f"         {bar}")

    avg_range = sum(all_ranges) / len(all_ranges)
    local_heads = sum(1 for r in all_ranges if r < 50)
    global_heads = sum(1 for r in all_ranges if r > 200)
    print(f"\n  KERNEL SUMMARY:")
    print(f"    Average effective range: {avg_range:.0f} field positions")
    print(f"    Local heads (<50): {local_heads}/48")
    print(f"    Global heads (>200): {global_heads}/48")
    print(f"    BPE seq_len=256 maps to field positions 0..{int(255 * (field_size-1) / max(max_seq_len-1, 1))}")

    # ============================================================
    # 2. FIELD COUPLING ANALYSIS
    # ============================================================
    print(f"\n{'='*65}")
    print("  2. STATIC FIELD COUPLING")
    print("  How heads share information")
    print(f"{'='*65}")

    for layer_idx, layer in enumerate(model.layers):
        attn = layer.attention
        coupling = F.softmax(attn.field_coupling.detach().cpu(), dim=-1)

        print(f"\n  Layer {layer_idx + 1} coupling matrix:")
        header = "        " + "".join([f"  H{i+1}  " for i in range(8)])
        print(header)

        for h in range(8):
            row = f"  H{h+1}  "
            for h2 in range(8):
                val = coupling[h, h2].item()
                if h == h2:
                    row += f" [{val:.2f}]"
                elif val > 0.15:
                    row += f" *{val:.2f}*"
                else:
                    row += f"  {val:.2f} "
            print(row)

        self_weight = coupling.diag().mean().item()
        off_diag = coupling.clone(); off_diag.fill_diagonal_(0)
        cross_weight = off_diag.sum().item()
        entropy = -(coupling * coupling.clamp(min=1e-10).log()).sum(dim=-1).mean().item()
        max_entropy = math.log(8)
        print(f"  Self-weight avg: {self_weight:.3f} | Cross-coupling total: {cross_weight:.3f}")
        print(f"  Coupling entropy: {entropy:.3f} / {max_entropy:.3f} "
              f"({'DIVERSE' if entropy > max_entropy * 0.7 else 'CONCENTRATED'})")

    # ============================================================
    # 3. FIELD UTILIZATION — Critical for BPE gap diagnosis
    # ============================================================
    print(f"\n{'='*65}")
    print("  3. FIELD UTILIZATION ANALYSIS")
    print("  How efficiently the 1024-cell field is used")
    print(f"{'='*65}")

    for sentence in test_sentences[:3]:
        ids = tok.encode(sentence)
        N = len(ids)
        tokens = [tok.decode([i]) for i in ids]

        seq_pos = torch.arange(N, dtype=torch.float32)
        stride = (field_size - 1) / max(max_seq_len - 1, 1)
        field_pos = (seq_pos * stride).clamp(0, field_size - 2)

        max_pos = field_pos[-1].item()
        utilization = max_pos / (field_size - 1) * 100

        avg_gap = (field_pos[1:] - field_pos[:-1]).mean().item() if N > 1 else 0

        print(f"\n  \"{sentence}\" ({N} BPE tokens)")
        print(f"    Field positions: {field_pos[0]:.1f} to {field_pos[-1]:.1f} (of {field_size})")
        print(f"    Field utilization: {utilization:.1f}%")
        print(f"    Avg gap between tokens: {avg_gap:.1f} field cells")
        print(f"    Tokens mapped: {', '.join(f'{t}→{p:.0f}' for t, p in zip(tokens[:8], field_pos[:8].tolist()))}")

    print(f"\n  UTILIZATION DIAGNOSIS:")
    print(f"    Stride: {stride:.2f} field cells per token")
    print(f"    With 256 BPE tokens → occupies {min(255*stride, field_size-1):.0f}/{field_size} field cells")
    short_util = min(10 * stride, field_size - 1) / (field_size - 1) * 100
    long_util = min(255 * stride, field_size - 1) / (field_size - 1) * 100
    print(f"    Short seq (10 tokens): {short_util:.1f}% utilization")
    print(f"    Long seq (256 tokens): {long_util:.1f}% utilization")
    if avg_gap > 2.0:
        print(f"    WARNING: Tokens are {avg_gap:.1f} cells apart — wave kernels need sufficient range")

    # ============================================================
    # 4. INFORMATION FLOW TRACE
    # ============================================================
    print(f"\n{'='*65}")
    print("  4. INFORMATION FLOW TRACE")
    print("  Energy through layers for each token")
    print(f"{'='*65}")

    for sentence in test_sentences[:3]:
        ids = tok.encode(sentence)
        tokens = [tok.decode([i]) for i in ids]
        input_ids = torch.tensor([ids], device=device)

        energies = []
        gate_vals = []

        def make_hook(layer_idx):
            def hook(module, inp, out):
                x = inp[0]
                if x.dim() == 2: x = x.unsqueeze(0)
                B, N, D = x.shape
                energies.append(x.norm(dim=-1).squeeze(0).detach().cpu())
                gate = torch.sigmoid(module.qkvg_proj(x).chunk(4, dim=-1)[3])
                gate_vals.append(gate.mean(dim=-1).squeeze(0).detach().cpu())
            return hook

        hooks = []
        for i, layer in enumerate(model.layers):
            hooks.append(layer.attention.register_forward_hook(make_hook(i)))

        with torch.no_grad():
            logits, _ = model(input_ids)

        for h in hooks: h.remove()

        print(f"\n  \"{sentence}\"")
        print(f"  {'Token':<20}", end="")
        for l in range(len(energies)):
            print(f" {'L'+str(l+1):>7}", end="")
        print(f" {'Change':>8}")

        for t_idx, token in enumerate(tokens):
            if t_idx >= energies[0].shape[0]: break
            print(f"  {token[:18]:<20}", end="")
            first_e, last_e = None, None
            for l in range(len(energies)):
                if t_idx < energies[l].shape[0]:
                    e = energies[l][t_idx].item()
                    if first_e is None: first_e = e
                    last_e = e
                    print(f" {e:>7.2f}", end="")
            if first_e and last_e:
                change = ((last_e - first_e) / max(first_e, 1e-8)) * 100
                print(f" {change:>+7.1f}%")
            else:
                print()

        # Gate analysis
        print(f"\n  Gate activations:")
        print(f"  {'Token':<20}", end="")
        for l in range(len(gate_vals)):
            print(f" {'L'+str(l+1):>7}", end="")
        print()
        for t_idx, token in enumerate(tokens):
            if t_idx >= gate_vals[0].shape[0]: break
            print(f"  {token[:18]:<20}", end="")
            for l in range(len(gate_vals)):
                if t_idx < gate_vals[l].shape[0]:
                    g = gate_vals[l][t_idx].item()
                    print(f" {g:>7.3f}", end="")
            print()

    # ============================================================
    # 5. INTERFERENCE ANALYSIS
    # ============================================================
    print(f"\n{'='*65}")
    print("  5. INTERFERENCE ANALYSIS")
    print("  Amplification vs suppression patterns")
    print(f"{'='*65}")

    print(f"\n  Interference temperatures:")
    for i, im in enumerate(model.interference_modules):
        raw_t = im.interference_temperature.item()
        eff_t = F.softplus(torch.tensor(raw_t)).item() + 0.05
        sharp = "SHARP" if eff_t < 0.3 else "MEDIUM" if eff_t < 0.8 else "SOFT"
        print(f"    IM{i+1}: raw={raw_t:.4f} → eff={eff_t:.4f} ({sharp})")

    for sentence in test_sentences[:3]:
        ids = tok.encode(sentence)
        tokens = [tok.decode([i]) for i in ids]
        input_ids = torch.tensor([ids], device=device)

        intf_data = []

        def make_intf_hook(idx):
            def hook(module, inp, out):
                x = inp[0]
                B, N, D = x.shape
                compressed = module.compress(x)
                cumsum = torch.cumsum(compressed, dim=1)
                counts = torch.arange(1, N+1, device=x.device, dtype=x.dtype).view(1,-1,1)
                global_ctx = module.expand(cumsum / counts)
                local_ph = F.normalize(module.local_phase_proj(x), dim=-1)
                global_ph = F.normalize(module.global_phase_proj(global_ctx), dim=-1)
                align = (local_ph * global_ph).sum(dim=-1, keepdim=True)
                temp = F.softplus(module.interference_temperature) + 0.05
                strength = torch.sigmoid(align / temp).squeeze(-1).squeeze(0)
                intf_data.append(strength.detach().cpu())
            return hook

        hooks = []
        for i, im in enumerate(model.interference_modules):
            hooks.append(im.register_forward_hook(make_intf_hook(i)))
        with torch.no_grad():
            model(input_ids)
        for h in hooks: h.remove()

        print(f"\n  \"{sentence}\"")
        if intf_data:
            print(f"  {'Token':<20}", end="")
            for i in range(len(intf_data)):
                print(f" {'IM'+str(i+1):>10}", end="")
            print(f"  Effect")
            for t_idx, token in enumerate(tokens):
                if t_idx >= intf_data[0].shape[0]: break
                print(f"  {token[:18]:<20}", end="")
                avg_s = 0; cnt = 0
                for im_idx in range(len(intf_data)):
                    if t_idx < intf_data[im_idx].shape[0]:
                        s = intf_data[im_idx][t_idx].item()
                        avg_s += s; cnt += 1
                        bar = "▓" * int(s * 10) + "░" * (10 - int(s * 10))
                        print(f" {s:.3f} {bar}", end="")
                if cnt > 0:
                    avg = avg_s / cnt
                    lbl = "AMPLIFIED" if avg > 0.55 else "SUPPRESSED" if avg < 0.45 else "NEUTRAL"
                    print(f"  {lbl}")
                else:
                    print()

    # ============================================================
    # 6. PREDICTION CONFIDENCE + ERROR ANALYSIS
    # ============================================================
    print(f"\n{'='*65}")
    print("  6. PREDICTION ANALYSIS")
    print("  Where the model is confident vs uncertain")
    print(f"{'='*65}")

    for sentence in test_sentences[:3]:
        ids = tok.encode(sentence)
        tokens = [tok.decode([i]) for i in ids]
        input_ids = torch.tensor([ids], device=device)

        with torch.no_grad():
            logits, _ = model(input_ids)

        probs = F.softmax(logits[0], dim=-1)
        max_p = probs.max(dim=-1)
        entropies = -(probs * probs.clamp(min=1e-10).log()).sum(dim=-1)

        print(f"\n  \"{sentence}\"")
        print(f"  {'Pos':<5} {'Token':<20} {'Conf':>8} {'Entropy':>8} {'Predicts':>18}")

        for t_idx in range(min(len(tokens), logits.shape[1])):
            conf = max_p.values[t_idx].item()
            ent = entropies[t_idx].item()
            pred_id = max_p.indices[t_idx].item()
            pred_tok = tok.decode([pred_id])
            bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
            print(f"  {t_idx:<5} {tokens[t_idx][:18]:<20} {conf:>7.1%} {ent:>8.2f} {pred_tok[:16]:>18}  {bar}")

        avg_ent = entropies[:len(tokens)].mean().item()
        avg_conf = max_p.values[:len(tokens)].mean().item()
        print(f"  Avg confidence: {avg_conf:.1%} | Avg entropy: {avg_ent:.2f} (max possible: {math.log(vocab_size):.2f})")

    # ============================================================
    # 7. SCATTER/GATHER COLLISION ANALYSIS
    # ============================================================
    print(f"\n{'='*65}")
    print("  7. SCATTER/GATHER COLLISION ANALYSIS")
    print("  Do BPE tokens collide on the field?")
    print(f"{'='*65}")

    for sentence in test_sentences[:3]:
        ids = tok.encode(sentence)
        tokens = [tok.decode([i]) for i in ids]
        N = len(ids)

        seq_pos = torch.arange(N, dtype=torch.float32)
        stride = (field_size - 1) / max(max_seq_len - 1, 1)
        fp = (seq_pos * stride).clamp(0, field_size - 2)

        idx_lo = fp.long().clamp(0, field_size - 2)

        bins_used = set()
        collisions = 0
        for i in range(N):
            lo = idx_lo[i].item()
            hi = lo + 1
            if lo in bins_used or hi in bins_used:
                collisions += 1
            bins_used.add(lo)
            bins_used.add(hi)

        print(f"\n  \"{sentence}\" ({N} tokens)")
        print(f"    Unique field bins used: {len(bins_used)}/{field_size}")
        print(f"    Bilinear overlaps: {collisions}")
        if stride < 2.0:
            print(f"    WARNING: stride={stride:.2f} < 2.0 — tokens share field bins!")
            print(f"    This causes information MIXING between adjacent tokens")

    # ============================================================
    # 8. GAP DIAGNOSIS — ROOT CAUSE
    # ============================================================
    print(f"\n{'='*65}")
    print("  8. GAP DIAGNOSIS — WHY Wave PPL 170.7 vs Standard 91.4")
    print(f"{'='*65}")

    stride = (field_size - 1) / max(max_seq_len - 1, 1)
    tokens_per_bin = 1.0 / stride if stride > 0 else float('inf')

    diagnosis = []

    # Check 1: Field resolution
    if stride < 2.0:
        diagnosis.append(("FIELD RESOLUTION", "CRITICAL",
            f"Stride={stride:.2f} means ~{tokens_per_bin:.1f} tokens share same bins. "
            f"Adjacent tokens COLLIDE on the field, mixing their information. "
            f"Fix: increase field_size to {int(max_seq_len * 4)} or reduce max_seq_len."))
    elif stride < 4.0:
        diagnosis.append(("FIELD RESOLUTION", "WARNING",
            f"Stride={stride:.2f} — tight but workable. "
            f"Wave kernels have limited room between tokens."))
    else:
        diagnosis.append(("FIELD RESOLUTION", "OK",
            f"Stride={stride:.2f} — good spacing between tokens."))

    # Check 2: Kernel range vs token spacing
    short_range_count = sum(1 for r in all_ranges if r < stride * 3)
    if short_range_count > len(all_ranges) * 0.5:
        diagnosis.append(("KERNEL RANGE", "CRITICAL",
            f"{short_range_count}/{len(all_ranges)} heads have range < {stride*3:.0f} "
            f"(3x token spacing). Many heads can't reach neighboring tokens!"))
    else:
        diagnosis.append(("KERNEL RANGE", "OK",
            f"Most heads have sufficient range to reach neighbors."))

    # Check 3: Vocab pressure
    vocab_ratio = vocab_size / 256  # vs embedding dim
    diagnosis.append(("VOCAB PRESSURE", "WARNING" if vocab_ratio > 20 else "OK",
        f"8000 vocab / 256 embedding = {vocab_ratio:.0f}x ratio. "
        f"Each embedding dimension must distinguish {vocab_ratio:.0f} tokens. "
        f"Standard Transformer handles this with O(n^2) direct token-to-token attention. "
        f"Wave Field routes through field intermediary — harder with large vocab."))

    # Check 4: Training epochs
    diagnosis.append(("TRAINING", "WARNING",
        f"Only 30 epochs. Wave Field converges slower (more params, complex arch). "
        f"Train loss still dropping at epoch 30. More epochs may close the gap."))

    # Check 5: Field size vs seq length
    field_ratio = field_size / max_seq_len
    if field_ratio < 4:
        diagnosis.append(("FIELD/SEQ RATIO", "WARNING",
            f"field_size/max_seq_len = {field_ratio:.1f}x. "
            f"Recommend 4-8x for sufficient wave propagation room."))
    else:
        diagnosis.append(("FIELD/SEQ RATIO", "OK",
            f"field_size/max_seq_len = {field_ratio:.1f}x — adequate."))

    for name, severity, desc in diagnosis:
        icon = "!!!" if severity == "CRITICAL" else " ! " if severity == "WARNING" else " ✓ "
        print(f"\n  [{icon}] {name} ({severity})")
        print(f"      {desc}")

    critical = [d for d in diagnosis if d[1] == "CRITICAL"]
    warnings = [d for d in diagnosis if d[1] == "WARNING"]

    print(f"\n  {'='*55}")
    print(f"  DIAGNOSIS SUMMARY")
    print(f"  {'='*55}")
    print(f"  Critical issues: {len(critical)}")
    print(f"  Warnings: {len(warnings)}")

    if critical:
        print(f"\n  TOP PRIORITY FIXES:")
        for i, (name, _, desc) in enumerate(critical, 1):
            print(f"    {i}. {name}: {desc.split('.')[0]}.")

    if warnings:
        print(f"\n  SECONDARY IMPROVEMENTS:")
        for i, (name, _, desc) in enumerate(warnings, 1):
            print(f"    {i}. {name}: {desc.split('.')[0]}.")

    print(f"\n  RECOMMENDED NEXT STEPS:")
    if any(d[0] == "FIELD RESOLUTION" and d[1] == "CRITICAL" for d in diagnosis):
        print(f"    1. INCREASE field_size from {field_size} to {int(max_seq_len * 4)}")
        print(f"       This is the #1 bottleneck — tokens are colliding on the field")
    print(f"    2. Train for 60+ epochs (Wave Field was still improving at epoch 30)")
    print(f"    3. Increase embedding_dim to 384 or 512 (helps with 8K vocab pressure)")
    print(f"    4. Try field_size=2048 with max_seq_len=256 (8x ratio)")

    print(f"\n{'='*65}")
    print("  DIAGNOSTICS COMPLETE")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
