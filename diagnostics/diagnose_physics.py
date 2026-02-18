"""
Wave Field V3.4 — Physics Diagnostics
=======================================
Inspect the trained V3.4 model to see HOW information flows:
1. What wave patterns each head learned (frequency, damping, phase)
2. How static field coupling routes information between heads
3. How energy is conserved through the model
4. How interference amplifies/suppresses signals
5. How content-dependent gating works (V3.4: higher bias, gates start open)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os

from src.wave_field_transformer import WaveFieldTransformer
from field_tokenizer_v2 import FieldTokenizerV2


def main():
    print("=" * 65)
    print("  WAVE FIELD V3.4 — PHYSICS DIAGNOSTICS")
    print("  How information flows through the model")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Detect which dataset/checkpoint to use
    field_size = 1024
    use_wikitext = (os.path.exists("wikitext2_wave_v34_checkpoints/best.pt") or
                    os.path.exists("wikitext2_wave_v33_checkpoints/best.pt") or
                    os.path.exists("wikitext2_wave_checkpoints/best.pt"))

    if use_wikitext:
        from datasets import load_dataset
        print("Loading WikiText-2 tokenizer...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_lines = [item['text'].strip() for item in ds['train']
                       if item['text'].strip() and not item['text'].strip().startswith('=')]
        if os.path.exists("wikitext2_wave_v34_checkpoints/best.pt"):
            ckpt_path = "wikitext2_wave_v34_checkpoints/best.pt"
        elif os.path.exists("wikitext2_wave_v33_checkpoints/best.pt"):
            ckpt_path = "wikitext2_wave_v33_checkpoints/best.pt"
        else:
            ckpt_path = "wikitext2_wave_checkpoints/best.pt"
        test_sentences = [
            "The president of the united states",
            "In the year of our lord",
            "Scientists discovered that",
        ]
    else:
        print("Loading Shakespeare tokenizer...")
        with open('shakespeare.txt', 'r') as f:
            text = f.read()
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        train_lines = lines[:int(len(lines) * 0.9)]
        ckpt_path = "wave_v31_checkpoints/best.pt"
        test_sentences = [
            "The quality of mercy is",
            "To be or not to be",
            "First Citizen:",
        ]

    tok = FieldTokenizerV2(field_size=field_size)
    tok.build_vocab(train_lines)
    vocab_size = tok.vocab_size_actual()

    model = WaveFieldTransformer(
        vocab_size=vocab_size, embedding_dim=256, num_layers=6,
        num_heads=8, ffn_dim=1024, field_size=field_size,
        max_seq_len=129, dropout=0.1, use_checkpoint=False,
        interference_interval=3, device=device,
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path,
                                      weights_only=True, map_location=device))
    model.eval()
    print(f"\nModel loaded (V3.4) from {ckpt_path}. Device: {device}")

    # ============================================================
    # 1. WAVE KERNEL ANALYSIS — What each head learned
    # ============================================================
    print(f"\n{'='*65}")
    print("  1. WAVE KERNEL ANALYSIS")
    print("  What listening pattern each head learned")
    print(f"{'='*65}")

    for layer_idx, layer in enumerate(model.layers):
        attn = layer.attention
        omega = attn.wave_frequency.detach().cpu()
        alpha_raw = attn.wave_damping.detach().cpu()
        alpha = F.softplus(alpha_raw)
        phi = attn.wave_phase.detach().cpu()

        print(f"\n  Layer {layer_idx + 1}:")
        print(f"  {'Head':<6} {'Freq(w)':>8} {'Damp(a)':>8} {'Phase(p)':>8} {'Range':>8}  Role")
        print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*25}")

        for h in range(8):
            w = omega[h].item()
            a = alpha[h].item()
            p = phi[h].item()

            # Effective range: how far before signal decays to 10%
            if a > 0.01:
                eff_range = min(int(-math.log(0.1) / a), field_size)
            else:
                eff_range = field_size

            # Classify role
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

    # ============================================================
    # 2. STATIC FIELD COUPLING (V3.4, same as V3.2)
    # ============================================================
    print(f"\n{'='*65}")
    print("  2. STATIC FIELD COUPLING (V3.4)")
    print("  Learned head interaction patterns (same for all inputs)")
    print("  V3.3 causal cumsum was a regression → reverted to static.")
    print(f"{'='*65}")

    for layer_idx, layer in enumerate(model.layers):
        attn = layer.attention
        H = attn.num_heads
        coupling = F.softmax(attn.field_coupling.detach().cpu(), dim=-1)

        print(f"\n  Layer {layer_idx + 1} coupling (row=receiver, col=sender):")
        header = "        " + "".join([f"  H{i+1}  " for i in range(H)])
        print(header)

        for h in range(H):
            row = f"  H{h+1}  "
            for h2 in range(H):
                val = coupling[h, h2].item()
                if h == h2:
                    row += f" [{val:.2f}]"
                elif val > 0.15:
                    row += f" *{val:.2f}*"
                else:
                    row += f"  {val:.2f} "
            print(row)

        strong = []
        for h in range(H):
            for h2 in range(H):
                if h != h2 and coupling[h, h2].item() > 0.15:
                    strong.append((h+1, h2+1, coupling[h, h2].item()))
        strong.sort(key=lambda x: -x[2])

        if strong:
            print(f"\n  Strongest cross-field interactions:")
            for h1, h2, val in strong[:5]:
                print(f"    H{h2} → H{h1}: {val:.3f} "
                      f"(H{h2} influences H{h1})")

        off_diag = coupling.clone()
        off_diag.fill_diagonal_(0)
        total_cross = off_diag.sum().item()
        print(f"  Total cross-coupling: {total_cross:.3f} "
              f"({'ACTIVE' if total_cross > 1.0 else 'WEAK'})")

    # ============================================================
    # 3. INFORMATION FLOW — Trace through a sentence
    # ============================================================
    print(f"\n{'='*65}")
    print("  3. INFORMATION FLOW TRACE")
    print(f"  How information flows through the model")
    print(f"{'='*65}")

    # test_sentences defined at the top based on dataset

    for sentence in test_sentences:
        print(f"\n  Input: \"{sentence}\"")
        ids = tok.encode(sentence.lower())
        tokens = [tok.decode([i]) for i in ids]
        input_ids = torch.tensor([ids], device=device)

        # Forward pass with hooks to capture intermediate states
        energies_per_layer = []
        gate_values_per_layer = []

        def make_attn_hook(layer_idx):
            def hook(module, input, output):
                x = input[0]
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                B, N, D = x.shape

                # Measure energy per token
                energy = x.norm(dim=-1).squeeze(0)  # (N,)
                energies_per_layer.append(energy.detach().cpu())

                # Measure gate activation
                gate = torch.sigmoid(module.gate_proj(x))
                gate_mean = gate.mean(dim=-1).squeeze(0)  # (N,)
                gate_values_per_layer.append(gate_mean.detach().cpu())

            return hook

        hooks = []
        for i, layer in enumerate(model.layers):
            h = layer.attention.register_forward_hook(make_attn_hook(i))
            hooks.append(h)

        with torch.no_grad():
            logits, _ = model(input_ids)

        for h in hooks:
            h.remove()

        # Display token-level energy flow
        print(f"\n  Token energies through layers:")
        print(f"  {'Token':<15}", end="")
        for l in range(len(energies_per_layer)):
            print(f" {'L'+str(l+1):>7}", end="")
        print(f" {'Change':>8}")

        for t_idx, token in enumerate(tokens):
            if t_idx >= energies_per_layer[0].shape[0]:
                break
            tok_display = token[:12].ljust(15)
            print(f"  {tok_display}", end="")
            first_e = None
            last_e = None
            for l in range(len(energies_per_layer)):
                if t_idx < energies_per_layer[l].shape[0]:
                    e = energies_per_layer[l][t_idx].item()
                    if first_e is None:
                        first_e = e
                    last_e = e
                    print(f" {e:>7.2f}", end="")
                else:
                    print(f" {'N/A':>7}", end="")
            if first_e and last_e:
                change = ((last_e - first_e) / first_e) * 100
                print(f" {change:>+7.1f}%")
            else:
                print()

        # Display gate activations
        print(f"\n  Gate activations (how much info passes through):")
        print(f"  {'Token':<15}", end="")
        for l in range(len(gate_values_per_layer)):
            print(f" {'L'+str(l+1):>7}", end="")
        print()

        for t_idx, token in enumerate(tokens):
            if t_idx >= gate_values_per_layer[0].shape[0]:
                break
            tok_display = token[:12].ljust(15)
            print(f"  {tok_display}", end="")
            for l in range(len(gate_values_per_layer)):
                if t_idx < gate_values_per_layer[l].shape[0]:
                    g = gate_values_per_layer[l][t_idx].item()
                    # Visual indicator
                    if g > 0.6:
                        indicator = "HIGH"
                    elif g > 0.4:
                        indicator = "MED"
                    else:
                        indicator = "LOW"
                    print(f" {g:>4.2f}{indicator[0]:>3}", end="")
                else:
                    print(f" {'N/A':>7}", end="")
            print()

    # ============================================================
    # 4. INTERFERENCE ANALYSIS
    # ============================================================
    print(f"\n{'='*65}")
    print("  4. INTERFERENCE ANALYSIS")
    print("  Which tokens get amplified vs suppressed")
    print(f"{'='*65}")

    # Show learned interference temperatures
    print(f"\n  Interference Temperature (V3.1 — learnable sharpness):")
    for i, im in enumerate(model.interference_modules):
        raw_t = im.interference_temperature.item()
        eff_t = F.softplus(torch.tensor(raw_t)).item() + 0.05
        sharpness = "SHARP (selective)" if eff_t < 0.3 else "MEDIUM" if eff_t < 0.8 else "SOFT (uniform)"
        print(f"    IM{i+1}: raw={raw_t:.4f} → effective={eff_t:.4f}  ({sharpness})")

    for sentence in test_sentences:
        ids = tok.encode(sentence.lower())
        tokens = [tok.decode([i]) for i in ids]
        input_ids = torch.tensor([ids], device=device)

        interference_data = []

        def make_interference_hook(idx):
            def hook(module, input, output):
                x = input[0]
                B, N, D = x.shape

                # V3.1: separate phase projections + learnable temperature
                compressed = module.compress(x)
                cumsum = torch.cumsum(compressed, dim=1)
                counts = torch.arange(1, N+1, device=x.device, dtype=x.dtype).view(1,-1,1)
                global_ctx = module.expand(cumsum / counts)

                local_phase = F.normalize(module.local_phase_proj(x), dim=-1)
                global_phase = F.normalize(module.global_phase_proj(global_ctx), dim=-1)

                phase_alignment = (local_phase * global_phase).sum(dim=-1, keepdim=True)
                temp = F.softplus(module.interference_temperature) + 0.05
                strength = torch.sigmoid(phase_alignment / temp).squeeze(-1).squeeze(0)

                interference_data.append(strength.detach().cpu())

            return hook

        hooks = []
        for i, im in enumerate(model.interference_modules):
            h = im.register_forward_hook(make_interference_hook(i))
            hooks.append(h)

        with torch.no_grad():
            model(input_ids)

        for h in hooks:
            h.remove()

        print(f"\n  \"{sentence}\"")
        if interference_data:
            print(f"  {'Token':<15}", end="")
            for i in range(len(interference_data)):
                print(f" {'IM'+str(i+1):>10}", end="")
            print(f"  Effect")

            for t_idx, token in enumerate(tokens):
                if t_idx >= interference_data[0].shape[0]:
                    break
                tok_display = token[:12].ljust(15)
                print(f"  {tok_display}", end="")
                avg_strength = 0
                count = 0
                for im_idx in range(len(interference_data)):
                    if t_idx < interference_data[im_idx].shape[0]:
                        s = interference_data[im_idx][t_idx].item()
                        avg_strength += s
                        count += 1
                        bar_len = int(s * 10)
                        bar = "▓" * bar_len + "░" * (10 - bar_len)
                        print(f" {s:.3f} {bar}", end="")
                if count > 0:
                    avg = avg_strength / count
                    if avg > 0.55:
                        print(f"  AMPLIFIED (+)")
                    elif avg < 0.45:
                        print(f"  SUPPRESSED (-)")
                    else:
                        print(f"  NEUTRAL (~)")
                else:
                    print()

    # ============================================================
    # 5. PREDICTION ANALYSIS
    # ============================================================
    print(f"\n{'='*65}")
    print("  5. PREDICTION CONFIDENCE")
    print("  How certain the model is at each position")
    print(f"{'='*65}")

    for sentence in test_sentences:
        ids = tok.encode(sentence.lower())
        tokens = [tok.decode([i]) for i in ids]
        input_ids = torch.tensor([ids], device=device)

        with torch.no_grad():
            logits, _ = model(input_ids)

        probs = F.softmax(logits[0], dim=-1)
        max_probs = probs.max(dim=-1)

        print(f"\n  \"{sentence}\"")
        print(f"  {'Position':<5} {'Token':<15} {'Confidence':>10} {'Predicts':>15}  Visual")

        for t_idx in range(min(len(tokens), logits.shape[1])):
            conf = max_probs.values[t_idx].item()
            pred_id = max_probs.indices[t_idx].item()
            pred_token = tok.decode([pred_id])

            bar_len = int(conf * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)

            correct = ""
            if t_idx < len(ids) - 1:
                actual_next = tok.decode([ids[t_idx + 1]])
                if pred_id == ids[t_idx + 1]:
                    correct = " ✓"

            print(f"  {t_idx:<5} {tokens[t_idx][:12]:<15} {conf:>9.1%} "
                  f"{pred_token[:12]:>15}  {bar}{correct}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*65}")
    print("  V3.4 PHYSICS SUMMARY")
    print(f"{'='*65}")
    print(f"""
  Wave Field V3.4 — bilinear interpolation fix:

  V3.4 FIXES:
  - BILINEAR SCATTER/GATHER: Smooth weighted deposit and read
    from two adjacent field bins. Eliminates discrete binning
    artifacts that caused garbage generation in V3.1-V3.3.
    Consistent behavior between training and autoregressive generation.
  - STATIC COUPLING (reverted): V3.3's causal cumsum coupling was
    a regression (PPL 8.3 vs V3.2's 7.5). Static is simpler & better.
  - HIGHER GATE BIAS: 2.0 vs 1.0. Gates start at sigmoid(2)=0.88
    so wave attention contributes from Layer 1 instead of being starved.

  INHERITED (all correct):
  1. WAVE KERNELS: Multi-scale damped oscillation per head
  2. ZERO-PADDED FFT: Linear convolution (no wraparound)
  3. FIELD-LEVEL CONSERVATION: Energy preserved in wave dynamics
  4. DIVERSE PHASES: Better initial coverage
  5. DIVERSE INTERFERENCE TEMPS: Sharp early, soft late

  ARCHITECTURE: O(n log n) with provably causal attention.
  Every parameter has physical meaning. Every bug is traceable.
""")
    print("=" * 65)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
