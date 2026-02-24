"""
Wave Field LLM — Training Monitor
===================================
Opens the black box. Tracks what the model actually learns inside:

  KERNEL SPACE:     What frequencies/dampings each head learns, diversity, collapse
  FEATURE MAPS:     Are they saturating? Are they diverse across heads?
  SPECTRAL GATE:    Is it active? What frequency profile does it modulate?
  FIELD DYNAMICS:   How sparse is the field? What's the effective rank of outputs?
  WEIGHT/GRADIENT:  Which params are moving? Which are dead?

Usage:
    from diagnostics.training_monitor import WaveFieldMonitor

    monitor = WaveFieldMonitor(model, log_dir='results/monitor')
    for step, (x, y) in enumerate(batches):
        loss = train_step(model, x, y)
        monitor.step(step, loss.item())  # lightweight per-step
        if step % 100 == 0:
            monitor.snapshot(step)       # full diagnostic (heavier)
    monitor.save_report()
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import math
from collections import defaultdict


class WaveFieldMonitor:
    """Non-invasive training monitor for Wave Field LLM.

    Reads parameter values and hooks into forward pass to capture
    intermediate states. Does NOT modify gradients or training.
    """

    def __init__(self, model, log_dir='results/monitor', device='cuda'):
        self.model = model
        self.log_dir = log_dir
        self.device = device
        os.makedirs(log_dir, exist_ok=True)

        # Per-step lightweight logs
        self.step_log = []

        # Periodic full snapshots
        self.snapshots = []

        # Find all attention layers
        self.attn_layers = []
        for name, module in model.named_modules():
            cls_name = module.__class__.__name__
            if cls_name == 'WaveFieldAttention':
                self.attn_layers.append((name, module))

        print(f"  [Monitor] Found {len(self.attn_layers)} WaveFieldAttention layers")

        # Forward hook storage (populated during snapshot)
        self._hook_data = {}
        self._hooks = []

    # ==================================================================
    # PER-STEP: Lightweight logging (< 1ms overhead)
    # ==================================================================

    def step(self, step_num, loss, lr=None):
        """Call every training step. Logs loss + kernel param summary."""
        entry = {'step': step_num, 'loss': loss}
        if lr is not None:
            entry['lr'] = lr

        # Quick kernel param summary (no computation, just read params)
        for i, (name, attn) in enumerate(self.attn_layers):
            prefix = f'L{i}'
            with torch.no_grad():
                # Kernel params: frequency, damping, phase
                freq = attn.wave_frequency.data
                damp = F.softplus(attn.wave_damping.data)
                phase = attn.wave_phase.data

                entry[f'{prefix}_freq_mean'] = freq.mean().item()
                entry[f'{prefix}_freq_std'] = freq.std().item()
                entry[f'{prefix}_damp_mean'] = damp.mean().item()
                entry[f'{prefix}_damp_std'] = damp.std().item()

                # Gate bias (should start at 2.0, may drift)
                D = attn.embedding_dim
                gate_bias = attn.qkvg_proj.bias.data[3 * D:]
                entry[f'{prefix}_gate_bias_mean'] = gate_bias.mean().item()

        self.step_log.append(entry)

    # ==================================================================
    # SNAPSHOT: Full diagnostic (call every N steps)
    # ==================================================================

    @torch.no_grad()
    def snapshot(self, step_num, sample_input=None):
        """Full diagnostic snapshot. Heavier but comprehensive.

        sample_input: Optional (B, N) tensor for forward-pass diagnostics.
                      If None, skips field/output analysis.
        """
        snap = {'step': step_num}

        for i, (name, attn) in enumerate(self.attn_layers):
            prefix = f'L{i}'
            layer_snap = {}

            # ---- KERNEL DIAGNOSTICS ----
            layer_snap.update(self._diagnose_kernels(attn, prefix))

            # ---- FEATURE MAP DIAGNOSTICS ----
            layer_snap.update(self._diagnose_feature_maps(attn, prefix))

            # ---- SPECTRAL GATE DIAGNOSTICS ----
            layer_snap.update(self._diagnose_spectral_gate(attn, prefix))

            # ---- WEIGHT NORMS ----
            layer_snap.update(self._diagnose_weights(attn, prefix))

            # ---- GRADIENT NORMS ----
            layer_snap.update(self._diagnose_gradients(attn, prefix))

            snap.update(layer_snap)

        # ---- FORWARD PASS DIAGNOSTICS (if sample provided) ----
        if sample_input is not None:
            snap.update(self._diagnose_forward(sample_input))

        self.snapshots.append(snap)
        return snap

    # ==================================================================
    # KERNEL DIAGNOSTICS
    # ==================================================================

    def _diagnose_kernels(self, attn, prefix):
        """What frequencies/dampings has each head learned?
        Are they diverse or collapsed to same values?"""
        diag = {}
        H = attn.num_heads
        G = attn.field_size

        freq = attn.wave_frequency.data.cpu()      # (H,) or (H, C)
        damp_raw = attn.wave_damping.data.cpu()
        damp = F.softplus(damp_raw)                 # positive
        phase = attn.wave_phase.data.cpu()

        is_multi = freq.dim() == 2

        if not is_multi:
            # Single component: per-head values
            diag[f'{prefix}_kernel_freq'] = freq.tolist()
            diag[f'{prefix}_kernel_damp'] = damp.tolist()
            diag[f'{prefix}_kernel_phase'] = phase.tolist()

            # Diversity: pairwise distance between head frequencies
            freq_dists = torch.cdist(freq.unsqueeze(1), freq.unsqueeze(1)).squeeze()
            diag[f'{prefix}_kernel_freq_diversity'] = freq_dists.mean().item()

            # Collapse detection: how many heads have ~same frequency (within 5%)
            freq_range = freq.max() - freq.min()
            threshold = max(freq_range * 0.05, 0.1)
            n_collapsed = 0
            for h1 in range(H):
                for h2 in range(h1 + 1, H):
                    if abs(freq[h1] - freq[h2]) < threshold:
                        n_collapsed += 1
            diag[f'{prefix}_kernel_collapsed_pairs'] = n_collapsed

            # Effective wavelength (in field positions)
            wavelength = (2 * math.pi / freq.abs().clamp(min=0.01)).tolist()
            diag[f'{prefix}_kernel_wavelength'] = wavelength

            # Effective reach: 1/damping = how far the kernel reaches
            reach = (1.0 / damp.clamp(min=0.01)).tolist()
            diag[f'{prefix}_kernel_reach'] = reach

        # Spectral entropy of the kernel frequency response
        if attn.use_analytic_kernel and not is_multi:
            kernel_fft = attn._build_analytic_kernel_fft(attn.wave_frequency.device)
        else:
            kernel_fft = attn._build_wave_kernels(attn.wave_frequency.device)

        mag = kernel_fft.abs().cpu()  # (H, freq_bins)
        # Normalize per head to probability distribution
        mag_norm = mag / mag.sum(dim=1, keepdim=True).clamp(min=1e-8)
        # Shannon entropy: -sum(p * log(p))
        entropy = -(mag_norm * (mag_norm + 1e-10).log()).sum(dim=1)
        max_entropy = math.log(mag.shape[1])  # uniform distribution

        diag[f'{prefix}_spectral_entropy'] = entropy.tolist()
        diag[f'{prefix}_spectral_entropy_ratio'] = (entropy / max_entropy).tolist()

        # Effective bandwidth: how many freq bins carry >1% of energy
        energy = mag ** 2
        energy_norm = energy / energy.sum(dim=1, keepdim=True).clamp(min=1e-8)
        active_bins = (energy_norm > 0.01).sum(dim=1)
        diag[f'{prefix}_effective_bandwidth'] = active_bins.tolist()

        # Peak frequency per head
        peak_freq_bin = mag.argmax(dim=1)
        diag[f'{prefix}_peak_freq_bin'] = peak_freq_bin.tolist()

        return diag

    # ==================================================================
    # FEATURE MAP DIAGNOSTICS
    # ==================================================================

    def _diagnose_feature_maps(self, attn, prefix):
        """Are learned feature maps diverse? Saturating? Active?"""
        diag = {}

        for fm_name, fm in [('q_fm', attn.q_feature_map), ('k_fm', attn.k_feature_map)]:
            key = f'{prefix}_{fm_name}'

            # Weight norms per layer in the feature map MLP
            layer_norms = []
            for module in fm.net:
                if hasattr(module, 'weight'):
                    w = module.weight.data
                    layer_norms.append(w.norm().item())

                    # How far from identity? (deviation = learning signal)
                    if w.shape[0] == w.shape[1]:
                        identity = torch.eye(w.shape[0], device=w.device)
                        deviation = (w - identity).norm().item()
                        diag[f'{key}_identity_deviation'] = deviation

            diag[f'{key}_weight_norms'] = layer_norms

            # Probe: pass unit vectors through FM to check output distribution
            d = attn.head_dim
            probe = torch.eye(d, device=attn.wave_frequency.device).unsqueeze(0)  # (1, d, d)
            with torch.no_grad():
                out = fm(probe)  # (1, d, d)
            out = out.squeeze(0)

            # Output statistics
            diag[f'{key}_out_mean'] = out.mean().item()
            diag[f'{key}_out_std'] = out.std().item()
            diag[f'{key}_out_max'] = out.max().item()
            diag[f'{key}_out_min'] = out.min().item()

            # Sparsity: how many outputs are zero (killed by ReLU)?
            dead_frac = (out < 1e-5).float().mean().item()
            diag[f'{key}_dead_fraction'] = dead_frac

            # Effective rank of feature map output matrix
            svd_vals = torch.linalg.svdvals(out.float())
            eff_rank = (svd_vals.sum() ** 2) / (svd_vals ** 2).sum()
            diag[f'{key}_effective_rank'] = eff_rank.item()

        return diag

    # ==================================================================
    # SPECTRAL GATE DIAGNOSTICS
    # ==================================================================

    def _diagnose_spectral_gate(self, attn, prefix):
        """Is the spectral gate active? What's its magnitude and profile?"""
        diag = {}

        if attn.spectral_gate is None:
            diag[f'{prefix}_sg_active'] = False
            return diag

        diag[f'{prefix}_sg_active'] = True
        sg = attn.spectral_gate

        # Output layer weight magnitude (should grow from near-zero init)
        out_weight = sg.net[-1].weight.data
        out_bias = sg.net[-1].bias.data

        diag[f'{prefix}_sg_out_weight_norm'] = out_weight.norm().item()
        diag[f'{prefix}_sg_out_weight_mean_abs'] = out_weight.abs().mean().item()
        diag[f'{prefix}_sg_out_bias_norm'] = out_bias.norm().item()
        diag[f'{prefix}_sg_out_bias_mean'] = out_bias.mean().item()

        # How far has it moved from init (0.01 * normal)?
        diag[f'{prefix}_sg_out_weight_max'] = out_weight.abs().max().item()

        # First layer diagnostics
        first_weight = sg.net[0].weight.data
        diag[f'{prefix}_sg_first_weight_norm'] = first_weight.norm().item()

        return diag

    # ==================================================================
    # WEIGHT NORM DIAGNOSTICS
    # ==================================================================

    def _diagnose_weights(self, attn, prefix):
        """Per-module weight norms — detect growth/decay/stability."""
        diag = {}
        D = attn.embedding_dim

        # QKV+G projection: break into Q, K, V, G rows
        w = attn.qkvg_proj.weight.data
        diag[f'{prefix}_w_Q_norm'] = w[:D].norm().item()
        diag[f'{prefix}_w_K_norm'] = w[D:2*D].norm().item()
        diag[f'{prefix}_w_V_norm'] = w[2*D:3*D].norm().item()
        diag[f'{prefix}_w_G_norm'] = w[3*D:].norm().item()

        # Output projection
        diag[f'{prefix}_w_out_norm'] = attn.out_proj.weight.data.norm().item()

        # Field coupling matrix
        coupling = F.softmax(attn.field_coupling.data, dim=-1)
        # How far from identity? (measures cross-head mixing strength)
        identity = torch.eye(attn.num_heads, device=coupling.device)
        diag[f'{prefix}_coupling_deviation'] = (coupling - identity).norm().item()
        # Entropy of coupling (high = more mixing, low = independent heads)
        coupling_entropy = -(coupling * (coupling + 1e-10).log()).sum(dim=1).mean()
        diag[f'{prefix}_coupling_entropy'] = coupling_entropy.item()

        return diag

    # ==================================================================
    # GRADIENT NORM DIAGNOSTICS
    # ==================================================================

    def _diagnose_gradients(self, attn, prefix):
        """Which params are getting gradients? Which are dead?"""
        diag = {}

        groups = {
            'freq': attn.wave_frequency,
            'damp': attn.wave_damping,
            'phase': attn.wave_phase,
            'qkvg': attn.qkvg_proj.weight,
            'out': attn.out_proj.weight,
            'coupling': attn.field_coupling,
        }

        if attn.spectral_gate is not None:
            groups['sg_out_w'] = attn.spectral_gate.net[-1].weight
            groups['sg_out_b'] = attn.spectral_gate.net[-1].bias

        for gname, param in groups.items():
            if param.grad is not None:
                g = param.grad.data
                diag[f'{prefix}_grad_{gname}_norm'] = g.norm().item()
                diag[f'{prefix}_grad_{gname}_max'] = g.abs().max().item()
                diag[f'{prefix}_grad_{gname}_mean'] = g.abs().mean().item()
                # Dead gradient check
                diag[f'{prefix}_grad_{gname}_zero_frac'] = (g.abs() < 1e-8).float().mean().item()
            else:
                diag[f'{prefix}_grad_{gname}_norm'] = None

        return diag

    # ==================================================================
    # FORWARD PASS DIAGNOSTICS (requires sample input)
    # ==================================================================

    def _diagnose_forward(self, sample_input):
        """Run a forward pass and capture intermediate states."""
        diag = {}

        # Install temporary hooks
        hook_data = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # Capture field state, gate values, attention output
                # output is (B, N, D) from the attention layer
                out = output.detach()

                # Effective rank of attention output
                B, N, D = out.shape
                # Flatten batch: (B*N, D) and take SVD
                flat = out.reshape(-1, D).float()
                if flat.shape[0] > 1000:
                    flat = flat[:1000]  # subsample for speed
                svd_vals = torch.linalg.svdvals(flat)
                eff_rank = (svd_vals.sum() ** 2) / ((svd_vals ** 2).sum() + 1e-10)
                hook_data[f'L{layer_idx}_output_eff_rank'] = eff_rank.item()

                # Output magnitude
                hook_data[f'L{layer_idx}_output_norm'] = out.norm(dim=-1).mean().item()
                hook_data[f'L{layer_idx}_output_std'] = out.std().item()
            return hook_fn

        handles = []
        for i, (name, attn) in enumerate(self.attn_layers):
            h = attn.register_forward_hook(make_hook(i))
            handles.append(h)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            sample = sample_input.to(self.device)
            try:
                self.model(sample)
            except Exception as e:
                diag['forward_error'] = str(e)
        self.model.train()

        # Remove hooks
        for h in handles:
            h.remove()

        diag.update(hook_data)
        return diag

    # ==================================================================
    # REPORTING
    # ==================================================================

    def save_report(self, filename=None):
        """Save all logs to JSON files."""
        if filename is None:
            filename = 'monitor'

        # Step log
        step_path = os.path.join(self.log_dir, f'{filename}_steps.json')
        with open(step_path, 'w') as f:
            json.dump(self.step_log, f, indent=2, default=_json_safe)
        print(f"  [Monitor] Step log saved: {step_path} ({len(self.step_log)} entries)")

        # Snapshots
        snap_path = os.path.join(self.log_dir, f'{filename}_snapshots.json')
        with open(snap_path, 'w') as f:
            json.dump(self.snapshots, f, indent=2, default=_json_safe)
        print(f"  [Monitor] Snapshots saved: {snap_path} ({len(self.snapshots)} entries)")

    def print_snapshot(self, snap=None):
        """Pretty-print the latest (or given) snapshot."""
        if snap is None:
            if not self.snapshots:
                print("  [Monitor] No snapshots yet")
                return
            snap = self.snapshots[-1]

        print(f"\n{'='*70}")
        print(f"  WAVE FIELD MONITOR — Step {snap.get('step', '?')}")
        print(f"{'='*70}")

        n_layers = len(self.attn_layers)

        # Kernel overview
        print(f"\n  KERNEL SPACE (per head)")
        print(f"  {'Layer':<8} {'Freq m+-s':<16} {'Damp m+-s':<16} {'Entropy ratio':<16} {'Bandwidth':<12} {'Collapsed'}")
        print(f"  {'-'*8} {'-'*16} {'-'*16} {'-'*16} {'-'*12} {'-'*10}")
        for i in range(n_layers):
            p = f'L{i}'
            freq = snap.get(f'{p}_kernel_freq', [])
            damp = snap.get(f'{p}_kernel_damp', [])
            ent = snap.get(f'{p}_spectral_entropy_ratio', [])
            bw = snap.get(f'{p}_effective_bandwidth', [])
            coll = snap.get(f'{p}_kernel_collapsed_pairs', '?')

            freq_s = f"{np.mean(freq):.2f}+-{np.std(freq):.2f}" if freq else "?"
            damp_s = f"{np.mean(damp):.2f}+-{np.std(damp):.2f}" if damp else "?"
            ent_s = f"{np.mean(ent):.3f}" if ent else "?"
            bw_s = f"{np.mean(bw):.0f}/{len(bw)*64 if bw else '?'}" if bw else "?"

            print(f"  L{i:<7} {freq_s:<16} {damp_s:<16} {ent_s:<16} {bw_s:<12} {coll}")

        # Feature map overview
        print(f"\n  FEATURE MAPS")
        print(f"  {'Layer':<8} {'Q dev':<10} {'Q dead%':<10} {'Q rank':<10} {'K dev':<10} {'K dead%':<10} {'K rank'}")
        print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for i in range(n_layers):
            p = f'L{i}'
            q_dev = snap.get(f'{p}_q_fm_identity_deviation', '?')
            q_dead = snap.get(f'{p}_q_fm_dead_fraction', '?')
            q_rank = snap.get(f'{p}_q_fm_effective_rank', '?')
            k_dev = snap.get(f'{p}_k_fm_identity_deviation', '?')
            k_dead = snap.get(f'{p}_k_fm_dead_fraction', '?')
            k_rank = snap.get(f'{p}_k_fm_effective_rank', '?')

            q_dev_s = f"{q_dev:.4f}" if isinstance(q_dev, float) else str(q_dev)
            q_dead_s = f"{q_dead*100:.1f}%" if isinstance(q_dead, float) else str(q_dead)
            q_rank_s = f"{q_rank:.1f}" if isinstance(q_rank, float) else str(q_rank)
            k_dev_s = f"{k_dev:.4f}" if isinstance(k_dev, float) else str(k_dev)
            k_dead_s = f"{k_dead*100:.1f}%" if isinstance(k_dead, float) else str(k_dead)
            k_rank_s = f"{k_rank:.1f}" if isinstance(k_rank, float) else str(k_rank)

            print(f"  L{i:<7} {q_dev_s:<10} {q_dead_s:<10} {q_rank_s:<10} {k_dev_s:<10} {k_dead_s:<10} {k_rank_s}")

        # Spectral gate overview
        print(f"\n  SPECTRAL GATE")
        print(f"  {'Layer':<8} {'Active':<8} {'Out W norm':<12} {'Out W |max|':<12} {'Out bias μ'}")
        print(f"  {'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
        for i in range(n_layers):
            p = f'L{i}'
            active = snap.get(f'{p}_sg_active', False)
            w_norm = snap.get(f'{p}_sg_out_weight_norm', '?')
            w_max = snap.get(f'{p}_sg_out_weight_max', '?')
            b_mean = snap.get(f'{p}_sg_out_bias_mean', '?')

            w_norm_s = f"{w_norm:.4f}" if isinstance(w_norm, float) else str(w_norm)
            w_max_s = f"{w_max:.4f}" if isinstance(w_max, float) else str(w_max)
            b_mean_s = f"{b_mean:.4f}" if isinstance(b_mean, float) else str(b_mean)

            print(f"  L{i:<7} {'YES' if active else 'NO':<8} {w_norm_s:<12} {w_max_s:<12} {b_mean_s}")

        # Gradient overview
        print(f"\n  GRADIENT NORMS (layer 0)")
        grad_keys = [k for k in snap if k.startswith('L0_grad_') and k.endswith('_norm')]
        for k in sorted(grad_keys):
            v = snap[k]
            param = k.replace('L0_grad_', '').replace('_norm', '')
            v_s = f"{v:.6f}" if isinstance(v, float) else "None (no grad)"
            print(f"  {param:<20} {v_s}")

        # Forward pass diagnostics
        if any(k.startswith('L0_output') for k in snap):
            print(f"\n  OUTPUT EFFECTIVE RANK")
            for i in range(n_layers):
                rank = snap.get(f'L{i}_output_eff_rank', '?')
                norm = snap.get(f'L{i}_output_norm', '?')
                rank_s = f"{rank:.1f}" if isinstance(rank, float) else str(rank)
                norm_s = f"{norm:.4f}" if isinstance(norm, float) else str(norm)
                print(f"  L{i}: rank={rank_s}  norm={norm_s}")

        print(f"{'='*70}\n")


def _json_safe(obj):
    """JSON serializer for non-standard types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return str(obj)


# ==================================================================
# STANDALONE: Run diagnostics on a saved model
# ==================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.wave_field_transformer import WaveFieldTransformer

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create a small model for testing
    model = WaveFieldTransformer(
        vocab_size=8000,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        ffn_dim=1024,
        field_size=512,
        max_seq_len=128,
        dropout=0.0,
        use_checkpoint=False,
        interference_interval=3,
        n_components=1,
        local_window=0,
        device=device,
        use_analytic_kernel=True,
        feature_map_depth=2,
        use_write_gate=False,
        use_3d_interference=False,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}")

    monitor = WaveFieldMonitor(model, log_dir='results/monitor_test', device=device)

    # Step 0: Initial state
    monitor.step(0, loss=9.0)

    # Snapshot: Full diagnostics
    sample = torch.randint(0, 8000, (2, 64), device=device)
    snap = monitor.snapshot(0, sample_input=sample)
    monitor.print_snapshot(snap)

    # Real training loop with gradient capture
    from datasets import load_dataset
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    print("\n  Loading WikiText-2 for real training...")
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train',
                      trust_remote_code=True)
    raw_text = '\n'.join([t for t in ds['text'] if len(t.strip()) > 50])

    # Byte-level BPE tokenizer (same as benchmarks)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=8000, special_tokens=["<pad>"])
    tok.train_from_iterator([raw_text[:500_000]], trainer=trainer)

    # Encode all text
    enc = tok.encode(raw_text)
    all_ids = torch.tensor(enc.ids, dtype=torch.long)
    print(f"  Tokens: {len(all_ids):,}")

    # Training config
    SEQ_LEN = 128
    BATCH = 16
    TOTAL_STEPS = 200
    SNAPSHOT_EVERY = 50
    LR = 3e-4

    # Use configure_optimizer for proper per-group LR (kernel 50x, qkvg 3x)
    if hasattr(model, 'configure_optimizer'):
        optimizer = model.configure_optimizer(base_lr=LR, kernel_lr_mult=50.0)
        print(f"  Optimizer: configure_optimizer (kernel LR={LR*50:.4f}, qkvg LR={LR*3:.4f})")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        print(f"  Optimizer: AdamW (flat LR={LR})")

    print(f"  Training: {TOTAL_STEPS} steps, batch={BATCH}, seq={SEQ_LEN}")
    print(f"  Snapshots at: {list(range(0, TOTAL_STEPS+1, SNAPSHOT_EVERY))}\n")

    for step in range(TOTAL_STEPS):
        # Random batch from data
        starts = torch.randint(0, len(all_ids) - SEQ_LEN - 1, (BATCH,))
        x = torch.stack([all_ids[s:s+SEQ_LEN] for s in starts]).to(device)
        y = torch.stack([all_ids[s+1:s+SEQ_LEN+1] for s in starts]).to(device)

        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 8000), y.reshape(-1))
        loss.backward()

        # Snapshot AFTER backward (gradients still alive) BEFORE optimizer step
        if step % SNAPSHOT_EVERY == 0 or step == TOTAL_STEPS - 1:
            snap = monitor.snapshot(step, sample_input=x[:2])
            monitor.print_snapshot(snap)

        # Log every step (lightweight)
        monitor.step(step, loss.item())

        optimizer.step()
        optimizer.zero_grad()

        if step % 25 == 0:
            ppl = min(torch.exp(loss).item(), 99999)
            print(f"  Step {step:4d}  loss={loss.item():.4f}  ppl={ppl:.1f}")

    print(f"\n  Final loss: {loss.item():.4f}")

    monitor.save_report(filename='monitored_run')
    print("Done.")
