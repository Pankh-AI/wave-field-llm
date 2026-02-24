"""
Wave Field Monitor — Visualization
====================================
Renders training_monitor output into a multi-panel dashboard.

Usage:
    python diagnostics/visualize_monitor.py results/monitor/monitor_snapshots.json

Or from code:
    from diagnostics.visualize_monitor import plot_monitor_dashboard
    plot_monitor_dashboard('results/monitor/monitor_snapshots.json',
                           'results/monitor/monitor_steps.json')
"""

import json
import os
import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 11,
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'text.color': '#c9d1d9',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
    'legend.fontsize': 7,
})

# Head colors (8 heads)
HEAD_COLORS = ['#58a6ff', '#f0883e', '#3fb950', '#bc8cff',
               '#f85149', '#79c0ff', '#ffa657', '#d2a8ff']


def plot_monitor_dashboard(snap_path, step_path=None, output_path=None):
    """Generate full monitor dashboard from saved JSON files."""

    with open(snap_path) as f:
        snapshots = json.load(f)

    steps_data = None
    if step_path and os.path.exists(step_path):
        with open(step_path) as f:
            steps_data = json.load(f)

    if not snapshots:
        print("No snapshots to visualize")
        return

    # Detect number of layers
    n_layers = 0
    while f'L{n_layers}_kernel_freq' in snapshots[0] or f'L{n_layers}_spectral_entropy' in snapshots[0]:
        n_layers += 1
    if n_layers == 0:
        n_layers = 1  # fallback

    snap_steps = [s['step'] for s in snapshots]

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle('Wave Field LLM — Training Monitor Dashboard',
                 fontsize=14, fontweight='bold', color='#f0f6fc', y=0.98)

    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3,
                          left=0.05, right=0.97, top=0.93, bottom=0.05)

    # ---- Panel 1: Loss curve (from step log) ----
    ax1 = fig.add_subplot(gs[0, 0])
    if steps_data:
        s_steps = [s['step'] for s in steps_data]
        s_loss = [s['loss'] for s in steps_data]
        ax1.plot(s_steps, s_loss, color='#58a6ff', linewidth=1, alpha=0.7)
        # Smoothed
        if len(s_loss) > 20:
            window = max(len(s_loss) // 20, 5)
            smoothed = np.convolve(s_loss, np.ones(window)/window, mode='valid')
            ax1.plot(s_steps[window-1:], smoothed, color='#f0883e', linewidth=2, label='smoothed')
            ax1.legend()
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: Kernel frequencies per head over training ----
    ax2 = fig.add_subplot(gs[0, 1])
    for snap in snapshots:
        step = snap['step']
        freq = snap.get('L0_kernel_freq', [])
        for h, f in enumerate(freq):
            color = HEAD_COLORS[h % len(HEAD_COLORS)]
            ax2.scatter(step, f, color=color, s=15, alpha=0.7, zorder=3)
    # Connect with lines
    for h in range(8):
        freqs = [snap.get('L0_kernel_freq', [None]*8)[h] for snap in snapshots if snap.get('L0_kernel_freq')]
        if freqs and freqs[0] is not None:
            valid_steps = [s for s, snap in zip(snap_steps, snapshots) if snap.get('L0_kernel_freq')]
            ax2.plot(valid_steps[:len(freqs)], freqs, color=HEAD_COLORS[h % len(HEAD_COLORS)],
                     linewidth=1, alpha=0.5, label=f'H{h}')
    ax2.set_title('L0 Kernel Frequencies (per head)', fontweight='bold')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('ω (frequency)')
    ax2.legend(ncol=4, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # ---- Panel 3: Kernel damping per head over training ----
    ax3 = fig.add_subplot(gs[0, 2])
    for h in range(8):
        damps = [snap.get('L0_kernel_damp', [None]*8)[h] for snap in snapshots if snap.get('L0_kernel_damp')]
        if damps and damps[0] is not None:
            valid_steps = [s for s, snap in zip(snap_steps, snapshots) if snap.get('L0_kernel_damp')]
            ax3.plot(valid_steps[:len(damps)], damps, color=HEAD_COLORS[h % len(HEAD_COLORS)],
                     linewidth=1.5, label=f'H{h}')
    ax3.set_title('L0 Kernel Damping (reach = 1/α)', fontweight='bold')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('α (damping)')
    ax3.legend(ncol=4, loc='upper right')
    ax3.grid(True, alpha=0.3)

    # ---- Panel 4: Spectral entropy ratio over training ----
    ax4 = fig.add_subplot(gs[0, 3])
    for i in range(min(n_layers, 6)):
        ent_means = []
        for snap in snapshots:
            ent = snap.get(f'L{i}_spectral_entropy_ratio', [])
            if ent:
                ent_means.append(np.mean(ent))
        if ent_means:
            ax4.plot(snap_steps[:len(ent_means)], ent_means,
                     linewidth=1.5, label=f'L{i}', alpha=0.8)
    ax4.set_title('Spectral Entropy Ratio (1.0 = uniform)', fontweight='bold')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Entropy / max_entropy')
    ax4.set_ylim(0, 1.1)
    ax4.axhline(y=0.5, color='#f85149', linestyle='--', alpha=0.3, label='low diversity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ---- Panel 5: Feature map deviation from identity ----
    ax5 = fig.add_subplot(gs[1, 0])
    for fm_type, color, ls in [('q_fm', '#58a6ff', '-'), ('k_fm', '#f0883e', '--')]:
        for i in [0, n_layers // 2, n_layers - 1]:
            devs = [snap.get(f'L{i}_{fm_type}_identity_deviation', None) for snap in snapshots]
            devs_clean = [(s, d) for s, d in zip(snap_steps, devs) if d is not None]
            if devs_clean:
                ax5.plot([s for s, _ in devs_clean], [d for _, d in devs_clean],
                         color=color, linestyle=ls, linewidth=1.5,
                         label=f'L{i} {fm_type.split("_")[0]}', alpha=0.8)
    ax5.set_title('Feature Map Deviation from Identity', fontweight='bold')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('||W - I||')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ---- Panel 6: Feature map effective rank ----
    ax6 = fig.add_subplot(gs[1, 1])
    for fm_type, color in [('q_fm', '#58a6ff'), ('k_fm', '#f0883e')]:
        for i in [0, n_layers - 1]:
            ranks = [snap.get(f'L{i}_{fm_type}_effective_rank', None) for snap in snapshots]
            ranks_clean = [(s, r) for s, r in zip(snap_steps, ranks) if r is not None]
            if ranks_clean:
                ax6.plot([s for s, _ in ranks_clean], [r for _, r in ranks_clean],
                         color=color, linewidth=1.5,
                         linestyle='-' if i == 0 else '--',
                         label=f'L{i} {fm_type.split("_")[0]}')
    ax6.set_title('Feature Map Effective Rank', fontweight='bold')
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Effective rank')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # ---- Panel 7: Feature map dead fraction ----
    ax7 = fig.add_subplot(gs[1, 2])
    for fm_type, color in [('q_fm', '#58a6ff'), ('k_fm', '#f0883e')]:
        for i in [0, n_layers - 1]:
            deads = [snap.get(f'L{i}_{fm_type}_dead_fraction', None) for snap in snapshots]
            deads_clean = [(s, d) for s, d in zip(snap_steps, deads) if d is not None]
            if deads_clean:
                ax7.plot([s for s, _ in deads_clean], [d * 100 for _, d in deads_clean],
                         color=color, linewidth=1.5,
                         linestyle='-' if i == 0 else '--',
                         label=f'L{i} {fm_type.split("_")[0]}')
    ax7.set_title('Feature Map Dead Neurons (%)', fontweight='bold')
    ax7.set_xlabel('Step')
    ax7.set_ylabel('Dead %')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # ---- Panel 8: Spectral gate activation ----
    ax8 = fig.add_subplot(gs[1, 3])
    for i in range(min(n_layers, 6)):
        norms = [snap.get(f'L{i}_sg_out_weight_norm', None) for snap in snapshots]
        norms_clean = [(s, n) for s, n in zip(snap_steps, norms) if n is not None]
        if norms_clean:
            ax8.plot([s for s, _ in norms_clean], [n for _, n in norms_clean],
                     linewidth=1.5, label=f'L{i}', alpha=0.8)
    ax8.set_title('Spectral Gate Output Weight Norm', fontweight='bold')
    ax8.set_xlabel('Step')
    ax8.set_ylabel('||W_out||')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # ---- Panel 9: Gradient norms for key params ----
    ax9 = fig.add_subplot(gs[2, 0])
    grad_params = ['freq', 'damp', 'phase', 'sg_out_w', 'qkvg']
    grad_colors = ['#58a6ff', '#f0883e', '#3fb950', '#bc8cff', '#8b949e']
    for gp, gc in zip(grad_params, grad_colors):
        vals = [snap.get(f'L0_grad_{gp}_norm', None) for snap in snapshots]
        vals_clean = [(s, v) for s, v in zip(snap_steps, vals) if v is not None]
        if vals_clean:
            ax9.plot([s for s, _ in vals_clean], [v for _, v in vals_clean],
                     color=gc, linewidth=1.5, label=gp)
    ax9.set_title('L0 Gradient Norms', fontweight='bold')
    ax9.set_xlabel('Step')
    ax9.set_ylabel('||∇||')
    ax9.set_yscale('log')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # ---- Panel 10: Gate bias drift ----
    ax10 = fig.add_subplot(gs[2, 1])
    if steps_data:
        for i in range(min(n_layers, 4)):
            gate_vals = [s.get(f'L{i}_gate_bias_mean', None) for s in steps_data]
            gate_steps = [s['step'] for s in steps_data]
            clean = [(st, g) for st, g in zip(gate_steps, gate_vals) if g is not None]
            if clean:
                ax10.plot([s for s, _ in clean], [g for _, g in clean],
                          linewidth=1, alpha=0.7, label=f'L{i}')
    ax10.axhline(y=2.0, color='#3fb950', linestyle='--', alpha=0.5, label='init (2.0)')
    ax10.set_title('Gate Bias Mean (should start at 2.0)', fontweight='bold')
    ax10.set_xlabel('Step')
    ax10.set_ylabel('mean(gate_bias)')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    # ---- Panel 11: Coupling matrix entropy ----
    ax11 = fig.add_subplot(gs[2, 2])
    for i in range(min(n_layers, 6)):
        ce = [snap.get(f'L{i}_coupling_entropy', None) for snap in snapshots]
        ce_clean = [(s, c) for s, c in zip(snap_steps, ce) if c is not None]
        if ce_clean:
            ax11.plot([s for s, _ in ce_clean], [c for _, c in ce_clean],
                      linewidth=1.5, label=f'L{i}', alpha=0.8)
    ax11.set_title('Cross-Head Coupling Entropy', fontweight='bold')
    ax11.set_xlabel('Step')
    ax11.set_ylabel('Entropy (high = more mixing)')
    ax11.legend()
    ax11.grid(True, alpha=0.3)

    # ---- Panel 12: Output effective rank (if available) ----
    ax12 = fig.add_subplot(gs[2, 3])
    has_rank = any(f'L0_output_eff_rank' in snap for snap in snapshots)
    if has_rank:
        for i in range(min(n_layers, 6)):
            ranks = [snap.get(f'L{i}_output_eff_rank', None) for snap in snapshots]
            ranks_clean = [(s, r) for s, r in zip(snap_steps, ranks) if r is not None]
            if ranks_clean:
                ax12.plot([s for s, _ in ranks_clean], [r for _, r in ranks_clean],
                          linewidth=1.5, label=f'L{i}', alpha=0.8)
        ax12.set_title('Attention Output Effective Rank', fontweight='bold')
    else:
        # Show kernel reach instead
        for h in range(8):
            reaches = [snap.get('L0_kernel_reach', [None]*8)[h] for snap in snapshots
                       if snap.get('L0_kernel_reach')]
            if reaches and reaches[0] is not None:
                valid_steps = [s for s, snap in zip(snap_steps, snapshots) if snap.get('L0_kernel_reach')]
                ax12.plot(valid_steps[:len(reaches)], reaches,
                          color=HEAD_COLORS[h % len(HEAD_COLORS)],
                          linewidth=1.5, label=f'H{h}')
        ax12.set_title('L0 Kernel Reach (1/damping)', fontweight='bold')

    ax12.set_xlabel('Step')
    ax12.legend(ncol=2)
    ax12.grid(True, alpha=0.3)

    # Save
    if output_path is None:
        output_path = snap_path.replace('.json', '_dashboard.png')
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Dashboard saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python diagnostics/visualize_monitor.py <snapshots.json> [steps.json]")
        sys.exit(1)

    snap_path = sys.argv[1]
    step_path = sys.argv[2] if len(sys.argv) > 2 else snap_path.replace('snapshots', 'steps')

    plot_monitor_dashboard(snap_path, step_path)
