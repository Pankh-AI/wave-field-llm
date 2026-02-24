"""
Wave Field LLM — Complete Benchmark Visualization
===================================================
Generates a comprehensive dashboard of all benchmark data.
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
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
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# Colors
C_WAVE = '#58a6ff'       # blue
C_STD = '#f0883e'        # orange
C_WAVE2 = '#79c0ff'      # light blue
C_STD2 = '#ffa657'       # light orange
C_BAD = '#f85149'        # red
C_GOOD = '#3fb950'       # green
C_NEUTRAL = '#8b949e'    # grey
C_ACCENT = '#bc8cff'     # purple


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_s1_scaling(ax_ppl, ax_acc):
    """S1 scaling: 20M tokens, Wave vs Standard training curves."""
    data = load_json('scaling_s1.json')
    if not data:
        data = load_json('scaling_benchmark.json')
    if not data:
        return

    results = data['results'] if 'results' in data else data

    for r in results:
        curve = r['curve']
        tokens = [p['tokens_M'] for p in curve]
        ppls = [p['ppl'] for p in curve]
        accs = [p['acc'] for p in curve]

        is_wave = 'wave' in r.get('model_type', r.get('run_name', '')).lower()
        color = C_WAVE if is_wave else C_STD
        label = f"{'Wave' if is_wave else 'Standard'} ({r['params']/1e6:.1f}M)"

        ax_ppl.plot(tokens, ppls, color=color, linewidth=2, label=label, marker='o', markersize=3)
        ax_acc.plot(tokens, accs, color=color, linewidth=2, label=label, marker='o', markersize=3)

    ax_ppl.set_title('S1 Training Curves (20M tokens, WikiText-103)', fontweight='bold')
    ax_ppl.set_xlabel('Tokens (M)')
    ax_ppl.set_ylabel('Perplexity (log scale)')
    ax_ppl.set_yscale('log')
    ax_ppl.legend()
    ax_ppl.grid(True, alpha=0.3)

    # Mark the persistent gap
    ax_ppl.annotate('1.69x gap\n(stubborn)',
                    xy=(20, 275), fontsize=9, color=C_BAD,
                    ha='center', fontweight='bold')

    ax_acc.set_title('S1 Accuracy Curves (20M tokens)', fontweight='bold')
    ax_acc.set_xlabel('Tokens (M)')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)


def plot_v44_ablation(ax):
    """V4.4 upgrade ablation — what didn't work."""
    data = load_json('v44_upgrade.json')
    if not data:
        return

    colors = {
        'V4.3 Best': C_WAVE,
        'Write Gate': C_ACCENT,
        '3D Interference': C_BAD,
        'Both': '#da3633',
        'Standard': C_STD,
    }

    name_map = {
        'A) V4.3 Best (baseline)': 'V4.3 Best',
        'B) + Write Gate (GLA-style)': 'Write Gate',
        'C) + 3D Interference': '3D Interference',
        'D) + Both (Write Gate + 3D)': 'Both',
        'E) Standard Transformer': 'Standard',
    }

    for r in data:
        name = name_map.get(r['run_name'], r['run_name'])
        curve = r['curve']
        tokens = [p['tokens_M'] for p in curve]
        ppls = [p['ppl'] for p in curve]
        color = colors.get(name, C_NEUTRAL)
        ls = '--' if name in ('3D Interference', 'Both') else '-'
        lw = 2.5 if name in ('V4.3 Best', 'Standard') else 1.5
        ax.plot(tokens, ppls, color=color, linewidth=lw, label=f"{name} ({r['best_ppl']:.0f})",
                linestyle=ls, marker='o', markersize=2)

    ax.set_title('V4.4 Ablation — What Didn\'t Work (5M tok)', fontweight='bold')
    ax.set_xlabel('Tokens (M)')
    ax.set_ylabel('Perplexity')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Annotate failures
    ax.annotate('All additive\nchanges REGRESSED',
                xy=(3.5, 950), fontsize=8, color=C_BAD, ha='center',
                fontweight='bold')


def plot_v42_ablation(ax):
    """V4.2 ablation — the QK LR 3x discovery."""
    data = load_json('v42_ablation.json')
    if not data:
        return

    colors_v42 = [C_NEUTRAL, '#8b949e', C_GOOD, C_ACCENT, C_WAVE, C_STD]
    markers = ['s', 'D', '^', 'v', 'o', 'P']

    for i, r in enumerate(data):
        curve = r['curve']
        tokens = [p['tokens_M'] for p in curve]
        ppls = [p['ppl'] for p in curve]
        name = r['run_name'].split(') ')[1] if ')' in r['run_name'] else r['run_name']
        if len(name) > 25:
            name = name[:25] + '...'
        lw = 2.5 if 'QK LR' in r['run_name'] or 'Standard' in r['run_name'] else 1.2
        ax.plot(tokens, ppls, color=colors_v42[i], linewidth=lw,
                label=f"{name} ({r['best_ppl']:.0f})",
                marker=markers[i], markersize=3)

    ax.set_title('V4.2 Ablation — QK LR 3x Is The Only Win', fontweight='bold')
    ax.set_xlabel('Tokens (M)')
    ax.set_ylabel('Perplexity')
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

    ax.annotate('QK LR 3x\n-21% PPL',
                xy=(4.5, 734), fontsize=9, color=C_GOOD, ha='center',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_GOOD, lw=1.5),
                xytext=(3.5, 600))


def plot_speed_crossover(ax):
    """Speed vs sequence length — the crossover plot."""
    data = load_json('long_context_benchmark.json')
    if not data:
        return

    seq_lens = sorted([int(k) for k in data['standard'].keys()])
    std_tps = [data['standard'][str(s)]['tokens_per_sec'] / 1000 for s in seq_lens]
    wave_tps = [data['wave'][str(s)]['tokens_per_sec'] / 1000 for s in seq_lens]

    ax.plot(seq_lens, std_tps, color=C_STD, linewidth=2.5, label='Standard O(n²)',
            marker='s', markersize=6)
    ax.plot(seq_lens, wave_tps, color=C_WAVE, linewidth=2.5, label='Wave Field O(n log n)',
            marker='o', markersize=6)

    ax.set_title('Speed Crossover — Wave Wins at 2K+', fontweight='bold')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Throughput (K tok/s)')
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Shade crossover region
    ax.axvspan(1500, 2500, alpha=0.15, color=C_GOOD)
    ax.annotate('CROSSOVER\n~2K tokens', xy=(2048, 90), fontsize=9,
                color=C_GOOD, ha='center', fontweight='bold')

    # Annotate 8K dominance
    ax.annotate('108x faster!', xy=(8192, 152), fontsize=10,
                color=C_WAVE, ha='center', fontweight='bold',
                xytext=(6000, 130),
                arrowprops=dict(arrowstyle='->', color=C_WAVE, lw=1.5))


def plot_memory_scaling(ax):
    """Memory usage vs sequence length."""
    data = load_json('long_context_benchmark.json')
    if not data:
        return

    seq_lens = sorted([int(k) for k in data['standard'].keys()])
    std_mem = [data['standard'][str(s)]['peak_mem_mb'] for s in seq_lens]
    wave_mem = [data['wave'][str(s)]['peak_mem_mb'] for s in seq_lens]

    ax.plot(seq_lens, std_mem, color=C_STD, linewidth=2.5, label='Standard O(n²)',
            marker='s', markersize=6)
    ax.plot(seq_lens, wave_mem, color=C_WAVE, linewidth=2.5, label='Wave Field O(n log n)',
            marker='o', markersize=6)

    ax.set_title('Memory Scaling', fontweight='bold')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Peak VRAM (MB)')
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_v43_upgrade(ax):
    """V4.3 architecture component ablation."""
    data = load_json('v43_upgrade.json')
    if not data:
        return

    names = []
    ppls = []
    colors = []

    name_map = {
        'A) Baseline (legacy kernel, 1L FM)': 'Legacy Kernel',
        'B) + Analytic kernel (S4D Z-transform)': '+ S4D Kernel',
        'C) + 2-Layer FM (Hedgehog)': '+ Hedgehog FM',
        'D) + Local window 64 (BASED)': '+ Local Win 64',
        'E) Standard Transformer': 'Standard',
    }

    for r in data:
        name = name_map.get(r['run_name'], r['run_name'])
        names.append(name)
        ppls.append(r['best_ppl'])
        if 'Local' in name:
            colors.append(C_BAD)
        elif 'Standard' in name:
            colors.append(C_STD)
        else:
            colors.append(C_WAVE)

    bars = ax.barh(range(len(names)), ppls, color=colors, height=0.6, edgecolor='#30363d')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Perplexity (lower = better)')
    ax.set_title('V4.3 Architecture Components (5M tok)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Value labels
    for i, (bar, ppl) in enumerate(zip(bars, ppls)):
        color = C_BAD if ppl > 900 else '#c9d1d9'
        ax.text(ppl + 15, i, f'{ppl:.0f}', va='center', fontsize=9,
                color=color, fontweight='bold')


def plot_gap_ratio(ax):
    """PPL gap ratio over time — does it converge?"""
    data = load_json('scaling_s1.json')
    if not data:
        data = load_json('scaling_benchmark.json')
    if not data:
        return

    results = data['results'] if 'results' in data else data
    wave_curve = None
    std_curve = None

    for r in results:
        is_wave = 'wave' in r.get('model_type', r.get('run_name', '')).lower()
        if is_wave:
            wave_curve = r['curve']
        else:
            std_curve = r['curve']

    if not wave_curve or not std_curve:
        return

    # Align by tokens_M
    tokens = []
    ratios = []
    for w, s in zip(wave_curve[1:], std_curve[1:]):  # skip step 0
        if w['tokens_M'] == s['tokens_M']:
            ratio = w['ppl'] / s['ppl']
            tokens.append(w['tokens_M'])
            ratios.append(ratio)

    ax.plot(tokens, ratios, color=C_BAD, linewidth=2.5, marker='o', markersize=4)
    ax.axhline(y=1.0, color=C_GOOD, linestyle='--', linewidth=1, alpha=0.7, label='Parity (1.0x)')
    ax.axhline(y=1.69, color=C_BAD, linestyle=':', linewidth=1, alpha=0.7, label='Current (1.69x)')

    ax.set_title('PPL Gap Ratio Over Training (Wave/Standard)', fontweight='bold')
    ax.set_xlabel('Tokens (M)')
    ax.set_ylabel('PPL Ratio (lower = better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 3.0)

    ax.annotate('Gap STABLE at 1.69x\n→ Architectural limit,\n   not data hunger',
                xy=(15, 1.69), fontsize=8, color=C_BAD, ha='center',
                fontweight='bold',
                xytext=(12, 2.3),
                arrowprops=dict(arrowstyle='->', color=C_BAD, lw=1.5))


def plot_plateau_analysis(ax):
    """Training plateau: where does the learning rate slow down?"""
    data = load_json('v44_upgrade.json')
    if not data:
        return

    # Use V4.3 baseline curve
    baseline = data[0]  # A) V4.3 Best
    curve = baseline['curve']
    tokens = [p['tokens_M'] for p in curve if p['tokens_M'] > 0]
    ppls = [p['ppl'] for p in curve if p['tokens_M'] > 0]

    # Compute improvement rate (delta PPL per 0.5M tokens)
    improvements = []
    imp_tokens = []
    for i in range(1, len(ppls)):
        delta = ppls[i-1] - ppls[i]
        imp_tokens.append(tokens[i])
        improvements.append(delta)

    ax.bar(imp_tokens, improvements, width=0.4, color=C_WAVE, alpha=0.8,
           edgecolor='#30363d')
    ax.set_title('PPL Improvement Rate Per 0.5M Tokens', fontweight='bold')
    ax.set_xlabel('Tokens (M)')
    ax.set_ylabel('ΔPPL (positive = improving)')
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight plateau
    ax.axvspan(0.5, 2.0, alpha=0.15, color=C_BAD)
    ax.annotate('PLATEAU\n(stuck at ~1380)',
                xy=(1.25, 20), fontsize=9, color=C_BAD, ha='center',
                fontweight='bold')

    ax.axvspan(2.0, 3.5, alpha=0.15, color=C_GOOD)
    ax.annotate('BREAKTHROUGH\n(SPECTRE activates)',
                xy=(2.75, 200), fontsize=9, color=C_GOOD, ha='center',
                fontweight='bold')


def plot_summary_bar(ax):
    """Summary: final PPL comparison across all experiments."""
    experiments = [
        ('V4.2 Baseline', 912, C_NEUTRAL),
        ('V4.2 +QK LR 3x', 724, C_GOOD),
        ('V4.3 Legacy', 742, C_NEUTRAL),
        ('V4.3 +S4D', 709, C_WAVE2),
        ('V4.3 +Hedgehog', 706, C_WAVE),
        ('V4.3 +LocalWin', 975, C_BAD),
        ('V4.4 +WriteGate', 710, C_ACCENT),
        ('V4.4 +3D Interf', 830, C_BAD),
        ('Standard', 407, C_STD),
    ]

    names = [e[0] for e in experiments]
    ppls = [e[1] for e in experiments]
    colors = [e[2] for e in experiments]

    bars = ax.barh(range(len(names)), ppls, color=colors, height=0.6,
                   edgecolor='#30363d')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Perplexity (lower = better)')
    ax.set_title('All Experiments Summary (S1 config, 5M tok)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Standard reference line
    ax.axvline(x=407, color=C_STD, linestyle='--', linewidth=1.5, alpha=0.5)

    for i, (bar, ppl) in enumerate(zip(bars, ppls)):
        ax.text(ppl + 10, i, f'{ppl}', va='center', fontsize=8,
                color='#c9d1d9', fontweight='bold')


def main():
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('Wave Field LLM — Complete Benchmark Dashboard\n'
                 'V4.3 SPECTRE-Wave Architecture Analysis',
                 fontsize=16, fontweight='bold', color='#f0f6fc', y=0.98)

    # Layout: 4 rows x 3 columns
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3,
                          left=0.06, right=0.97, top=0.93, bottom=0.04)

    # Row 1: S1 scaling (big) + gap ratio
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1b = fig.add_subplot(gs[0, 2])
    plot_s1_scaling(ax1, ax1b)

    # Row 2: V4.4 ablation + V4.2 ablation + V4.3 components
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])
    plot_v44_ablation(ax2)
    plot_v42_ablation(ax3)
    plot_v43_upgrade(ax4)

    # Row 3: Speed crossover + Memory + Gap ratio
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    ax7 = fig.add_subplot(gs[2, 2])
    plot_speed_crossover(ax5)
    plot_memory_scaling(ax6)
    plot_gap_ratio(ax7)

    # Row 4: Plateau analysis + Summary bar
    ax8 = fig.add_subplot(gs[3, 0])
    ax9 = fig.add_subplot(gs[3, 1:3])
    plot_plateau_analysis(ax8)
    plot_summary_bar(ax9)

    output_path = os.path.join(RESULTS_DIR, 'complete_benchmark_dashboard.png')
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Dashboard saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    main()
