"""
Plot long-context benchmark results.
Generates 4 charts from results/long_context_benchmark.json
"""

import json
import os
import sys

# Install matplotlib if missing
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib', 'numpy'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

# Load results
results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'long_context_benchmark.json')
with open(results_path) as f:
    data = json.load(f)

seq_lens = [256, 512, 1024, 2048, 4096, 8192]

std = data['standard']
wave = data['wave']

std_fwd = [std[str(s)]['fwd_ms'] for s in seq_lens]
wave_fwd = [wave[str(s)]['fwd_ms'] for s in seq_lens]

std_mem = [std[str(s)]['peak_mem_mb'] for s in seq_lens]
wave_mem = [wave[str(s)]['peak_mem_mb'] for s in seq_lens]

std_tps = [std[str(s)]['tokens_per_sec'] for s in seq_lens]
wave_tps = [wave[str(s)]['tokens_per_sec'] for s in seq_lens]

speedup = [s / w for s, w in zip(std_fwd, wave_fwd)]

# ---- Style ----
COLORS = {
    'std': '#E74C3C',    # red
    'wave': '#2E86C1',   # blue
    'cross': '#27AE60',  # green
    'bg': '#FAFAFA',
    'grid': '#E0E0E0',
}

plt.rcParams.update({
    'figure.facecolor': COLORS['bg'],
    'axes.facecolor': '#FFFFFF',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': COLORS['grid'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Wave Field LLM vs Standard Transformer — Long-Context Benchmark',
             fontsize=16, fontweight='bold', y=0.98)

# ============================================================
# PLOT 1: Forward Pass Time (log-log)
# ============================================================
ax = axes[0, 0]
ax.loglog(seq_lens, std_fwd, 'o-', color=COLORS['std'], linewidth=2.5,
          markersize=8, label='Standard Transformer O(n²)', zorder=5)
ax.loglog(seq_lens, wave_fwd, 's-', color=COLORS['wave'], linewidth=2.5,
          markersize=8, label='Wave Field O(n log n)', zorder=5)

# Crossover annotation
ax.axvline(x=2048, color=COLORS['cross'], linestyle='--', alpha=0.7, linewidth=1.5)
ax.annotate('Crossover\n2048 tokens', xy=(2048, 30), fontsize=9,
            color=COLORS['cross'], fontweight='bold',
            ha='center', va='bottom')

# Shade the "Wave wins" region
ax.axvspan(2048, 10000, alpha=0.06, color=COLORS['wave'])

ax.set_xlabel('Sequence Length (tokens)')
ax.set_ylabel('Forward Pass Time (ms)')
ax.set_title('Forward Pass Latency')
ax.legend(loc='upper left', fontsize=9)
ax.set_xticks(seq_lens)
ax.set_xticklabels([str(s) for s in seq_lens])

# Add data labels for the extreme point
ax.annotate(f'{std_fwd[-1]:.0f}ms', xy=(8192, std_fwd[-1]),
            textcoords="offset points", xytext=(-40, -15),
            fontsize=8, color=COLORS['std'], fontweight='bold')
ax.annotate(f'{wave_fwd[-1]:.0f}ms', xy=(8192, wave_fwd[-1]),
            textcoords="offset points", xytext=(-35, 10),
            fontsize=8, color=COLORS['wave'], fontweight='bold')

# ============================================================
# PLOT 2: Speedup Factor
# ============================================================
ax = axes[0, 1]
colors_bar = [COLORS['std'] if s < 1.0 else COLORS['wave'] for s in speedup]
bars = ax.bar([str(s) for s in seq_lens], speedup, color=colors_bar, edgecolor='white',
              linewidth=0.8, width=0.6, zorder=5)

# Add value labels on bars
for bar, sp in zip(bars, speedup):
    label = f'{sp:.1f}x' if sp < 10 else f'{sp:.0f}x'
    y_pos = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y_pos + 1,
            label, ha='center', va='bottom', fontweight='bold', fontsize=9,
            color=COLORS['wave'] if sp >= 1.0 else COLORS['std'])

ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.set_xlabel('Sequence Length (tokens)')
ax.set_ylabel('Speedup (Standard / Wave)')
ax.set_title('Wave Field Speedup Factor')
ax.set_yscale('log')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

# Annotate regions
ax.text(0.25, 0.92, 'Standard faster', transform=ax.transAxes,
        fontsize=9, color=COLORS['std'], alpha=0.7, ha='center')
ax.text(0.75, 0.92, 'Wave faster', transform=ax.transAxes,
        fontsize=9, color=COLORS['wave'], alpha=0.7, ha='center')

# ============================================================
# PLOT 3: Throughput (tokens/sec)
# ============================================================
ax = axes[1, 0]
ax.plot(seq_lens, [t/1000 for t in std_tps], 'o-', color=COLORS['std'],
        linewidth=2.5, markersize=8, label='Standard Transformer')
ax.plot(seq_lens, [t/1000 for t in wave_tps], 's-', color=COLORS['wave'],
        linewidth=2.5, markersize=8, label='Wave Field')

ax.axvline(x=2048, color=COLORS['cross'], linestyle='--', alpha=0.7, linewidth=1.5)
ax.axvspan(2048, 10000, alpha=0.06, color=COLORS['wave'])

ax.set_xlabel('Sequence Length (tokens)')
ax.set_ylabel('Throughput (K tokens/sec)')
ax.set_title('Throughput')
ax.legend(loc='center right', fontsize=9)
ax.set_xscale('log', base=2)
ax.set_xticks(seq_lens)
ax.set_xticklabels([str(s) for s in seq_lens])

# Key insight annotation
ax.annotate(f'Standard collapses\nto {std_tps[-1]:,} tok/s',
            xy=(8192, std_tps[-1]/1000), fontsize=8, color=COLORS['std'],
            textcoords="offset points", xytext=(-80, 20),
            arrowprops=dict(arrowstyle='->', color=COLORS['std'], lw=1.2))
ax.annotate(f'Wave scales to\n{wave_tps[-1]:,} tok/s',
            xy=(8192, wave_tps[-1]/1000), fontsize=8, color=COLORS['wave'],
            textcoords="offset points", xytext=(-80, 15),
            arrowprops=dict(arrowstyle='->', color=COLORS['wave'], lw=1.2))

# ============================================================
# PLOT 4: Peak Memory
# ============================================================
ax = axes[1, 1]

x = np.arange(len(seq_lens))
width = 0.35

bars1 = ax.bar(x - width/2, [m/1000 for m in std_mem], width,
               label='Standard Transformer', color=COLORS['std'],
               edgecolor='white', linewidth=0.8, zorder=5)
bars2 = ax.bar(x + width/2, [m/1000 for m in wave_mem], width,
               label='Wave Field', color=COLORS['wave'],
               edgecolor='white', linewidth=0.8, zorder=5)

ax.set_xlabel('Sequence Length (tokens)')
ax.set_ylabel('Peak Memory (GB)')
ax.set_title('Peak GPU Memory')
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in seq_lens])
ax.legend(fontsize=9)

# Add value labels
for bar in bars1:
    h = bar.get_height()
    if h > 0.05:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=7, color=COLORS['std'])
for bar in bars2:
    h = bar.get_height()
    if h > 0.05:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=7, color=COLORS['wave'])

plt.tight_layout(rect=[0, 0, 1, 0.95])

out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'long_context_benchmark.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
print(f"Saved: {out_path}")

# Also save individual high-res plots
for idx, name in enumerate(['latency', 'speedup', 'throughput', 'memory']):
    extent = axes[idx // 2, idx % 2].get_tightbbox(fig.canvas.get_renderer())
    fig_single, ax_single = plt.subplots(1, 1, figsize=(8, 5))

    # Recreate each plot individually for cleaner single exports
    if name == 'latency':
        ax_single.loglog(seq_lens, std_fwd, 'o-', color=COLORS['std'], linewidth=2.5,
                         markersize=8, label='Standard Transformer O(n²)')
        ax_single.loglog(seq_lens, wave_fwd, 's-', color=COLORS['wave'], linewidth=2.5,
                         markersize=8, label='Wave Field O(n log n)')
        ax_single.axvline(x=2048, color=COLORS['cross'], linestyle='--', alpha=0.7)
        ax_single.axvspan(2048, 10000, alpha=0.06, color=COLORS['wave'])
        ax_single.set_xlabel('Sequence Length (tokens)')
        ax_single.set_ylabel('Forward Pass Time (ms)')
        ax_single.set_title('Forward Pass Latency — Log Scale')
        ax_single.legend(fontsize=10)
        ax_single.set_xticks(seq_lens)
        ax_single.set_xticklabels([str(s) for s in seq_lens])
        ax_single.annotate(f'{std_fwd[-1]:.0f}ms', xy=(8192, std_fwd[-1]),
                           textcoords="offset points", xytext=(-50, -15),
                           fontsize=10, color=COLORS['std'], fontweight='bold')
        ax_single.annotate(f'{wave_fwd[-1]:.0f}ms', xy=(8192, wave_fwd[-1]),
                           textcoords="offset points", xytext=(-45, 10),
                           fontsize=10, color=COLORS['wave'], fontweight='bold')

    elif name == 'speedup':
        colors_bar = [COLORS['std'] if s < 1.0 else COLORS['wave'] for s in speedup]
        bars = ax_single.bar([str(s) for s in seq_lens], speedup, color=colors_bar,
                             edgecolor='white', linewidth=0.8, width=0.5)
        for bar, sp in zip(bars, speedup):
            label = f'{sp:.1f}x' if sp < 10 else f'{sp:.0f}x'
            ax_single.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           label, ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax_single.axhline(y=1.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax_single.set_xlabel('Sequence Length (tokens)')
        ax_single.set_ylabel('Speedup (Standard / Wave)')
        ax_single.set_title('Wave Field Speedup Over Standard Transformer')
        ax_single.set_yscale('log')

    elif name == 'throughput':
        ax_single.plot(seq_lens, [t/1000 for t in std_tps], 'o-', color=COLORS['std'],
                       linewidth=2.5, markersize=8, label='Standard Transformer')
        ax_single.plot(seq_lens, [t/1000 for t in wave_tps], 's-', color=COLORS['wave'],
                       linewidth=2.5, markersize=8, label='Wave Field')
        ax_single.axvline(x=2048, color=COLORS['cross'], linestyle='--', alpha=0.7)
        ax_single.axvspan(2048, 10000, alpha=0.06, color=COLORS['wave'])
        ax_single.set_xlabel('Sequence Length (tokens)')
        ax_single.set_ylabel('Throughput (K tokens/sec)')
        ax_single.set_title('Throughput — Standard Collapses, Wave Scales')
        ax_single.legend(fontsize=10)
        ax_single.set_xscale('log', base=2)
        ax_single.set_xticks(seq_lens)
        ax_single.set_xticklabels([str(s) for s in seq_lens])

    elif name == 'memory':
        x = np.arange(len(seq_lens))
        ax_single.bar(x - 0.175, [m/1000 for m in std_mem], 0.35,
                      label='Standard Transformer', color=COLORS['std'], edgecolor='white')
        ax_single.bar(x + 0.175, [m/1000 for m in wave_mem], 0.35,
                      label='Wave Field', color=COLORS['wave'], edgecolor='white')
        ax_single.set_xlabel('Sequence Length (tokens)')
        ax_single.set_ylabel('Peak Memory (GB)')
        ax_single.set_title('Peak GPU Memory Usage')
        ax_single.set_xticks(x)
        ax_single.set_xticklabels([str(s) for s in seq_lens])
        ax_single.legend(fontsize=10)

    ax_single.grid(True, alpha=0.3)
    single_path = os.path.join(os.path.dirname(__file__), '..', 'results', f'{name}.png')
    fig_single.savefig(single_path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig_single)
    print(f"Saved: {single_path}")

print("\nAll plots generated.")
