"""
V4.3.2 Training Analysis Dashboard
Compares V4.3 → V4.3.2 improvements vs Standard Transformer baseline.

Generates: results/v432_analysis.png
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# ─── Style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.8,
    'font.family': 'monospace',
    'font.size': 10,
})

COLORS = {
    'standard': '#58a6ff',   # blue
    'v43':      '#f78166',   # orange
    'v432':     '#3fb950',   # green
    'gap':      '#d2a8ff',   # purple
    'target':   '#ffd700',   # gold
}

# ─── Data ────────────────────────────────────────────────────────────
results_dir = Path(__file__).parent

with open(results_dir / 'scaling_s1.json') as f:
    s1_data = json.load(f)

# Parse V4.3 and Standard curves from saved results
v43_curve = next(r for r in s1_data['results'] if r['model_type'] == 'wave')['curve']
std_curve = next(r for r in s1_data['results'] if r['model_type'] == 'standard')['curve']

v43_tokens = [p['tokens_M'] for p in v43_curve if p['tokens_M'] > 0]
v43_ppl    = [p['ppl']      for p in v43_curve if p['tokens_M'] > 0]
v43_acc    = [p['acc']      for p in v43_curve if p['tokens_M'] > 0]

std_tokens = [p['tokens_M'] for p in std_curve if p['tokens_M'] > 0]
std_ppl    = [p['ppl']      for p in std_curve if p['tokens_M'] > 0]
std_acc    = [p['acc']      for p in std_curve if p['tokens_M'] > 0]

# V4.3.2 live data (from running benchmark)
v432_data = [
    (1.0,  1380.2,  3.8),
    (2.0,  1367.3,  5.5),
    (3.0,   924.1,  8.6),
    (4.0,   638.4, 10.4),
    (5.0,   521.0, 11.3),
    (6.0,   453.3, 12.0),
    (7.0,   401.1, 12.9),
    (8.0,   372.1, 13.4),
    (9.0,   341.8, 13.9),
    (10.0,  322.3, 14.2),
    (11.0,  302.1, 14.6),
    (12.0,  287.8, 14.9),
    (13.0,  272.9, 15.2),
]
v432_tokens = [d[0] for d in v432_data]
v432_ppl    = [d[1] for d in v432_data]
v432_acc    = [d[2] for d in v432_data]

# ─── Derived Metrics ────────────────────────────────────────────────

# PPL ratio (Wave / Standard) at matched token counts
# Interpolate standard to V4.3 token points
std_ppl_interp_v43 = np.interp(v43_tokens, std_tokens, std_ppl)
std_ppl_interp_v432 = np.interp(v432_tokens, std_tokens, std_ppl)

gap_v43  = [w / s for w, s in zip(v43_ppl, std_ppl_interp_v43)]
gap_v432 = [w / s for w, s in zip(v432_ppl, std_ppl_interp_v432)]

# V4.3.2 improvement over V4.3 (% PPL reduction)
v43_ppl_interp = np.interp(v432_tokens, v43_tokens, v43_ppl)
improvement = [(old - new) / old * 100 for old, new in zip(v43_ppl_interp, v432_ppl)]

# ─── Figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.suptitle('Wave Field LLM — V4.3.2 Training Analysis\n'
             'Fixes: Kernel LR ×50 + Per-Layer Init + ELU+1 Activation',
             fontsize=15, fontweight='bold', color='white', y=0.98)

gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3,
                      left=0.08, right=0.95, top=0.92, bottom=0.06)

# ── Panel 1: PPL Learning Curves (log scale) ────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.semilogy(std_tokens, std_ppl, '-', color=COLORS['standard'], linewidth=2.5,
             label=f'Standard Transformer (final: {162.3})', alpha=0.9)
ax1.semilogy(v43_tokens, v43_ppl, '-', color=COLORS['v43'], linewidth=2.0,
             label=f'V4.3 SPECTRE-Wave (final: {274.65})', alpha=0.8)
ax1.semilogy(v432_tokens, v432_ppl, '-o', color=COLORS['v432'], linewidth=2.5,
             markersize=4, label=f'V4.3.2 (latest: {v432_ppl[-1]} @ {v432_tokens[-1]}M)',
             zorder=5)

# Crossover annotation: V4.3.2 beats V4.3 final at 13M
ax1.axhline(y=274.65, color=COLORS['v43'], linestyle=':', alpha=0.4, linewidth=1)
ax1.annotate(f'V4.3 final: 274.65',
             xy=(20, 274.65), fontsize=8, color=COLORS['v43'], alpha=0.7,
             ha='right', va='bottom')

# V4.3.2 crosses V4.3-final line
ax1.annotate(f'V4.3.2 beats V4.3 final\nat only 13M tokens!',
             xy=(13, 272.9), xytext=(8, 200),
             fontsize=9, color=COLORS['v432'], fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COLORS['v432'], lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#161b22',
                       edgecolor=COLORS['v432'], alpha=0.9))

# Target zone
ax1.axhspan(155, 212, alpha=0.08, color=COLORS['target'])
ax1.text(1.5, 175, 'TARGET: <1.3× gap (PPL < 211)', fontsize=8,
         color=COLORS['target'], alpha=0.8, fontweight='bold')

ax1.set_xlabel('Tokens (M)')
ax1.set_ylabel('Validation PPL (log)')
ax1.set_title('Perplexity Learning Curves — S1 Scale (22M params, WikiText-2)',
              fontsize=12, pad=10)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.8,
           facecolor='#161b22', edgecolor='#30363d')
ax1.set_xlim(0, 21)
ax1.set_ylim(100, 10000)
ax1.grid(True, which='both', alpha=0.3)
ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax1.yaxis.set_minor_formatter(ticker.NullFormatter())

# ── Panel 2: Accuracy Curves ────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(std_tokens, std_acc, '-', color=COLORS['standard'], linewidth=2, label='Standard')
ax2.plot(v43_tokens, v43_acc, '-', color=COLORS['v43'], linewidth=1.5, label='V4.3', alpha=0.8)
ax2.plot(v432_tokens, v432_acc, '-o', color=COLORS['v432'], linewidth=2,
         markersize=3, label='V4.3.2')

ax2.axhline(y=18.83, color=COLORS['standard'], linestyle=':', alpha=0.3)
ax2.text(0.5, 19.1, 'Std final: 18.83%', fontsize=7, color=COLORS['standard'], alpha=0.6)

ax2.set_xlabel('Tokens (M)')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Token Prediction Accuracy', fontsize=11)
ax2.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d')
ax2.set_xlim(0, 21)
ax2.grid(True, alpha=0.3)

# ── Panel 3: PPL Gap Ratio ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(v43_tokens, gap_v43, '-', color=COLORS['v43'], linewidth=1.5,
         label='V4.3 / Standard')
ax3.plot(v432_tokens, gap_v432, '-o', color=COLORS['v432'], linewidth=2,
         markersize=3, label='V4.3.2 / Standard')

ax3.axhline(y=1.0, color=COLORS['standard'], linestyle='-', alpha=0.3, linewidth=1)
ax3.axhline(y=1.3, color=COLORS['target'], linestyle='--', alpha=0.5, linewidth=1.5)
ax3.text(0.5, 1.32, 'Target: 1.3×', fontsize=8, color=COLORS['target'])

# Final gap annotations
v43_final_gap = v43_ppl[-1] / std_ppl[-1]
v432_current_gap = v432_ppl[-1] / float(np.interp(v432_tokens[-1], std_tokens, std_ppl))
ax3.annotate(f'V4.3 final: {v43_final_gap:.2f}×',
             xy=(v43_tokens[-1], gap_v43[-1]),
             fontsize=8, color=COLORS['v43'], ha='right')
ax3.annotate(f'V4.3.2 now: {v432_current_gap:.2f}×',
             xy=(v432_tokens[-1], gap_v432[-1]),
             xytext=(v432_tokens[-1]-3, gap_v432[-1]+0.1),
             fontsize=9, color=COLORS['v432'], fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COLORS['v432'], lw=1))

ax3.set_xlabel('Tokens (M)')
ax3.set_ylabel('PPL Ratio (Wave ÷ Standard)')
ax3.set_title('Quality Gap vs Standard Transformer', fontsize=11)
ax3.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d')
ax3.set_xlim(0, 21)
ax3.set_ylim(0.8, 2.5)
ax3.grid(True, alpha=0.3)

# ── Panel 4: V4.3.2 Improvement % ──────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
bars = ax4.bar(v432_tokens, improvement, width=0.7, color=COLORS['v432'], alpha=0.8,
               edgecolor=COLORS['v432'], linewidth=0.5)

# Color-code bars by improvement magnitude
for bar, imp in zip(bars, improvement):
    if imp > 15:
        bar.set_facecolor('#238636')  # strong green
    elif imp > 10:
        bar.set_facecolor('#3fb950')  # green
    else:
        bar.set_facecolor('#56d364')  # light green
        bar.set_alpha(0.6)

# Add value labels
for tok, imp in zip(v432_tokens, improvement):
    ax4.text(tok, imp + 0.5, f'{imp:.0f}%', ha='center', va='bottom',
             fontsize=7, color='#c9d1d9')

ax4.set_xlabel('Tokens (M)')
ax4.set_ylabel('PPL Improvement (%)')
ax4.set_title('V4.3.2 vs V4.3 — PPL Reduction per Checkpoint', fontsize=11)
ax4.set_xlim(0, 14)
ax4.grid(True, alpha=0.3, axis='y')

# ── Panel 5: What the Fixes Did ─────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')

# Summary table
summary_text = """
┌───────────────────────────────────────────┐
│         V4.3.2 FIX IMPACT SUMMARY         │
├───────────────────────────────────────────┤
│                                           │
│  FIX 1: Kernel LR ×50 + No Weight Decay  │
│  ├─ Gradients were 27-80× too small       │
│  └─ Now: physics params learn properly    │
│                                           │
│  FIX 2: Per-Layer Kernel Diversity        │
│  ├─ All 6 layers had identical init       │
│  └─ Now: freq 0.5×→2× across depth       │
│                                           │
│  FIX 3: ReLU → ELU+1 Activation          │
│  ├─ ReLU killed 35-55% of neurons         │
│  └─ Now: 0% dead, smooth gradients        │
│                                           │
├───────────────────────────────────────────┤
│  RESULT @ 13M tokens:                     │
│  V4.3:     310.69  (gap: 1.65×)           │
│  V4.3.2:   272.9   (gap: 1.45×)           │
│  Standard: 188.87                         │
│  Improvement: 12.2% PPL reduction         │
│                                           │
│  V4.3.2 @ 13M ≈ V4.3 @ 20M (final)      │
│  → 35% less data for same quality!        │
└───────────────────────────────────────────┘
"""

ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
         fontsize=9, fontfamily='monospace', color='#c9d1d9',
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='#0d1117',
                   edgecolor='#30363d', alpha=0.9))

# ── Save ─────────────────────────────────────────────────────────────
out_path = results_dir / 'v432_analysis.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'Dashboard saved to {out_path}')
print(f'  V4.3.2 latest: PPL {v432_ppl[-1]} at {v432_tokens[-1]}M tokens')
print(f'  V4.3 final:    PPL 274.65 at 20M tokens')
print(f'  Standard final: PPL 162.3 at 20M tokens')
print(f'  V4.3.2 gap:     {v432_current_gap:.2f}× (target: <1.3×)')
print(f'  V4.3.2 already beat V4.3 final with 35% less data')
