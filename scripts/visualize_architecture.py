#!/usr/bin/env python3
"""
Wave Field LLM V4.3.9 — Full Architecture Visualization
=========================================================
Generates a 3-panel publication-quality figure:
  Panel 1: Full model overview (token → output)
  Panel 2: Single transformer layer detail
  Panel 3: Wave Field Attention internals (the core innovation)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Color palette (dark theme, consistent with existing figures) ──────────
BG       = '#0d1117'
BG_PANEL = '#161b22'
TEXT     = '#e6edf3'
TEXT_DIM = '#8b949e'
ACCENT   = '#58a6ff'   # blue
GREEN    = '#3fb950'
ORANGE   = '#d29922'
PURPLE   = '#bc8cff'
RED      = '#f85149'
PINK     = '#f778ba'
TEAL     = '#39d353'
CYAN     = '#79c0ff'

# Box colors for different component types
C_INPUT  = '#1f3a5f'   # input/output
C_PROJ   = '#2d333b'   # projections
C_FEAT   = '#2a4858'   # feature maps
C_FIELD  = '#1a3528'   # field operations
C_FFT    = '#3b2a1a'   # FFT / frequency domain
C_GATE   = '#3a1a3a'   # gating
C_NORM   = '#2a2a3a'   # normalization
C_RESID  = '#1a2a1a'   # residual connections


def draw_box(ax, x, y, w, h, text, color=C_PROJ, text_color=TEXT,
             fontsize=9, border_color=None, alpha=0.9, bold=False,
             subtext=None, subcolor=TEXT_DIM):
    """Draw a rounded rectangle with centered text."""
    bc = border_color or color
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor=bc,
        alpha=alpha, linewidth=1.2,
        transform=ax.transData
    )
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y + (0.008 if subtext else 0), text,
            ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight=weight, family='monospace')
    if subtext:
        ax.text(x, y - 0.022, subtext,
                ha='center', va='center', fontsize=fontsize - 2,
                color=subcolor, family='monospace')


def draw_arrow(ax, x1, y1, x2, y2, color=TEXT_DIM, style='->', lw=1.2):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def draw_curved_arrow(ax, x1, y1, x2, y2, color=TEXT_DIM, connectionstyle="arc3,rad=0.3"):
    """Draw a curved arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2,
                                connectionstyle=connectionstyle))


def draw_label(ax, x, y, text, fontsize=7, color=TEXT_DIM, ha='center'):
    """Draw a label."""
    ax.text(x, y, text, ha=ha, va='center', fontsize=fontsize,
            color=color, family='monospace')


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL 1: Full Model Overview
# ═══════════════════════════════════════════════════════════════════════════
def draw_panel1_full_model(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(BG_PANEL)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(0.5, 0.96, 'Wave Field Transformer — Full Model',
            ha='center', va='top', fontsize=13, color=TEXT,
            fontweight='bold', family='monospace')
    ax.text(0.5, 0.92, 'O(n log n) physics-based language model',
            ha='center', va='top', fontsize=8, color=TEXT_DIM, family='monospace')

    # Vertical stack from bottom to top
    bw, bh = 0.28, 0.045
    cx = 0.5

    # Token input
    y = 0.10
    draw_box(ax, cx, y, bw, bh, 'Input Tokens', C_INPUT, ACCENT, fontsize=10)
    draw_label(ax, cx + 0.2, y, '(B, N)', fontsize=7)

    # Embedding + PE
    y_emb = 0.18
    draw_arrow(ax, cx, y + bh/2, cx, y_emb - bh/2, ACCENT)
    draw_box(ax, cx, y_emb, bw, bh, 'Embedding + Sinusoidal PE', C_PROJ,
             fontsize=9)
    draw_label(ax, cx + 0.22, y_emb, '(B, N, D)', fontsize=7)

    # Dropout
    y_drop = 0.24
    draw_arrow(ax, cx, y_emb + bh/2, cx, y_drop - bh/2, TEXT_DIM)
    draw_box(ax, cx, y_drop, 0.15, 0.03, 'Dropout', C_NORM, fontsize=8)

    # Transformer layers block
    y_block_bot = 0.30
    y_block_top = 0.76
    # Draw the stacked layer block
    block_rect = FancyBboxPatch(
        (cx - 0.22, y_block_bot), 0.44, y_block_top - y_block_bot,
        boxstyle="round,pad=0.02",
        facecolor='#0d1117', edgecolor=ACCENT,
        alpha=0.6, linewidth=1.5, linestyle='--'
    )
    ax.add_patch(block_rect)
    ax.text(cx + 0.24, y_block_top - 0.01, '×12 layers',
            ha='left', va='top', fontsize=9, color=ACCENT,
            fontweight='bold', family='monospace')

    draw_arrow(ax, cx, y_drop + 0.015, cx, y_block_bot + 0.02, TEXT_DIM)

    # Inside: single layer representation
    yl1 = y_block_bot + 0.06
    draw_box(ax, cx, yl1, 0.36, 0.04, 'LayerNorm', C_NORM, fontsize=8)

    yl2 = yl1 + 0.07
    draw_arrow(ax, cx, yl1 + 0.02, cx, yl2 - 0.025, TEXT_DIM)
    draw_box(ax, cx, yl2, 0.36, 0.055, '★ Wave Field Attention', C_FFT,
             ORANGE, fontsize=10, bold=True, border_color=ORANGE,
             subtext='O(n log n) FFT convolution')

    # Residual arrow for attention
    draw_curved_arrow(ax, cx - 0.19, yl1 - 0.02, cx - 0.19, yl2 + 0.04,
                      GREEN, "arc3,rad=-0.5")
    draw_label(ax, cx - 0.28, (yl1 + yl2)/2, '+', fontsize=11, color=GREEN)

    yl3 = yl2 + 0.08
    draw_arrow(ax, cx, yl2 + 0.028, cx, yl3 - 0.02, TEXT_DIM)
    draw_box(ax, cx, yl3, 0.36, 0.04, 'LayerNorm', C_NORM, fontsize=8)

    yl4 = yl3 + 0.07
    draw_arrow(ax, cx, yl3 + 0.02, cx, yl4 - 0.02, TEXT_DIM)
    draw_box(ax, cx, yl4, 0.36, 0.045, 'FFN (GELU)', C_PROJ, fontsize=9,
             subtext='4× expand → contract')

    # Residual arrow for FFN
    draw_curved_arrow(ax, cx - 0.19, yl3 - 0.02, cx - 0.19, yl4 + 0.035,
                      GREEN, "arc3,rad=-0.5")
    draw_label(ax, cx - 0.28, (yl3 + yl4)/2, '+', fontsize=11, color=GREEN)

    # Interference module (conditional)
    yl5 = yl4 + 0.075
    draw_arrow(ax, cx, yl4 + 0.023, cx, yl5 - 0.02, TEXT_DIM)
    interf_box = FancyBboxPatch(
        (cx - 0.18, yl5 - 0.022), 0.36, 0.044,
        boxstyle="round,pad=0.02",
        facecolor=C_GATE, edgecolor=PURPLE,
        alpha=0.7, linewidth=1.0, linestyle=':'
    )
    ax.add_patch(interf_box)
    ax.text(cx, yl5, 'Field Interference Module',
            ha='center', va='center', fontsize=8,
            color=PURPLE, family='monospace')
    ax.text(cx + 0.22, yl5, 'every 3rd\nlayer',
            ha='left', va='center', fontsize=6,
            color=PURPLE, family='monospace')

    # Global context
    yl6 = yl5 + 0.065
    draw_arrow(ax, cx, yl5 + 0.022, cx, yl6 - 0.02, TEXT_DIM)
    gc_box = FancyBboxPatch(
        (cx - 0.18, yl6 - 0.022), 0.36, 0.044,
        boxstyle="round,pad=0.02",
        facecolor=C_FIELD, edgecolor=TEAL,
        alpha=0.7, linewidth=1.0, linestyle=':'
    )
    ax.add_patch(gc_box)
    ax.text(cx, yl6, 'Global Context (Causal CumMean)',
            ha='center', va='center', fontsize=8,
            color=TEAL, family='monospace')
    ax.text(cx + 0.22, yl6, 'O(n)',
            ha='left', va='center', fontsize=6,
            color=TEAL, family='monospace')

    # Output
    y_ln = 0.80
    draw_arrow(ax, cx, y_block_top - 0.01, cx, y_ln - 0.02, TEXT_DIM)
    draw_box(ax, cx, y_ln, 0.2, 0.035, 'LayerNorm', C_NORM, fontsize=8)

    y_out = 0.86
    draw_arrow(ax, cx, y_ln + 0.018, cx, y_out - 0.02, TEXT_DIM)
    draw_box(ax, cx, y_out, bw, bh, 'Output Head (tied)', C_INPUT, ACCENT,
             fontsize=10, subtext='weight-tied with embedding')
    draw_label(ax, cx + 0.22, y_out, '(B, N, V)', fontsize=7)


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL 2: Single Layer Detail
# ═══════════════════════════════════════════════════════════════════════════
def draw_panel2_layer(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(BG_PANEL)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(0.5, 0.96, 'Single Transformer Layer',
            ha='center', va='top', fontsize=13, color=TEXT,
            fontweight='bold', family='monospace')
    ax.text(0.5, 0.92, 'Pre-norm residual with wave field attention',
            ha='center', va='top', fontsize=8, color=TEXT_DIM, family='monospace')

    cx = 0.5
    bw = 0.32

    # Input
    y_in = 0.08
    draw_box(ax, cx, y_in, 0.2, 0.04, 'x  (B, N, D)', C_INPUT, ACCENT, fontsize=9)

    # Pre-norm 1
    y_ln1 = 0.16
    draw_arrow(ax, cx, y_in + 0.02, cx, y_ln1 - 0.02, TEXT_DIM)
    draw_box(ax, cx, y_ln1, 0.22, 0.035, 'LayerNorm₁', C_NORM, fontsize=9)

    # Wave Field Attention (big box)
    y_wf = 0.27
    draw_arrow(ax, cx, y_ln1 + 0.018, cx, y_wf - 0.035, TEXT_DIM)
    draw_box(ax, cx, y_wf, bw, 0.06, '★ Wave Field Attention', C_FFT,
             ORANGE, fontsize=10, bold=True, border_color=ORANGE)

    # Components inside WFA
    components = [
        ('QKV + Gate', 0.15, y_wf + 0.045),
        ('Feature Maps φ(Q), φ(K)', 0.35, y_wf + 0.045),
        ('FFT ⊛ Kernel', 0.55, y_wf + 0.045),
        ('SpectralGate', 0.72, y_wf + 0.045),
        ('Gating', 0.88, y_wf + 0.045),
    ]
    for label, xp, yp in components:
        ax.text(xp, yp, label, ha='center', va='bottom', fontsize=5.5,
                color=TEXT_DIM, family='monospace', alpha=0.7)

    # Residual connection 1
    y_add1 = 0.36
    draw_arrow(ax, cx, y_wf + 0.03, cx, y_add1 - 0.015, TEXT_DIM)
    draw_box(ax, cx, y_add1, 0.08, 0.03, '+', '#1a3a1a', GREEN, fontsize=12,
             border_color=GREEN)
    # Skip connection
    draw_curved_arrow(ax, cx - 0.12, y_in + 0.02, cx - 0.06, y_add1,
                      GREEN, "arc3,rad=-0.4")
    draw_label(ax, cx - 0.18, 0.22, 'residual', fontsize=6, color=GREEN)

    # Pre-norm 2
    y_ln2 = 0.43
    draw_arrow(ax, cx, y_add1 + 0.015, cx, y_ln2 - 0.018, TEXT_DIM)
    draw_box(ax, cx, y_ln2, 0.22, 0.035, 'LayerNorm₂', C_NORM, fontsize=9)

    # FFN
    y_ffn_start = 0.50
    y_ffn_end = 0.62

    draw_arrow(ax, cx, y_ln2 + 0.018, cx, y_ffn_start - 0.02, TEXT_DIM)

    # FFN expanded view
    ffn_rect = FancyBboxPatch(
        (cx - 0.17, y_ffn_start - 0.02), 0.34, 0.14,
        boxstyle="round,pad=0.02",
        facecolor=C_PROJ, edgecolor=TEXT_DIM,
        alpha=0.5, linewidth=1.0
    )
    ax.add_patch(ffn_rect)
    ax.text(cx, y_ffn_start + 0.095, 'Feed-Forward Network',
            ha='center', va='center', fontsize=9, color=TEXT,
            fontweight='bold', family='monospace')

    draw_box(ax, cx - 0.06, y_ffn_start + 0.03, 0.12, 0.03,
             'Linear₁', C_PROJ, fontsize=7, subtext='D → 4D')
    draw_box(ax, cx + 0.06, y_ffn_start + 0.03, 0.08, 0.03,
             'GELU', C_GATE, fontsize=7)
    draw_box(ax, cx, y_ffn_start + 0.00, 0.12, 0.03,
             'Linear₂', C_PROJ, fontsize=7, subtext='4D → D')

    # Residual connection 2
    y_add2 = y_ffn_end + 0.04
    draw_arrow(ax, cx, y_ffn_end, cx, y_add2 - 0.015, TEXT_DIM)
    draw_box(ax, cx, y_add2, 0.08, 0.03, '+', '#1a3a1a', GREEN, fontsize=12,
             border_color=GREEN)
    draw_curved_arrow(ax, cx - 0.12, y_add1 + 0.015, cx - 0.06, y_add2,
                      GREEN, "arc3,rad=-0.4")
    draw_label(ax, cx - 0.18, (y_add1 + y_add2)/2, 'residual', fontsize=6, color=GREEN)

    # Output
    y_out = y_add2 + 0.06
    draw_arrow(ax, cx, y_add2 + 0.015, cx, y_out - 0.02, TEXT_DIM)
    draw_box(ax, cx, y_out, 0.2, 0.04, 'x\'  (B, N, D)', C_INPUT, ACCENT, fontsize=9)

    # Param annotations on right side
    annotations = [
        (0.88, 0.20, 'D = embed_dim'),
        (0.88, 0.24, 'H = num_heads'),
        (0.88, 0.28, 'd = D/H (head_dim)'),
        (0.88, 0.32, 'G = field_size'),
        (0.88, 0.36, 'N = seq_len'),
    ]
    for x, y, t in annotations:
        ax.text(x, y, t, ha='right', va='center', fontsize=6,
                color=TEXT_DIM, family='monospace')


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL 3: Wave Field Attention Internals
# ═══════════════════════════════════════════════════════════════════════════
def draw_panel3_attention(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(BG_PANEL)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(0.5, 0.97, '★ Wave Field Attention — Data Flow',
            ha='center', va='top', fontsize=13, color=ORANGE,
            fontweight='bold', family='monospace')
    ax.text(0.5, 0.935, 'The core O(n log n) mechanism replacing O(n²) self-attention',
            ha='center', va='top', fontsize=7.5, color=TEXT_DIM, family='monospace')

    # ── INPUT ──
    y_input = 0.88
    draw_box(ax, 0.5, y_input, 0.18, 0.035, 'x  (B, N, D)', C_INPUT, ACCENT, fontsize=8)

    # ── FUSED QKV + GATE PROJECTION ──
    y_proj = 0.82
    draw_arrow(ax, 0.5, y_input - 0.018, 0.5, y_proj + 0.02, TEXT_DIM)
    draw_box(ax, 0.5, y_proj, 0.30, 0.04, 'Fused QKV+Gate Projection', C_PROJ,
             fontsize=8, subtext='Linear(D → 4D)')

    # Split into Q, K, V, Gate
    y_split = 0.75
    q_x, k_x, v_x, g_x = 0.13, 0.37, 0.63, 0.87

    for x, label, col in [(q_x, 'Q', CYAN), (k_x, 'K', GREEN),
                           (v_x, 'V', PURPLE), (g_x, 'Gate', PINK)]:
        draw_arrow(ax, 0.5, y_proj - 0.02, x, y_split + 0.015, col, lw=0.8)
        draw_box(ax, x, y_split, 0.1, 0.03, label, C_PROJ, col, fontsize=8, bold=True)
        draw_label(ax, x, y_split - 0.025, '(B,H,N,d)', fontsize=5.5)

    # ── FEATURE MAPS ──
    y_fm = 0.67
    draw_arrow(ax, q_x, y_split - 0.015, q_x, y_fm + 0.018, CYAN)
    draw_box(ax, q_x, y_fm, 0.11, 0.035, 'φ_Q', C_FEAT, CYAN, fontsize=8,
             subtext='GELU+ReLU')
    draw_label(ax, q_x - 0.07, y_fm, 'Hedgehog\nfeature map', fontsize=5, color=CYAN)

    draw_arrow(ax, k_x, y_split - 0.015, k_x, y_fm + 0.018, GREEN)
    draw_box(ax, k_x, y_fm, 0.11, 0.035, 'φ_K', C_FEAT, GREEN, fontsize=8,
             subtext='GELU+ReLU')

    # ── K⊙V DEPOSIT ──
    y_dep = 0.59
    draw_arrow(ax, k_x, y_fm - 0.018, 0.5, y_dep + 0.015, GREEN, lw=0.8)
    draw_arrow(ax, v_x, y_split - 0.015, 0.5, y_dep + 0.015, PURPLE, lw=0.8)
    draw_box(ax, 0.5, y_dep, 0.14, 0.03, 'φ_K ⊙ V', C_FEAT, fontsize=8)
    draw_label(ax, 0.5, y_dep - 0.022, 'K-weighted deposit', fontsize=5.5)

    # ── BILINEAR SCATTER ──
    y_scatter = 0.52
    draw_arrow(ax, 0.5, y_dep - 0.015, 0.5, y_scatter + 0.025, TEXT_DIM)
    draw_box(ax, 0.5, y_scatter, 0.18, 0.04, 'Bilinear Scatter', C_FIELD,
             TEAL, fontsize=8, subtext='tokens → field cells')
    draw_label(ax, 0.5 + 0.12, y_scatter, '(B,H,N,d)→(B,H,G,d)', fontsize=5)
    draw_label(ax, 0.5 + 0.12, y_scatter - 0.015, 'stride ≥ 1.0 (causal)', fontsize=5, color=RED)

    # ═══ CORE INNOVATION: FFT CONVOLUTION ═══
    y_fft_top = 0.475
    y_fft_bot = 0.32

    # Dashed box around core
    core_rect = FancyBboxPatch(
        (0.15, y_fft_bot - 0.02), 0.70, y_fft_top - y_fft_bot + 0.04,
        boxstyle="round,pad=0.01",
        facecolor='none', edgecolor=ORANGE,
        linewidth=1.5, linestyle='--', alpha=0.7
    )
    ax.add_patch(core_rect)
    ax.text(0.86, y_fft_top + 0.01, 'Core Innovation',
            ha='right', va='bottom', fontsize=7, color=ORANGE,
            fontweight='bold', family='monospace')

    # FFT of field
    y_fft1 = 0.45
    draw_arrow(ax, 0.5, y_scatter - 0.02, 0.35, y_fft1 + 0.015, TEXT_DIM)
    draw_box(ax, 0.35, y_fft1, 0.13, 0.03, 'FFT(field)', C_FFT, ORANGE, fontsize=8)

    # Kernel FFT (from wave params)
    draw_box(ax, 0.65, y_fft1, 0.18, 0.03, 'Kernel FFT', C_FFT, ORANGE, fontsize=8)
    draw_label(ax, 0.65, y_fft1 + 0.025, 'Z-transform of', fontsize=5)
    draw_label(ax, 0.65, y_fft1 + 0.04, 'damped wave poles', fontsize=5)

    # Wave params feeding into kernel
    wp_x, wp_y = 0.87, y_fft1 + 0.02
    ax.text(wp_x, wp_y + 0.02, 'ω (freq)', ha='center', fontsize=5.5,
            color=CYAN, family='monospace')
    ax.text(wp_x, wp_y + 0.005, 'α (damp)', ha='center', fontsize=5.5,
            color=RED, family='monospace')
    ax.text(wp_x, wp_y - 0.01, 'φ (phase)', ha='center', fontsize=5.5,
            color=GREEN, family='monospace')
    draw_arrow(ax, 0.82, wp_y, 0.75, y_fft1, TEXT_DIM, lw=0.8)

    # Spectral multiply
    y_mul = 0.41
    draw_arrow(ax, 0.35, y_fft1 - 0.015, 0.45, y_mul + 0.012, TEXT_DIM)
    draw_arrow(ax, 0.65, y_fft1 - 0.015, 0.55, y_mul + 0.012, TEXT_DIM)
    draw_box(ax, 0.5, y_mul, 0.08, 0.025, '⊛', C_FFT, ORANGE, fontsize=11)
    draw_label(ax, 0.5, y_mul - 0.02, 'spectral multiply', fontsize=5.5)

    # SpectralGate
    y_sg = 0.41
    sg_x = 0.25
    draw_box(ax, sg_x, y_sg, 0.15, 0.035, 'SpectralGate', C_GATE,
             PINK, fontsize=7, subtext='MLP(q₀) → gate(f)')
    draw_arrow(ax, sg_x + 0.08, y_sg, 0.46, y_mul, PINK, lw=0.8)
    # Q[0] feeding SpectralGate
    draw_curved_arrow(ax, q_x, y_fm - 0.018, sg_x - 0.05, y_sg + 0.01,
                      PINK, "arc3,rad=0.3")
    draw_label(ax, sg_x - 0.07, y_sg + 0.03, 'q[:,:,0,:]', fontsize=5, color=PINK)
    draw_label(ax, sg_x - 0.07, y_sg + 0.015, '(token 0 only)', fontsize=4.5, color=PINK)

    # IFFT
    y_ifft = 0.36
    draw_arrow(ax, 0.5, y_mul - 0.013, 0.5, y_ifft + 0.012, TEXT_DIM)
    draw_box(ax, 0.5, y_ifft, 0.13, 0.03, 'IFFT → field', C_FFT, ORANGE, fontsize=8)

    # Cross-head coupling
    y_couple = 0.33
    draw_arrow(ax, 0.5, y_ifft - 0.015, 0.5, y_couple + 0.012, TEXT_DIM)
    draw_box(ax, 0.5, y_couple, 0.18, 0.025, 'Cross-Head Coupling', C_PROJ,
             fontsize=7)
    draw_label(ax, 0.5 + 0.12, y_couple, 'H×H matrix', fontsize=5)

    # ═══ END CORE ═══

    # Bilinear Gather
    y_gather = 0.26
    draw_arrow(ax, 0.5, y_couple - 0.013, 0.5, y_gather + 0.02, TEXT_DIM)
    draw_box(ax, 0.5, y_gather, 0.18, 0.04, 'Bilinear Gather', C_FIELD,
             TEAL, fontsize=8, subtext='field cells → tokens')

    # Q-weighted reading
    y_read = 0.19
    draw_arrow(ax, 0.5, y_gather - 0.02, 0.5, y_read + 0.015, TEXT_DIM)
    draw_arrow(ax, q_x, y_fm - 0.018, 0.35, y_read + 0.01, CYAN, lw=0.8)
    draw_box(ax, 0.5, y_read, 0.16, 0.03, 'φ_Q ⊙ gathered', C_FEAT, CYAN,
             fontsize=8)
    draw_label(ax, 0.5, y_read - 0.022, 'Q-weighted reading', fontsize=5.5)

    # Gating
    y_gate = 0.12
    draw_arrow(ax, 0.5, y_read - 0.015, 0.5, y_gate + 0.015, TEXT_DIM)
    draw_arrow(ax, g_x, y_split - 0.015, 0.62, y_gate + 0.01, PINK, lw=0.8)
    draw_box(ax, 0.5, y_gate, 0.14, 0.03, 'σ(Gate) ⊙ out', C_GATE,
             PINK, fontsize=8)
    draw_label(ax, 0.5, y_gate - 0.022, 'content-dependent gate', fontsize=5.5)

    # Output projection
    y_out = 0.06
    draw_arrow(ax, 0.5, y_gate - 0.015, 0.5, y_out + 0.015, TEXT_DIM)
    draw_box(ax, 0.5, y_out, 0.18, 0.035, 'Linear(D → D)', C_PROJ, fontsize=8)

    # Final output
    y_final = 0.015
    draw_arrow(ax, 0.5, y_out - 0.018, 0.5, y_final + 0.01, ACCENT)
    draw_box(ax, 0.5, y_final, 0.16, 0.025, 'output (B,N,D)', C_INPUT, ACCENT, fontsize=7)

    # Complexity annotation
    ax.text(0.92, 0.06, 'Total: O(n log n)\nFFT dominates',
            ha='right', va='center', fontsize=6,
            color=ORANGE, family='monospace',
            bbox=dict(boxstyle='round', facecolor=BG, edgecolor=ORANGE, alpha=0.5))


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL 3 WIDE: Wave Field Attention — Landscape Layout
# ═══════════════════════════════════════════════════════════════════════════
def draw_panel3_attention_wide(ax):
    """Landscape layout: left-to-right data flow for attention internals."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(BG_PANEL)
    ax.axis('off')

    ax.text(0.5, 0.97, '★ Wave Field Attention — Internal Data Flow',
            ha='center', va='top', fontsize=14, color=ORANGE,
            fontweight='bold', family='monospace')
    ax.text(0.5, 0.93, 'O(n log n) FFT convolution on continuous wave fields replaces O(n²) dot-product attention',
            ha='center', va='top', fontsize=8.5, color=TEXT_DIM, family='monospace')

    # Layout: left-to-right flow with main path at y=0.5
    # ── Step 1: Input + Projection ──
    bh = 0.07
    y_main = 0.50  # main flow line

    x_in = 0.04
    draw_box(ax, x_in, y_main, 0.05, bh, 'x', C_INPUT, ACCENT, fontsize=10, bold=True)
    draw_label(ax, x_in, y_main - 0.055, '(B,N,D)', fontsize=7)

    # Fused projection
    x_proj = 0.11
    draw_arrow(ax, x_in + 0.025, y_main, x_proj - 0.03, y_main, TEXT_DIM)
    draw_box(ax, x_proj, y_main, 0.055, bh, 'QKV\n+Gate', C_PROJ, TEXT, fontsize=8)
    draw_label(ax, x_proj, y_main - 0.055, 'Linear\nD→4D', fontsize=6)

    # Split into Q, K, V, Gate (fan out vertically)
    x_split = 0.18
    y_q = y_main + 0.24
    y_k = y_main + 0.10
    y_v = y_main - 0.05
    y_g = y_main - 0.22

    for yy, label, col in [(y_q, 'Q', CYAN), (y_k, 'K', GREEN),
                             (y_v, 'V', PURPLE), (y_g, 'Gate', PINK)]:
        draw_arrow(ax, x_proj + 0.028, y_main, x_split - 0.02, yy, col, lw=1.0)
        draw_box(ax, x_split, yy, 0.04, 0.05, label, C_PROJ, col, fontsize=9, bold=True)

    # ── Step 2: Feature Maps ──
    x_fm = 0.24
    draw_arrow(ax, x_split + 0.02, y_q, x_fm - 0.025, y_q, CYAN)
    draw_box(ax, x_fm, y_q, 0.05, 0.05, 'φ(Q)', C_FEAT, CYAN, fontsize=9)
    draw_label(ax, x_fm, y_q + 0.04, 'Hedgehog', fontsize=6, color=CYAN)
    draw_label(ax, x_fm, y_q + 0.055, 'GELU→ReLU', fontsize=5.5, color=CYAN)

    draw_arrow(ax, x_split + 0.02, y_k, x_fm - 0.025, y_k, GREEN)
    draw_box(ax, x_fm, y_k, 0.05, 0.05, 'φ(K)', C_FEAT, GREEN, fontsize=9)

    # ── Step 3: K⊙V Deposit ──
    x_dep = 0.31
    draw_arrow(ax, x_fm + 0.025, y_k, x_dep - 0.025, y_k, GREEN)
    draw_arrow(ax, x_split + 0.02, y_v, x_dep - 0.025, y_k - 0.01, PURPLE, lw=0.8)
    draw_box(ax, x_dep, y_k, 0.05, 0.06, 'φ(K)\n⊙ V', C_FEAT, TEXT, fontsize=8)
    draw_label(ax, x_dep, y_k - 0.045, 'deposit', fontsize=6)

    # ── Step 4: Bilinear Scatter ──
    x_scatter = 0.38
    draw_arrow(ax, x_dep + 0.025, y_k, x_scatter - 0.03, y_main, TEAL)
    draw_box(ax, x_scatter, y_main, 0.055, bh, 'Scatter', C_FIELD, TEAL,
             fontsize=9, bold=True)
    draw_label(ax, x_scatter, y_main - 0.055, 'tokens→field', fontsize=6)
    draw_label(ax, x_scatter, y_main - 0.075, '(B,H,N,d)→(B,H,G,d)', fontsize=5)
    draw_label(ax, x_scatter, y_main + 0.055, 'stride≥1.0', fontsize=5.5, color=RED)

    # ═══ CORE: FFT Convolution (highlighted zone) ═══
    core_x1, core_x2 = 0.43, 0.68
    core_rect = FancyBboxPatch(
        (core_x1, y_main - 0.14), core_x2 - core_x1, 0.42,
        boxstyle="round,pad=0.01",
        facecolor='#1a1500', edgecolor=ORANGE,
        linewidth=2.0, linestyle='--', alpha=0.4
    )
    ax.add_patch(core_rect)
    ax.text((core_x1 + core_x2)/2, y_main + 0.295, 'Core Innovation: FFT Convolution',
            ha='center', va='top', fontsize=10, color=ORANGE,
            fontweight='bold', family='monospace')
    ax.text((core_x1 + core_x2)/2, y_main + 0.26, 'O(n log n) — same kernel for all positions (shift-invariant)',
            ha='center', va='top', fontsize=7, color=ORANGE, family='monospace', alpha=0.8)

    # FFT of field
    x_fft = 0.465
    draw_arrow(ax, x_scatter + 0.028, y_main, x_fft - 0.025, y_main, ORANGE)
    draw_box(ax, x_fft, y_main, 0.045, bh, 'FFT', C_FFT, ORANGE, fontsize=10, bold=True)
    draw_label(ax, x_fft, y_main - 0.055, 'field→freq', fontsize=6)

    # Spectral multiply
    x_mul = 0.53
    draw_arrow(ax, x_fft + 0.023, y_main, x_mul - 0.02, y_main, ORANGE)
    draw_box(ax, x_mul, y_main, 0.035, bh, '⊛', C_FFT, ORANGE, fontsize=14)

    # Kernel FFT (from above)
    y_kernel = y_main + 0.16
    draw_box(ax, x_mul, y_kernel, 0.07, 0.06, 'Kernel\nFFT', C_FFT, ORANGE,
             fontsize=9, bold=True)
    draw_arrow(ax, x_mul, y_kernel - 0.03, x_mul, y_main + bh/2, ORANGE)

    # Wave params
    x_params = x_mul + 0.06
    ax.text(x_params, y_kernel + 0.04, 'ω (freq)', ha='left', fontsize=7,
            color=CYAN, family='monospace')
    ax.text(x_params, y_kernel + 0.015, 'α (damp) ≤ 0.5', ha='left', fontsize=7,
            color=RED, family='monospace')
    ax.text(x_params, y_kernel - 0.01, 'φ (phase)', ha='left', fontsize=7,
            color=GREEN, family='monospace')
    ax.text(x_params, y_kernel - 0.035, '3 learnable/head', ha='left', fontsize=6,
            color=TEXT_DIM, family='monospace')

    # SpectralGate (from below)
    y_sg = y_main - 0.10
    draw_box(ax, x_mul, y_sg, 0.08, 0.05, 'Spectral\nGate', C_GATE, PINK, fontsize=8)
    draw_arrow(ax, x_mul, y_sg + 0.025, x_mul, y_main - bh/2, PINK)
    draw_label(ax, x_mul - 0.05, y_sg, 'MLP(q₀)', fontsize=6, color=PINK)
    draw_label(ax, x_mul - 0.05, y_sg - 0.02, 'per-sample', fontsize=5.5, color=PINK)
    # Q feeding SG
    draw_curved_arrow(ax, x_fm + 0.025, y_q - 0.02, x_mul - 0.04, y_sg + 0.02,
                      PINK, "arc3,rad=0.4")
    draw_label(ax, 0.36, y_q - 0.10, 'q[:,:,0,:]', fontsize=6, color=PINK)

    # IFFT
    x_ifft = 0.585
    draw_arrow(ax, x_mul + 0.018, y_main, x_ifft - 0.02, y_main, ORANGE)
    draw_box(ax, x_ifft, y_main, 0.045, bh, 'IFFT', C_FFT, ORANGE, fontsize=10, bold=True)
    draw_label(ax, x_ifft, y_main - 0.055, 'freq→field', fontsize=6)

    # Cross-head coupling
    x_couple = 0.65
    draw_arrow(ax, x_ifft + 0.023, y_main, x_couple - 0.02, y_main, TEXT_DIM)
    draw_box(ax, x_couple, y_main, 0.04, bh, 'C', C_PROJ, TEXT, fontsize=10, bold=True)
    draw_label(ax, x_couple, y_main - 0.055, 'H×H\ncoupling', fontsize=6)

    # ═══ END CORE ═══

    # ── Step 7: Bilinear Gather ──
    x_gather = 0.72
    draw_arrow(ax, x_couple + 0.02, y_main, x_gather - 0.028, y_main, TEAL)
    draw_box(ax, x_gather, y_main, 0.055, bh, 'Gather', C_FIELD, TEAL,
             fontsize=9, bold=True)
    draw_label(ax, x_gather, y_main - 0.055, 'field→tokens', fontsize=6)
    draw_label(ax, x_gather, y_main - 0.075, '(B,H,G,d)→(B,H,N,d)', fontsize=5)

    # ── Step 8: Q-weighted read ──
    x_read = 0.80
    draw_arrow(ax, x_gather + 0.028, y_main, x_read - 0.025, y_main, TEXT_DIM)
    draw_box(ax, x_read, y_main, 0.045, bh, 'φ(Q)\n⊙', C_FEAT, CYAN, fontsize=9)
    draw_label(ax, x_read, y_main - 0.055, 'Q-read', fontsize=6)
    # Q feature map feeding read
    draw_curved_arrow(ax, x_fm + 0.025, y_q, x_read - 0.01, y_main + bh/2 + 0.01,
                      CYAN, "arc3,rad=-0.2")

    # ── Step 9: Gating ──
    x_gate = 0.87
    draw_arrow(ax, x_read + 0.023, y_main, x_gate - 0.025, y_main, TEXT_DIM)
    draw_box(ax, x_gate, y_main, 0.05, bh, 'σ(G)\n⊙', C_GATE, PINK, fontsize=9)
    draw_label(ax, x_gate, y_main - 0.055, 'gate', fontsize=6)
    # Gate feeding
    draw_curved_arrow(ax, x_split + 0.02, y_g, x_gate, y_main - bh/2 - 0.01,
                      PINK, "arc3,rad=0.3")

    # ── Step 10: Output ──
    x_out = 0.94
    draw_arrow(ax, x_gate + 0.025, y_main, x_out - 0.025, y_main, ACCENT)
    draw_box(ax, x_out, y_main, 0.05, bh, 'out', C_INPUT, ACCENT, fontsize=10, bold=True)
    draw_label(ax, x_out, y_main - 0.055, 'Linear\nD→D', fontsize=6)

    # ── Complexity annotations ──
    ax.text(0.95, 0.12, 'Total complexity: O(n log n)',
            ha='right', va='center', fontsize=9,
            color=ORANGE, family='monospace', fontweight='bold')
    ax.text(0.95, 0.07, 'FFT/IFFT: O(n log n)  |  Scatter/Gather: O(n)\n'
            'Feature Maps: O(n)  |  SpectralGate: O(1)  |  Coupling: O(H²)',
            ha='right', va='center', fontsize=7,
            color=TEXT_DIM, family='monospace')


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN: Compose all panels
# ═══════════════════════════════════════════════════════════════════════════
def main():
    fig = plt.figure(figsize=(28, 22), facecolor=BG)
    fig.suptitle('Wave Field LLM V4.3.9 — SPECTRE-Wave Architecture',
                 fontsize=20, color=TEXT, fontweight='bold', y=0.985,
                 family='monospace')
    fig.text(0.5, 0.97,
             'Physics-based O(n log n) language model replacing O(n²) self-attention '
             'with damped wave equation dynamics',
             ha='center', fontsize=11, color=TEXT_DIM, family='monospace')

    # Top row: Full model + Single layer (side by side)
    ax1 = fig.add_axes([0.02, 0.50, 0.45, 0.45])   # Full model
    ax2 = fig.add_axes([0.52, 0.50, 0.45, 0.45])   # Single layer

    # Bottom row: Attention detail (full width, landscape)
    ax3 = fig.add_axes([0.02, 0.02, 0.96, 0.44])   # Attention detail

    draw_panel1_full_model(ax1)
    draw_panel2_layer(ax2)
    draw_panel3_attention_wide(ax3)

    # Panel labels
    for ax, label in [(ax1, 'A'), (ax2, 'B'), (ax3, 'C')]:
        ax.text(0.01, 0.99, label, transform=ax.transAxes,
                fontsize=18, color=ACCENT, fontweight='bold',
                va='top', family='monospace')

    out_path = 'results/figures/v439_architecture_full.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    print(f'Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
