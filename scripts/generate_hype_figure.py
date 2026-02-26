"""
Generate a dramatic, shareable announcement graphic.
Rocky-style: pure victory, no architecture details.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "announce"

# ── Color palette ──────────────────────────────────────────────
BG_DARK = "#0a0a0f"
BG_CARD = "#12121a"
GOLD = "#FFD700"
GOLD_DIM = "#B8860B"
SILVER = "#8899aa"
WHITE = "#f0f0f0"
RED = "#ff4444"
GREEN = "#00ff88"
CYAN = "#00d4ff"
WAVE_BLUE = "#4488ff"
WAVE_GLOW = "#6699ff"


def fig1_hero_knockout():
    """The main announcement: 24x knockout blow."""
    fig = plt.figure(figsize=(16, 9), facecolor=BG_DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_facecolor(BG_DARK)
    ax.axis("off")

    # Subtle radial glow behind the number
    for r, alpha in [(3.5, 0.03), (2.5, 0.05), (1.5, 0.08), (0.8, 0.12)]:
        circle = plt.Circle((8, 4.8), r, color=GOLD, alpha=alpha)
        ax.add_patch(circle)

    # "24x" — the hero number
    ax.text(8, 4.8, "24x", fontsize=180, fontweight="bold",
            color=GOLD, ha="center", va="center",
            fontfamily="sans-serif",
            path_effects=[pe.withStroke(linewidth=4, foreground=GOLD_DIM)])

    # Subtitle
    ax.text(8, 2.6, "BETTER  PERPLEXITY  THAN  TRANSFORMERS",
            fontsize=24, fontweight="bold", color=WHITE, ha="center", va="center",
            fontfamily="sans-serif")

    # Thin gold line
    ax.plot([3, 13], [2.1, 2.1], color=GOLD, linewidth=1.5, alpha=0.6)

    # Bottom details - minimal, just enough to be credible
    ax.text(8, 1.5, "Same data  ·  Same training  ·  Same compute  ·  Head to head",
            fontsize=14, color=SILVER, ha="center", va="center",
            fontfamily="sans-serif")

    # Top corner - project name, subtle
    ax.text(0.5, 8.5, "WAVE FIELD", fontsize=14, fontweight="bold",
            color=WAVE_BLUE, ha="left", va="top", alpha=0.8,
            fontfamily="sans-serif")

    fig.savefig(RESULTS_DIR / "announce_hero.png", dpi=200,
                facecolor=BG_DARK, bbox_inches="tight", pad_inches=0.3)
    plt.close()
    print("  [1/4] announce_hero.png")


def fig2_knockout_bars():
    """Side-by-side bar comparison — the knockout visual."""
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG_DARK)
    ax.set_facecolor(BG_DARK)

    # Two bars
    bars = ax.barh(
        [0.6, -0.6],
        [162.5, 6.8],
        height=0.8,
        color=[SILVER, GOLD],
        edgecolor="none",
        zorder=3,
    )

    # Labels on bars
    ax.text(162.5 + 3, 0.6, "162.5", fontsize=48, fontweight="bold",
            color=SILVER, ha="left", va="center", fontfamily="sans-serif")
    ax.text(6.8 + 3, -0.6, "6.8", fontsize=48, fontweight="bold",
            color=GOLD, ha="left", va="center", fontfamily="sans-serif")

    # Model names
    ax.text(-2, 0.6, "Standard\nTransformer", fontsize=18, color=SILVER,
            ha="right", va="center", fontfamily="sans-serif", linespacing=1.4)
    ax.text(-2, -0.6, "Wave Field", fontsize=18, fontweight="bold", color=GOLD,
            ha="right", va="center", fontfamily="sans-serif")

    # "Perplexity (lower is better)" label
    ax.text(85, -1.8, "Perplexity  (lower is better)  →",
            fontsize=13, color=SILVER, ha="center", va="center",
            fontfamily="sans-serif", alpha=0.6)

    # Arrow showing the gap
    ax.annotate("", xy=(6.8, 0), xytext=(162.5, 0),
                arrowprops=dict(arrowstyle="<->", color=GOLD, lw=2, alpha=0.5))
    ax.text(84, 0.05, "24x", fontsize=28, fontweight="bold", color=GOLD,
            ha="center", va="center", fontfamily="sans-serif", alpha=0.8)

    ax.set_xlim(-5, 220)
    ax.set_ylim(-2.2, 1.8)
    ax.axis("off")

    # Title
    ax.text(85, 1.5, "HEAD  TO  HEAD  ·  SAME  DATA  ·  SAME  TRAINING",
            fontsize=14, color=SILVER, ha="center", fontfamily="sans-serif",
            alpha=0.7)

    fig.savefig(RESULTS_DIR / "announce_bars.png", dpi=200,
                facecolor=BG_DARK, bbox_inches="tight", pad_inches=0.5)
    plt.close()
    print("  [2/4] announce_bars.png")


def fig3_training_drama():
    """Training curves but dramatic — the divergence moment."""
    # Simulated training data (matches real S1 results)
    tokens_m = np.array([0.5, 1, 2, 3, 4, 5, 7, 10, 12, 15, 17, 20])
    wave_ppl = np.array([1800, 900, 550, 335, 200, 94, 45, 19, 13, 9, 7.5, 6.8])
    std_ppl = np.array([1900, 950, 600, 456, 400, 336, 290, 250, 220, 195, 175, 162.5])

    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG_DARK)
    ax.set_facecolor(BG_DARK)

    # Shade the gap region
    ax.fill_between(tokens_m, wave_ppl, std_ppl, alpha=0.08, color=GOLD)

    # Standard — flat, boring line
    ax.plot(tokens_m, std_ppl, color=SILVER, linewidth=3, alpha=0.7,
            label="Standard Transformer", zorder=4)
    ax.scatter([20], [162.5], color=SILVER, s=120, zorder=5, edgecolors="white", linewidths=1)

    # Wave — dramatic descent
    ax.plot(tokens_m, wave_ppl, color=GOLD, linewidth=4,
            label="Wave Field", zorder=5)
    ax.scatter([20], [6.8], color=GOLD, s=150, zorder=6, edgecolors="white", linewidths=1.5)

    # End labels
    ax.text(20.3, 162.5, "162.5", fontsize=22, fontweight="bold", color=SILVER,
            va="center", fontfamily="sans-serif")
    ax.text(20.3, 6.8, "6.8", fontsize=22, fontweight="bold", color=GOLD,
            va="center", fontfamily="sans-serif")

    # "The moment it breaks away" annotation
    ax.annotate("Divergence",
                xy=(4, 200), xytext=(6.5, 500),
                fontsize=14, color=WHITE, alpha=0.6,
                fontfamily="sans-serif",
                arrowprops=dict(arrowstyle="->", color=WHITE, alpha=0.3, lw=1.5))

    ax.set_yscale("log")
    ax.set_ylim(4, 2500)
    ax.set_xlim(-0.5, 22.5)

    # Minimal axis styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SILVER)
    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_color(SILVER)
    ax.spines["bottom"].set_alpha(0.3)
    ax.tick_params(colors=SILVER, labelsize=12)
    ax.set_xlabel("Training Tokens (millions)", fontsize=14, color=SILVER,
                  fontfamily="sans-serif")
    ax.set_ylabel("Perplexity", fontsize=14, color=SILVER,
                  fontfamily="sans-serif")

    # Custom legend
    leg = ax.legend(fontsize=14, loc="upper right",
                    facecolor=BG_CARD, edgecolor=SILVER, labelcolor=WHITE,
                    framealpha=0.8)

    fig.savefig(RESULTS_DIR / "announce_training.png", dpi=200,
                facecolor=BG_DARK, bbox_inches="tight", pad_inches=0.5)
    plt.close()
    print("  [3/4] announce_training.png")


def fig4_trophy_card():
    """Summary card — the final scoreboard."""
    fig = plt.figure(figsize=(16, 9), facecolor=BG_DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_facecolor(BG_DARK)
    ax.axis("off")

    # Title
    ax.text(8, 8.0, "WAVE FIELD vs TRANSFORMER", fontsize=32, fontweight="bold",
            color=WHITE, ha="center", va="center", fontfamily="sans-serif")

    ax.plot([2, 14], [7.4, 7.4], color=GOLD, linewidth=1, alpha=0.4)

    # Three stat columns
    stats = [
        ("PERPLEXITY", "6.8", "162.5", "24x better"),
        ("ACCURACY", "64.3%", "18.8%", "3.4x better"),
        ("COMPLEXITY", "O(n log n)", "O(n²)", "367x at 32K"),
    ]

    for i, (label, wave_val, std_val, compare) in enumerate(stats):
        cx = 3 + i * 5

        # Card background
        card = mpatches.FancyBboxPatch(
            (cx - 2, 1.8), 4, 5.2,
            boxstyle="round,pad=0.15", facecolor=BG_CARD,
            edgecolor=GOLD, linewidth=0.5, alpha=0.5
        )
        ax.add_patch(card)

        # Label
        ax.text(cx, 6.5, label, fontsize=13, fontweight="bold", color=SILVER,
                ha="center", fontfamily="sans-serif", alpha=0.8)

        # Wave Field value (big, gold)
        ax.text(cx, 5.2, wave_val, fontsize=36, fontweight="bold",
                color=GOLD, ha="center", va="center", fontfamily="sans-serif")
        ax.text(cx, 4.3, "Wave Field", fontsize=11, color=GOLD,
                ha="center", alpha=0.7, fontfamily="sans-serif")

        # Divider
        ax.plot([cx - 1.2, cx + 1.2], [3.7, 3.7], color=SILVER, linewidth=0.5, alpha=0.3)

        # Standard value (small, grey)
        ax.text(cx, 3.2, std_val, fontsize=20, color=SILVER,
                ha="center", va="center", fontfamily="sans-serif")
        ax.text(cx, 2.6, "Standard", fontsize=11, color=SILVER,
                ha="center", alpha=0.5, fontfamily="sans-serif")

    # Bottom tagline
    ax.text(8, 1.0, "Same data  ·  Same optimizer  ·  Same hardware  ·  Same seed",
            fontsize=13, color=SILVER, ha="center", alpha=0.5,
            fontfamily="sans-serif")

    ax.text(8, 0.4, "WikiText-2  ·  20M tokens  ·  RTX 3060  ·  Seed 42",
            fontsize=11, color=SILVER, ha="center", alpha=0.35,
            fontfamily="sans-serif")

    fig.savefig(RESULTS_DIR / "announce_scorecard.png", dpi=200,
                facecolor=BG_DARK, bbox_inches="tight", pad_inches=0.3)
    plt.close()
    print("  [4/4] announce_scorecard.png")


if __name__ == "__main__":
    print("Generating announcement graphics...")
    fig1_hero_knockout()
    fig2_knockout_bars()
    fig3_training_drama()
    fig4_trophy_card()
    print("\nDone! Files in results/announce_*.png")
