"""
Wave Field LLM V4.3.9 — Architecture Visualization with Manim
==============================================================
Renders the full SPECTRE-Wave attention pipeline as a publication-quality diagram.

Usage:
  manim -pql scripts/manim_architecture.py WaveFieldPipeline    # low quality preview
  manim -pqh scripts/manim_architecture.py WaveFieldPipeline    # high quality
  manim -sqh scripts/manim_architecture.py WaveFieldPipeline    # save last frame (no preview)
"""

from manim import *

# ── Color palette ──────────────────────────────────────────────────────────
BG        = "#0d1117"
C_BLUE    = "#58a6ff"
C_GREEN   = "#3fb950"
C_ORANGE  = "#d29922"
C_PURPLE  = "#bc8cff"
C_RED     = "#f85149"
C_PINK    = "#f778ba"
C_TEAL    = "#39d353"
C_CYAN    = "#79c0ff"
C_DIM     = "#484f58"
C_TEXT    = "#e6edf3"

# Box fill colors
F_INPUT   = "#1f3a5f"
F_PROJ    = "#2d333b"
F_FEAT    = "#2a4858"
F_FIELD   = "#1a3528"
F_FFT     = "#3b2a1a"
F_GATE    = "#3a1a3a"
F_NORM    = "#2a2a3a"


def make_box(label, sublabel=None, width=1.8, height=0.65,
             fill=F_PROJ, border=C_DIM, text_color=C_TEXT,
             font_size=22, sublabel_size=16, corner_radius=0.12):
    """Create a rounded rectangle with centered label and optional sublabel."""
    rect = RoundedRectangle(
        corner_radius=corner_radius, width=width, height=height,
        fill_color=fill, fill_opacity=0.92,
        stroke_color=border, stroke_width=1.5,
    )
    txt = Text(label, font_size=font_size, color=text_color, font="Consolas")
    txt.move_to(rect.get_center())

    if sublabel:
        txt.shift(UP * 0.08)
        sub = Text(sublabel, font_size=sublabel_size, color=C_DIM, font="Consolas")
        sub.next_to(txt, DOWN, buff=0.06)
        group = VGroup(rect, txt, sub)
    else:
        group = VGroup(rect, txt)

    return group


def make_arrow(start_mob, end_mob, color=C_DIM, buff=0.08):
    """Arrow between two mobjects."""
    return Arrow(
        start_mob.get_right(), end_mob.get_left(),
        buff=buff, color=color, stroke_width=2.5,
        max_tip_length_to_length_ratio=0.15,
    )


def make_arrow_down(start_mob, end_mob, color=C_DIM, buff=0.08):
    """Vertical arrow."""
    return Arrow(
        start_mob.get_bottom(), end_mob.get_top(),
        buff=buff, color=color, stroke_width=2.5,
        max_tip_length_to_length_ratio=0.15,
    )


def make_curved_arrow(start_pos, end_pos, color=C_DIM, angle=PI/4):
    """Curved arrow between positions."""
    return CurvedArrow(
        start_pos, end_pos,
        color=color, stroke_width=2.0, angle=angle,
        tip_length=0.15,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Scene 1: Full Attention Pipeline (landscape, left-to-right)
# ═══════════════════════════════════════════════════════════════════════════
class WaveFieldPipeline(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ── Title ──
        title = Text(
            "★ Wave Field Attention — SPECTRE-Wave V4.3.9",
            font_size=30, color=C_ORANGE, font="Consolas", weight=BOLD,
        ).to_edge(UP, buff=0.25)
        subtitle = Text(
            "O(n log n) FFT convolution on continuous wave fields replaces O(n²) self-attention",
            font_size=16, color=C_DIM, font="Consolas",
        ).next_to(title, DOWN, buff=0.12)
        self.add(title, subtitle)

        # ── Build pipeline boxes (left to right) ──
        y_main = -0.3

        # 1. Input
        b_input = make_box("x", "(B, N, D)", width=1.0, height=0.7,
                           fill=F_INPUT, border=C_BLUE, text_color=C_BLUE,
                           font_size=26)
        b_input.move_to(LEFT * 6.2 + UP * y_main)

        # 2. QKV + Gate
        b_qkvg = make_box("QKV+G", "Linear D→4D", width=1.2, height=0.7,
                          fill=F_PROJ, font_size=22)
        b_qkvg.move_to(LEFT * 4.8 + UP * y_main)

        # 3. Split: Q, K, V, Gate (vertical fan)
        y_q = y_main + 1.8
        y_k = y_main + 0.7
        y_v = y_main - 0.5
        y_g = y_main - 1.6
        x_split = -3.3

        b_q = make_box("Q", width=0.7, height=0.5, fill=F_PROJ,
                       border=C_CYAN, text_color=C_CYAN, font_size=24)
        b_q.move_to(RIGHT * x_split + UP * y_q)

        b_k = make_box("K", width=0.7, height=0.5, fill=F_PROJ,
                       border=C_GREEN, text_color=C_GREEN, font_size=24)
        b_k.move_to(RIGHT * x_split + UP * y_k)

        b_v = make_box("V", width=0.7, height=0.5, fill=F_PROJ,
                       border=C_PURPLE, text_color=C_PURPLE, font_size=24)
        b_v.move_to(RIGHT * x_split + UP * y_v)

        b_gate = make_box("Gate", width=0.7, height=0.5, fill=F_PROJ,
                          border=C_PINK, text_color=C_PINK, font_size=20)
        b_gate.move_to(RIGHT * x_split + UP * y_g)

        # 4. Feature maps
        x_fm = -2.2

        b_fq = make_box("φ(Q)", "GELU→ReLU", width=1.0, height=0.6,
                        fill=F_FEAT, border=C_CYAN, text_color=C_CYAN, font_size=22)
        b_fq.move_to(RIGHT * x_fm + UP * y_q)

        b_fk = make_box("φ(K)", "GELU→ReLU", width=1.0, height=0.6,
                        fill=F_FEAT, border=C_GREEN, text_color=C_GREEN, font_size=22)
        b_fk.move_to(RIGHT * x_fm + UP * y_k)

        # Hedgehog label
        lbl_hedge = Text("Hedgehog\nFeature Maps", font_size=13, color=C_CYAN,
                         font="Consolas").next_to(b_fq, UP, buff=0.12)

        # 5. K⊙V Deposit
        x_dep = -1.0
        b_dep = make_box("φ(K) ⊙ V", "deposit", width=1.3, height=0.6,
                         fill=F_FEAT, font_size=22)
        b_dep.move_to(RIGHT * x_dep + UP * (y_k + y_v) / 2)

        # 6. Bilinear Scatter
        x_scat = 0.2
        b_scat = make_box("Scatter", "tokens → field", width=1.3, height=0.7,
                          fill=F_FIELD, border=C_TEAL, text_color=C_TEAL, font_size=22)
        b_scat.move_to(RIGHT * x_scat + UP * y_main)

        stride_lbl = Text("stride ≥ 1.0 (causal)", font_size=12, color=C_RED,
                          font="Consolas").next_to(b_scat, DOWN, buff=0.08)

        # ═══ CORE: FFT Convolution ═══
        x_fft = 1.6
        x_mul = 2.6
        x_ifft = 3.6

        b_fft = make_box("FFT", width=0.9, height=0.65,
                         fill=F_FFT, border=C_ORANGE, text_color=C_ORANGE,
                         font_size=26)
        b_fft.move_to(RIGHT * x_fft + UP * y_main)

        b_mul = make_box("⊛", width=0.65, height=0.65,
                         fill=F_FFT, border=C_ORANGE, text_color=C_ORANGE,
                         font_size=32)
        b_mul.move_to(RIGHT * x_mul + UP * y_main)

        b_ifft = make_box("IFFT", width=0.9, height=0.65,
                          fill=F_FFT, border=C_ORANGE, text_color=C_ORANGE,
                          font_size=26)
        b_ifft.move_to(RIGHT * x_ifft + UP * y_main)

        # Kernel FFT (above multiply)
        b_kernel = make_box("Kernel FFT", "Z-transform", width=1.5, height=0.65,
                            fill=F_FFT, border=C_ORANGE, text_color=C_ORANGE,
                            font_size=20)
        b_kernel.move_to(RIGHT * x_mul + UP * (y_main + 1.4))

        # Wave params
        params_txt = VGroup(
            Text("ω  freq", font_size=14, color=C_CYAN, font="Consolas"),
            Text("α  damp ≤ 0.5", font_size=14, color=C_RED, font="Consolas"),
            Text("φ  phase", font_size=14, color=C_GREEN, font="Consolas"),
            Text("3 learnable / head", font_size=12, color=C_DIM, font="Consolas"),
        ).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
        params_txt.next_to(b_kernel, RIGHT, buff=0.2)

        # SpectralGate (below multiply)
        b_sg = make_box("Spectral\nGate", "MLP(q₀)", width=1.3, height=0.7,
                        fill=F_GATE, border=C_PINK, text_color=C_PINK,
                        font_size=20)
        b_sg.move_to(RIGHT * x_mul + UP * (y_main - 1.4))

        sg_note = Text("per-sample\n(not per-token)", font_size=11, color=C_PINK,
                       font="Consolas").next_to(b_sg, DOWN, buff=0.08)

        # Coupling
        x_coup = 4.5
        b_coup = make_box("C", "H×H coupling", width=0.9, height=0.65,
                          fill=F_PROJ, font_size=24)
        b_coup.move_to(RIGHT * x_coup + UP * y_main)

        # ═══ END CORE ═══

        # 7. Bilinear Gather
        x_gath = 5.4
        b_gath = make_box("Gather", "field → tokens", width=1.3, height=0.7,
                          fill=F_FIELD, border=C_TEAL, text_color=C_TEAL, font_size=22)
        b_gath.move_to(RIGHT * x_gath + UP * y_main)

        # 8. Q-weighted read
        x_read = 6.5
        b_read = make_box("φ(Q)⊙", "Q-read", width=1.0, height=0.6,
                          fill=F_FEAT, border=C_CYAN, text_color=C_CYAN, font_size=20)
        b_read.move_to(RIGHT * x_read + UP * y_main)

        # ── Core innovation box (dashed border) ──
        core_rect = RoundedRectangle(
            corner_radius=0.2, width=5.2, height=4.2,
            stroke_color=C_ORANGE, stroke_width=2.0,
            fill_opacity=0.03, fill_color=C_ORANGE,
        )
        core_rect.set_stroke(opacity=0.7)
        # Make dashed
        core_rect.set_stroke(width=2)
        core_rect.move_to(RIGHT * x_mul + UP * y_main)

        core_label = Text("Core: FFT Convolution  O(n log n)",
                          font_size=14, color=C_ORANGE, font="Consolas",
                          weight=BOLD)
        core_label.next_to(core_rect, UP, buff=0.05)

        # ── ARROWS (main flow) ──
        arrows = VGroup()

        # Input → QKV
        arrows.add(make_arrow(b_input, b_qkvg, C_DIM))

        # QKV → split (fan out)
        for b, c in [(b_q, C_CYAN), (b_k, C_GREEN), (b_v, C_PURPLE), (b_gate, C_PINK)]:
            arrows.add(Arrow(
                b_qkvg.get_right(), b.get_left(),
                buff=0.08, color=c, stroke_width=2, max_tip_length_to_length_ratio=0.12,
            ))

        # Q → φ(Q), K → φ(K)
        arrows.add(make_arrow(b_q, b_fq, C_CYAN))
        arrows.add(make_arrow(b_k, b_fk, C_GREEN))

        # φ(K) → deposit, V → deposit
        arrows.add(Arrow(b_fk.get_right(), b_dep.get_left() + UP * 0.1,
                         buff=0.08, color=C_GREEN, stroke_width=2,
                         max_tip_length_to_length_ratio=0.12))
        arrows.add(Arrow(b_v.get_right(), b_dep.get_left() + DOWN * 0.1,
                         buff=0.08, color=C_PURPLE, stroke_width=2,
                         max_tip_length_to_length_ratio=0.12))

        # Deposit → Scatter
        arrows.add(Arrow(b_dep.get_right(), b_scat.get_left(),
                         buff=0.08, color=C_DIM, stroke_width=2,
                         max_tip_length_to_length_ratio=0.12))

        # Scatter → FFT → ⊛ → IFFT → Coupling
        arrows.add(make_arrow(b_scat, b_fft, C_ORANGE))
        arrows.add(make_arrow(b_fft, b_mul, C_ORANGE))
        arrows.add(make_arrow(b_mul, b_ifft, C_ORANGE))
        arrows.add(make_arrow(b_ifft, b_coup, C_DIM))

        # Kernel → ⊛ (from above)
        arrows.add(Arrow(b_kernel.get_bottom(), b_mul.get_top(),
                         buff=0.08, color=C_ORANGE, stroke_width=2,
                         max_tip_length_to_length_ratio=0.15))

        # SpectralGate → ⊛ (from below)
        arrows.add(Arrow(b_sg.get_top(), b_mul.get_bottom(),
                         buff=0.08, color=C_PINK, stroke_width=2,
                         max_tip_length_to_length_ratio=0.15))

        # Coupling → Gather → Q-read
        arrows.add(make_arrow(b_coup, b_gath, C_TEAL))
        arrows.add(make_arrow(b_gath, b_read, C_DIM))

        # φ(Q) → Q-read (long curved arrow)
        q_to_read = CurvedArrow(
            b_fq.get_right() + RIGHT * 0.1,
            b_read.get_top() + UP * 0.05,
            color=C_CYAN, stroke_width=1.8, angle=-PI/6,
            tip_length=0.12,
        )

        # Q[0] → SpectralGate (curved arrow)
        q_to_sg = CurvedArrow(
            b_fq.get_bottom() + DOWN * 0.05,
            b_sg.get_left() + LEFT * 0.05,
            color=C_PINK, stroke_width=1.5, angle=PI/4,
            tip_length=0.1,
        )
        q0_label = Text("q[:,:,0,:]", font_size=11, color=C_PINK,
                        font="Consolas")
        q0_label.next_to(q_to_sg, LEFT, buff=0.05)

        # Gate → (goes to output, curved from bottom)
        # We'll skip the output box for now since it's beyond the attention

        # ── Complexity annotation ──
        complexity = VGroup(
            Text("Complexity breakdown:", font_size=14, color=C_ORANGE,
                 font="Consolas", weight=BOLD),
            Text("FFT/IFFT: O(n log n)  ·  Scatter/Gather: O(n)  ·  SpectralGate: O(1)",
                 font_size=12, color=C_DIM, font="Consolas"),
        ).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
        complexity.to_edge(DOWN, buff=0.2).to_edge(RIGHT, buff=0.3)

        # ── Add everything ──
        self.add(
            core_rect, core_label,
            b_input, b_qkvg,
            b_q, b_k, b_v, b_gate,
            b_fq, b_fk, lbl_hedge,
            b_dep, b_scat, stride_lbl,
            b_fft, b_mul, b_ifft,
            b_kernel, params_txt,
            b_sg, sg_note,
            b_coup, b_gath, b_read,
            arrows, q_to_read, q_to_sg, q0_label,
            complexity,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Scene 2: Full Model Stack (vertical)
# ═══════════════════════════════════════════════════════════════════════════
class WaveFieldModel(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text(
            "Wave Field Transformer — Full Model",
            font_size=28, color=C_TEXT, font="Consolas", weight=BOLD,
        ).to_edge(UP, buff=0.3)
        subtitle = Text(
            "V4.3.9 SPECTRE-Wave  ·  12 layers  ·  8-12 heads  ·  weight-tied output",
            font_size=14, color=C_DIM, font="Consolas",
        ).next_to(title, DOWN, buff=0.1)

        # Stack from bottom to top
        bw, bh = 3.5, 0.55
        gap = 0.15

        b_tokens = make_box("Input Tokens (B, N)", width=bw, height=bh,
                            fill=F_INPUT, border=C_BLUE, text_color=C_BLUE)
        b_embed = make_box("Token Embedding + Sinusoidal PE", width=bw, height=bh,
                           fill=F_PROJ)
        b_drop = make_box("Dropout", width=2.0, height=0.4, fill=F_NORM)

        # Transformer block
        b_ln1 = make_box("LayerNorm₁", width=bw, height=0.45, fill=F_NORM)
        b_wfa = make_box("★ Wave Field Attention", "O(n log n) FFT convolution",
                         width=bw, height=0.65, fill=F_FFT,
                         border=C_ORANGE, text_color=C_ORANGE)
        b_ln2 = make_box("LayerNorm₂", width=bw, height=0.45, fill=F_NORM)
        b_ffn = make_box("Feed-Forward Network", "Linear → GELU → Linear (4× expand)",
                         width=bw, height=0.55, fill=F_PROJ)

        b_interf = make_box("Field Interference Module", "every 3rd layer",
                            width=bw, height=0.5, fill=F_GATE,
                            border=C_PURPLE, text_color=C_PURPLE)
        b_gc = make_box("Global Context (Causal CumMean)", "O(n)",
                        width=bw, height=0.5, fill=F_FIELD,
                        border=C_TEAL, text_color=C_TEAL)

        b_lnf = make_box("Final LayerNorm", width=2.5, height=0.45, fill=F_NORM)
        b_out = make_box("Output Head (weight-tied)", "(B, N, Vocab)",
                         width=bw, height=bh, fill=F_INPUT,
                         border=C_BLUE, text_color=C_BLUE)

        # Arrange vertically
        stack = VGroup(
            b_tokens, b_embed, b_drop,
            b_ln1, b_wfa, b_ln2, b_ffn,
            b_interf, b_gc,
            b_lnf, b_out,
        ).arrange(UP, buff=gap)
        stack.next_to(subtitle, DOWN, buff=0.3)

        # Transformer block bracket
        block_rect = RoundedRectangle(
            corner_radius=0.15, width=4.2,
            height=b_gc.get_top()[1] - b_ln1.get_bottom()[1] + 0.3,
            stroke_color=C_BLUE, stroke_width=1.5,
            fill_opacity=0, stroke_opacity=0.5,
        )
        block_rect.move_to((b_ln1.get_center() + b_gc.get_center()) / 2)

        x12_label = Text("×12", font_size=22, color=C_BLUE, font="Consolas",
                         weight=BOLD)
        x12_label.next_to(block_rect, RIGHT, buff=0.15)

        # Residual arrows
        res1 = CurvedArrow(
            b_ln1.get_left() + LEFT * 0.1 + DOWN * 0.1,
            b_wfa.get_left() + LEFT * 0.1 + UP * 0.1,
            color=C_GREEN, stroke_width=1.5, angle=-PI/3, tip_length=0.1,
        )
        res1_lbl = Text("+", font_size=20, color=C_GREEN, font="Consolas")
        res1_lbl.next_to(res1, LEFT, buff=0.05)

        res2 = CurvedArrow(
            b_ln2.get_left() + LEFT * 0.1 + DOWN * 0.1,
            b_ffn.get_left() + LEFT * 0.1 + UP * 0.1,
            color=C_GREEN, stroke_width=1.5, angle=-PI/3, tip_length=0.1,
        )
        res2_lbl = Text("+", font_size=20, color=C_GREEN, font="Consolas")
        res2_lbl.next_to(res2, LEFT, buff=0.05)

        # Arrows between boxes
        box_arrows = VGroup()
        boxes = [b_tokens, b_embed, b_drop, b_ln1, b_wfa, b_ln2, b_ffn,
                 b_interf, b_gc, b_lnf, b_out]
        for i in range(len(boxes) - 1):
            box_arrows.add(Arrow(
                boxes[i].get_top(), boxes[i+1].get_bottom(),
                buff=0.05, color=C_DIM, stroke_width=1.5,
                max_tip_length_to_length_ratio=0.2,
            ))

        self.add(title, subtitle, stack, block_rect, x12_label,
                 box_arrows, res1, res1_lbl, res2, res2_lbl)


# ═══════════════════════════════════════════════════════════════════════════
#  Scene 3: Animated Pipeline (shows data flowing through)
# ═══════════════════════════════════════════════════════════════════════════
class WaveFieldAnimated(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text(
            "Wave Field Attention — Data Flow Animation",
            font_size=26, color=C_ORANGE, font="Consolas", weight=BOLD,
        ).to_edge(UP, buff=0.3)
        self.play(Write(title))

        # Build simplified pipeline
        boxes_data = [
            ("Input x", F_INPUT, C_BLUE),
            ("QKV+Gate", F_PROJ, C_DIM),
            ("Feature\nMaps", F_FEAT, C_CYAN),
            ("Deposit\nφ(K)⊙V", F_FEAT, C_GREEN),
            ("Scatter", F_FIELD, C_TEAL),
            ("FFT", F_FFT, C_ORANGE),
            ("⊛ Kernel", F_FFT, C_ORANGE),
            ("IFFT", F_FFT, C_ORANGE),
            ("Gather", F_FIELD, C_TEAL),
            ("φ(Q)⊙\nRead", F_FEAT, C_CYAN),
            ("Gate⊙\nOutput", F_GATE, C_PINK),
        ]

        boxes = VGroup()
        for label, fill, border in boxes_data:
            b = make_box(label, width=1.1, height=0.7, fill=fill,
                         border=border, text_color=border, font_size=15)
            boxes.add(b)

        boxes.arrange(RIGHT, buff=0.15)
        boxes.next_to(title, DOWN, buff=0.6)
        boxes.scale(0.95)

        # Arrows
        arrows = VGroup()
        for i in range(len(boxes) - 1):
            arrows.add(Arrow(
                boxes[i].get_right(), boxes[i+1].get_left(),
                buff=0.05, color=C_DIM, stroke_width=2,
                max_tip_length_to_length_ratio=0.2,
            ))

        # Complexity labels
        labels = VGroup()
        complexities = [
            "", "", "O(n)", "O(n)", "O(n)",
            "O(n log n)", "O(n log n)", "O(n log n)",
            "O(n)", "O(n)", "O(n)",
        ]
        for i, c in enumerate(complexities):
            if c:
                lbl = Text(c, font_size=12, color=C_ORANGE if "log" in c else C_DIM,
                           font="Consolas")
                lbl.next_to(boxes[i], DOWN, buff=0.15)
                labels.add(lbl)

        # Animate: boxes appear one by one with arrows
        for i in range(len(boxes)):
            self.play(FadeIn(boxes[i], shift=RIGHT * 0.3), run_time=0.3)
            if i < len(arrows):
                self.play(Create(arrows[i]), run_time=0.15)

        self.play(FadeIn(labels))

        # Highlight core (FFT section)
        highlight = SurroundingRectangle(
            VGroup(boxes[5], boxes[6], boxes[7]),
            color=C_ORANGE, buff=0.15, stroke_width=2.5,
        )
        hl_label = Text("Core: O(n log n)", font_size=18, color=C_ORANGE,
                        font="Consolas", weight=BOLD)
        hl_label.next_to(highlight, DOWN, buff=0.2)

        self.play(Create(highlight), Write(hl_label))
        self.wait(2)
