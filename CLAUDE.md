# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wave Field LLM is a physics-based language model architecture that replaces O(n²) self-attention with damped wave equation dynamics on continuous fields, achieving O(n log n) complexity. Pure PyTorch, no frameworks.

**License:** AGPL-3.0 (derivatives must be open-sourced, network services must disclose source).

## Repository Structure

```
src/                        # Core architecture (V4.3 SPECTRE-Wave)
  wave_field_attention.py   #   Attention mechanism (kernels, feature maps, spectral gate)
  wave_field_transformer.py #   Transformer layers, model, optimizer config
  global_context.py         #   O(n) global context via causal cumulative mean
  legacy/                   #   V1/V2 implementations (superseded)

benchmarks/                 # Active benchmarks (run via Docker)
  benchmark_scaling.py      #   S1-S4 scaling runs (primary)
  benchmark_v43.py          #   V4.3 5M-token comparison
  benchmark_v42.py          #   V4.2 ablation study
  benchmark_v43_upgrade.py  #   V4.3 component ablation (S4D, Hedgehog, local window)
  benchmark_v44_upgrade.py  #   V4.4 experiments (write gate, 3D interference)
  benchmark_lr_sweep.py     #   Learning rate sweep
  benchmark_long_context.py #   Long context speed/memory benchmark
  benchmark_kernel_mixture.py # Kernel mixture experiments
  legacy/                   #   Old/one-off benchmarks

diagnostics/                # Training observability
  training_monitor.py       #   WaveFieldMonitor — hooks into all internals
  visualize_monitor.py      #   12-panel dashboard from monitor JSON
  diagnose_physics.py       #   Energy flow / field state diagnostics
  diagnose_bpe.py           #   Tokenizer diagnostics

scripts/                    # Visualization and utilities
  visualize_all_benchmarks.py # Master dashboard from all result JSONs
  plot_long_context.py      #   Long context plots
  visualize_v432.py         #   V4.3.2 analysis plots
  visualize_v432_scaling.py #   V4.3.2 scaling analysis
  vram_calc.py              #   VRAM estimation calculator
  data_loader.py            #   Data loading utilities
  pretokenize.py            #   Pre-tokenization script

tests/                      # Standalone test scripts
  test_causality.py         #   FFT causality verification (primary)
  causality_probe.py        #   Detailed causality probing
  causality_ablation.py     #   Causality ablation study
  future_shuffle_eval.py    #   Future token shuffle evaluation

docs/                       # Documentation
  ARCHITECTURE.md           #   Full architecture doc
  ARCHITECTURE_V43.md       #   V4.3 specifics
  RESEARCH_V4.md            #   V4 research notes
  WHAT_WORKS.md             #   What worked / what didn't
  BENCHMARK_RESULTS.md      #   Historical benchmark results
  research_gpt.md           #   GPT analysis of architecture gaps
  indiaai/                  #   IndiaAI compute portal application docs
  papers/                   #   Reference papers (PDFs)
  plans/                    #   Design plans

results/                    # Generated output (gitignored, Docker volume mount)
  figures/                  #   README/publication figures (fig_*.png, v435_*.png/gif)
  announce/                 #   Shareable announcement graphics
  data/                     #   Benchmark JSON results (scaling_s1.json, etc.)
  checkpoints/              #   Model checkpoints (.pt files)
  plots/                    #   Auto-generated benchmark plots
  monitor/                  #   Training monitor snapshots
  legacy/                   #   Old version plots/scripts (V4.3.2, V5, etc.)
  cache/                    #   HF dataset/tokenizer cache

tokenizers/                 # Custom tokenizer implementations
```

## Commands

```bash
# Install dependencies (local dev only — benchmarks run in Docker)
pip install -r requirements.txt

# ---- Docker (standard way to run benchmarks) ----
# Build once:
docker compose build

# Run scaling benchmark (S1 by default):
docker compose run --rm s1

# Run specific benchmark:
docker compose run --rm v43 python benchmarks/benchmark_v43.py

# Run training monitor on GPU:
docker compose run --rm v43 python diagnostics/training_monitor.py

# Available Docker services: s1, s2, s3, v43

# Results auto-saved to ./results/ via volume mount

# ---- Local (no GPU required) ----
# Causality test (verifies FFT doesn't leak future tokens)
python tests/test_causality.py

# Module-level smoke tests (each has __main__ block)
python src/wave_field_transformer.py
python src/global_context.py

# Physics diagnostics
python diagnostics/diagnose_physics.py
python diagnostics/diagnose_bpe.py

# Generate visualization dashboards
python scripts/visualize_all_benchmarks.py
```

No pytest, no linter, no CI/CD configured. Tests are run as standalone scripts.

## Architecture

### Core: Wave Field Attention (`src/wave_field_attention.py`)

V4.3 SPECTRE-Wave architecture. Each attention head is a damped wave with 3 learnable parameters:
- `wave_frequency` — oscillation speed (HiPPO harmonic init: `ω_n = π(2n+1)/2`)
- `wave_damping` — decay rate (uniform init: `softplus(-0.69) ≈ 0.5`)
- `wave_phase` — offset (head diversity)

Key modules:
- `LearnedFeatureMap` — identity-init Linear + ReLU (Hedgehog, ICLR 2024). Replaces `elu(x)+1`.
- `SpectralGate` — MLP conditioned on `mean(Q)` that modulates kernel FFT per-sample (SPECTRE, arXiv:2502.18394). Makes attention input-dependent while staying O(n log n).

**Pipeline:** Tokens → QKV projection → learned feature maps φ(K), φ(Q) → K-weighted deposit → bilinear scatter onto field → content-adaptive spectral modulation of kernel FFT → FFT convolution → cross-head coupling → bilinear gather → Q-weighted read → gating → output.

Causality is enforced by zeroing the kernel for t < 0 before FFT.

### Transformer (`src/wave_field_transformer.py`)

- `WaveFieldTransformerLayer`: pre-norm → wave field attention → residual → pre-norm → FFN (GELU) → residual
- `FieldInterferenceModule`: physics-based signal routing via constructive/destructive interference, applied every `interference_interval` layers (default 3)
- `WaveFieldTransformer`: full model with sinusoidal PE and weight-tied output projection

### Supporting Modules

- `src/global_context.py` — O(n) global context via causal cumulative mean pooling. Extends receptive field beyond wave kernel locality (~18 positions).
- `src/legacy/` — V1/V2 historical implementations (causal_field_attention, causal_field_transformer), superseded by V3.5.

### Tokenizers (`tokenizers/`)

Three tokenizer strategies, not imported by `src/__init__.py`:
- `field_tokenizer_v2.py` — words-first with character fallback, zero UNK
- `field_tokenizer_v3.py` — morphological subwords + BPE support
- `field_aware_tokenizer.py` — co-occurrence-based field-aware vocab

Benchmarks use HuggingFace `tokenizers` library for byte-level BPE (8K vocab).

## Key Design Decisions

- **Absolute position mapping** (`field_pos = token_i × stride`): tokens map to fixed field positions regardless of sequence length. This prevents position shifting during generation but means most field cells are empty for short sequences.
- **No energy conservation** (V3.5): removed because sparse occupancy + conservation rescaling → signal collapse. Residual connections + LayerNorm handle normalization.
- **Gate bias = 2.0**: content-dependent gates initialized open so gradients flow freely at start of training.
- **Static field coupling**: learned cross-head interaction matrix applied after wave convolution, before gating.
- **fp32 FFT operations**: All FFT/IFFT in `_wave_convolve` and `_build_wave_kernels` are forced to fp32, then cast back to input dtype. bf16 twiddle factors lose precision through butterfly stages (Hyena/H3 best practice).
- **Param count note**: At "100M config" (embed=768, 12 layers, 12 heads), Wave Field = ~139M params vs Standard = ~110M. The +29M overhead comes from learned feature maps, spectral gate, and interference modules. Accepted as architecture cost.

## Dtype/AMP Guidelines

- **A100/H100**: Use `bf16` autocast, **no** GradScaler (bf16 has same exponent range as fp32).
- **T4/V100**: Use `fp16` autocast **with** GradScaler (fp16 has narrow exponent range).
- **CPU**: Use `fp32` fallback, no autocast.
- FFT operations are always fp32 regardless of autocast dtype (handled internally).

## Debugging Approach

Bugs in this codebase are diagnosed through physics quantities, not profilers:
- Suspiciously low training PPL → check for future token leakage via `tests/test_causality.py`
- Collapsed outputs → inspect energy flow with `diagnostics/diagnose_physics.py`
- Training dynamics → run `diagnostics/training_monitor.py` for kernel/FM/gate/gradient snapshots
- The version history (V3.0→V3.5) documents each physics-diagnosed bug fix
