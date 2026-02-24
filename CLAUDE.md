# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wave Field LLM is a physics-based language model architecture that replaces O(n²) self-attention with damped wave equation dynamics on continuous fields, achieving O(n log n) complexity. Pure PyTorch, no frameworks.

**License:** AGPL-3.0 (derivatives must be open-sourced, network services must disclose source).

## Commands

```bash
# Install dependencies (local dev only — benchmarks run in Docker)
pip install -r requirements.txt

# ---- Docker (standard way to run benchmarks) ----
# Build once:
docker compose build

# Run latest benchmark (V4.3 by default):
docker compose run --rm benchmark

# Run a specific benchmark:
docker compose run --rm benchmark python benchmarks/benchmark_v43.py
docker compose run --rm benchmark python benchmarks/benchmark_v42.py

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
- `src/causal_field_attention.py`, `src/causal_field_transformer.py` — V1/V2 historical implementations, superseded by V3.5.

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

## Debugging Approach

Bugs in this codebase are diagnosed through physics quantities, not profilers:
- Suspiciously low training PPL → check for future token leakage via `tests/test_causality.py`
- Collapsed outputs → inspect energy flow with `diagnostics/diagnose_physics.py`
- The version history (V3.0→V3.5) documents each physics-diagnosed bug fix
