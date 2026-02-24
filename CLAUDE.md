# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wave Field LLM is a physics-based language model architecture that replaces O(n²) self-attention with damped wave equation dynamics on continuous fields, achieving O(n log n) complexity. Pure PyTorch, no frameworks.

**License:** AGPL-3.0 (derivatives must be open-sourced, network services must disclose source).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run causality test (verifies FFT doesn't leak future tokens)
python tests/test_causality.py

# Benchmarks
python benchmarks/benchmark_wikitext2.py      # Wave V3.4 vs Standard Transformer on WikiText-2
python benchmarks/train_wave_v35_bpe.py        # V3.5 + BPE tokenizer training
python benchmarks/train_100m_bpe.py            # 100M parameter scaling experiment

# Physics diagnostics
python diagnostics/diagnose_physics.py         # Energy flow, conservation, causality checks
python diagnostics/diagnose_bpe.py             # BPE tokenizer diagnostics

# Module-level smoke tests (each has __main__ block)
python src/wave_field_transformer.py
python src/global_context.py
```

No pytest, no linter, no CI/CD configured. Tests are run as standalone scripts.

## Architecture

### Core: Wave Field Attention (`src/wave_field_attention.py`)

Each attention head is a damped wave with 3 learnable parameters:
- `wave_frequency` — oscillation speed (controls attention pattern periodicity)
- `wave_damping` — decay rate (how far back to attend)
- `wave_phase` — offset (head diversity)

**Pipeline:** Tokens → QKV projection → bilinear scatter onto continuous field → FFT convolution with wave kernel → static cross-head field coupling → content-dependent gating → bilinear gather back to token positions.

Causality is enforced by zeroing the kernel for t < 0 before FFT. Energy conservation was intentionally removed in V3.5 because absolute position mapping leaves most of the field empty, and rescaling crushes information.

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
