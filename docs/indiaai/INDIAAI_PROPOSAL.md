# IndiaAI Compute Portal — Project Proposal

## Wave Field LLM: Physics-Based O(n log n) Language Model

**Applicant:** Pankh AI (DPIIT Registered Startup)
**Project Lead:** Badaramoni Avinash
**Date:** February 2026
**License:** AGPL-3.0 (fully open-source, network-use disclosure required)

---

## 1. What Is This?

Wave Field LLM is a **new type of language model architecture** built from scratch in India.

Instead of using standard self-attention (which costs O(n²) and chokes on long text), we use **damped wave equation physics** — the same math that describes how sound travels through air — to route information between tokens. This gives us **O(n log n) complexity**, meaning the model gets proportionally faster as text gets longer.

**This is not a fine-tune. Not a wrapper. Not a LoRA.** It's a ground-up architecture with a new attention mechanism, written in pure PyTorch with zero framework dependencies.

---

## 2. What We've Already Built (Proof It Works)

### Results at 5 Million Tokens (WikiText-2)

| Metric | Wave Field LLM | Standard Transformer | Advantage |
|--------|---------------|---------------------|-----------|
| Perplexity (lower = better) | **117.6** | 457.2 | **3.9x better** |
| Next-token accuracy | **28.7%** | 11.8% | **2.4x better** |
| Parameters | 8.58M | 6.92M | +24% overhead |

### Speed Advantage at Long Sequences

| Sequence Length | Standard | Wave Field | Who Wins |
|-----------------|----------|-----------|----------|
| 512 tokens | 102.8K tok/s | 29.7K tok/s | Standard 3.5x faster |
| 1,024 tokens | 101.9K tok/s | 70.2K tok/s | Standard 1.45x faster |
| **2,048 tokens** | 49.9K tok/s | 101.3K tok/s | **Wave 2x faster** |
| **4,096 tokens** | 30.9K tok/s | 136.4K tok/s | **Wave 4.4x faster** |
| **8,192 tokens** | 1.4K tok/s | 151.6K tok/s | **Wave 108x faster** |

At 8K context length, our architecture is **108 times faster** than standard attention.

### Causality Verified

We've mathematically and empirically verified that the model cannot cheat by looking at future tokens — a non-trivial requirement for FFT-based attention. Test suite included in the repository.

### What's Running Today

- 21.79M parameter model trained on WikiText-103 (20M tokens)
- Perplexity: 274.65 (causally verified)
- Fully containerized with Docker for reproducibility
- All benchmarks, tests, and diagnostics included

---

## 3. The Core Innovation (Why This Matters)

### Standard Transformers (GPT, LLaMA, etc.)

Every token attends to every other token → O(n²) cost → context windows are expensive to scale → inference costs explode with long documents.

### Wave Field LLM

Tokens deposit information onto a continuous wave field → physics-based kernel propagates information via damped oscillations → tokens read from the field → **O(n log n) total cost**.

The kernel is not arbitrary — it's grounded in wave physics with 3 learnable parameters per head:
- **Frequency** (how fast information oscillates)
- **Damping** (how far information travels)
- **Phase** (head diversity)

Plus a **content-adaptive spectral gate** (SPECTRE) that modulates the kernel in the frequency domain based on input content, making each forward pass input-dependent while keeping O(n log n) complexity.

### Key Technical Components (All From Published Research)

| Component | Source | What It Does |
|-----------|--------|-------------|
| Learned Feature Maps | Hedgehog, ICLR 2024 | Replaces softmax with learnable nonlinearity |
| HiPPO Initialization | S4D, NeurIPS 2022 | Optimal basis for temporal memory |
| Spectral Gate | SPECTRE, arXiv 2025 | Input-dependent kernel modulation |
| Wave Physics Kernel | Original contribution | Damped oscillation dynamics |
| Cross-head Interference | Original contribution | Physics-based signal routing |

---

## 4. What We'll Do With GPU Compute

### Phase 1: Prove Scaling (Months 1-2)

**Goal:** Show that Wave Field LLM's quality advantage holds as we scale up.

| Parameter | Value |
|-----------|-------|
| Model size | 55M parameters |
| Training data | 100M tokens (WikiText-103) |
| GPU requirement | 1x A100, ~8 hours |
| Success metric | Maintain ≥3x perplexity advantage over Standard Transformer |

**Deliverable:** Published benchmark comparing 55M Wave Field vs 55M Standard Transformer on identical data.

### Phase 2: Match Subquadratic SOTA (Months 2-4)

**Goal:** Match or beat Mamba-200M and RWKV-200M — the current best subquadratic models.

| Parameter | Value |
|-----------|-------|
| Model size | 200M parameters |
| Training data | 1B tokens (SlimPajama / FineWeb) |
| GPU requirement | 8x A100, ~24 hours |
| Success metric | Match Mamba-200M perplexity on standard benchmarks |

**Deliverable:** Head-to-head benchmark against Mamba, RWKV, and RetNet on LM Eval Harness.

### Phase 3: Production Viability (Months 4-6)

**Goal:** Train a model that generates usable text and demonstrates real-world value.

| Parameter | Value |
|-----------|-------|
| Model size | 1B parameters |
| Training data | 10B+ tokens (FineWeb / RedPajama) |
| GPU requirement | 16-32x A100, ~72 hours |
| Success metric | Match LLaMA-1B quality at ≤50% training tokens |

**Deliverable:** Open-source 1B model with inference code, benchmarks, and a technical report.

### Engineering Milestones (Parallel Track)

| Milestone | Impact | Timeline |
|-----------|--------|----------|
| Flash-FFT CUDA kernel | Close 3x throughput gap at short sequences | Month 1-2 |
| muP hyperparameter transfer | Don't waste GPU hours re-tuning at each scale | Month 1 |
| FSDP distributed training | Required for 1B+ models across multiple GPUs | Month 2-3 |
| Sliding window hybrid | Combine local attention with wave global context | Month 2 |

---

## 5. Bill of Materials (GPU Hours Estimate)

### Conservative Estimate

| Phase | GPUs | Hours | Total GPU-Hours | At ₹65/hr | At Market Rate (~₹400/hr) |
|-------|------|-------|-----------------|-----------|--------------------------|
| Phase 1 (55M, 100M tok) | 1x A100 | 8 | 8 | ₹520 | ₹3,200 |
| Phase 2 (200M, 1B tok) | 8x A100 | 24 | 192 | ₹12,480 | ₹76,800 |
| Phase 3 (1B, 10B tok) | 32x A100 | 72 | 2,304 | ₹1,49,760 | ₹9,21,600 |
| Engineering & iteration | 4x A100 | 100 | 400 | ₹26,000 | ₹1,60,000 |
| **Total** | | | **2,904** | **₹1,88,760** | **₹11,61,600** |

**Subsidy request:** 100% (foundational AI model development)
**Note:** Total GPU hours < 5,000, eligible for auto-approval.

### What ₹1.89 Lakh Buys (vs Market)

At cloud market rates, this work would cost **₹11.6 lakh**. The IndiaAI subsidy makes an India-origin foundational model economically viable for a bootstrapped startup.

---

## 6. Why This Qualifies as a Foundational Model

1. **Ground-up architecture** — not a derivative of GPT/LLaMA/Mamba
2. **Novel attention mechanism** — physics-based, not just "linear attention"
3. **Published research foundations** — builds on ICLR, NeurIPS, ICML papers
4. **Open-source under AGPL-3.0** — derivatives must remain open
5. **Scaling path defined** — S1 (21M) → S2 (55M) → S3 (200M) → S4 (1B)
6. **Made in India** — designed, coded, and benchmarked entirely in India

---

## 7. Expected Outcomes & Deliverables

### At End of 6 Months

| Deliverable | Description |
|-------------|-------------|
| Open-source 1B model | Trained Wave Field LLM with weights on HuggingFace |
| Technical report | Scaling analysis, benchmark suite, architecture documentation |
| Flash-FFT kernel | Open-source CUDA kernel for wave field convolution |
| Benchmark suite | Reproducible Docker-based comparisons against Mamba, RWKV, Standard Transformer |
| Inference code | Production-ready inference with KV-cache equivalent for wave fields |

### Long-Term Impact

- **For Indian AI ecosystem:** A sovereign, open-source architecture that doesn't depend on US-origin transformer patents or code
- **For the field:** Proof that physics-grounded models can compete with empirically-designed architectures
- **For long-context applications:** O(n log n) attention enables document-level understanding at fraction of the cost
- **Applications:** Legal document analysis, multilingual processing (Indian languages have long compound words), code generation, scientific literature

---

## 8. Team

**Badaramoni Avinash** — Solo developer/researcher. Designed, implemented, and benchmarked the entire architecture. Background in physics-informed computing.

**Pankh AI** — DPIIT-registered startup focused on AI research and development.

---

## 9. Repository & Verification

- **Code:** Available on GitHub (Pankh-AI organization)
- **License:** AGPL-3.0
- **Reproducibility:** `docker compose build && docker compose run --rm benchmark`
- **Causality test:** `python tests/test_causality.py`
- **All results:** Machine-generated JSON in `results/` directory, not hand-written

---

## 10. Risks & Honest Assessment

We believe in transparency. Here's what could go wrong:

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Quality advantage disappears at scale | Medium | Many subquadratic models (Hyena, RetNet) collapsed at 1B. We'll check at each milestone. |
| Throughput gap doesn't close | Medium | Flash-FFT kernel is engineering, not research. Known solutions exist (FlashFFTConv, ICLR 2024). |
| Causality leak at larger models | Low | Existing test suite + `_enforce_causal_kernel()` fix is robust. Will add fuzz testing. |
| 6-month timeline too aggressive | Medium | Phase 1 and 2 are achievable regardless. Phase 3 may extend to 9 months. |

**Bottom line:** Phase 1 (55M) is near-certain. Phase 2 (200M) is the real test. Phase 3 (1B) depends on Phase 2 results. We'll publish results honestly regardless of outcome.
