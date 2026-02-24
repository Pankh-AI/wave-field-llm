# Bill of Materials — IndiaAI Compute Request

**Project:** Wave Field LLM (Physics-Based Foundational Language Model)
**Applicant:** Pankh AI (DPIIT Registered Startup)
**Date:** February 2026
**Subsidy Category:** Foundational AI Model (100% subsidy requested)

---

## GPU Hardware Requirements

**Preferred GPU:** NVIDIA A100 80GB (or H100 equivalent)
**Reason:** bf16 training with fp32 FFT operations requires ≥40GB VRAM at 200M+ params. A100's bf16 tensor cores + large memory are optimal for our FFT-heavy workload.

**Alternative:** NVIDIA H200 (available on IndiaAI portal) — would reduce training time by ~30%.

---

## Compute Breakdown by Phase

### Phase 1: Scaling Validation (55M parameters)

| Item | Details |
|------|---------|
| **Objective** | Validate quality advantage scales from 8M → 55M params |
| **Model config** | embed=512, 10 layers, 12 heads, FFN=2048 |
| **Dataset** | WikiText-103, 100M tokens, byte-level BPE (8K vocab) |
| **Training** | AdamW, cosine LR schedule, bf16 autocast |
| **GPUs** | 1x A100 80GB |
| **Estimated time** | 8 hours |
| **GPU-hours** | **8** |
| **Includes** | 3 runs: Wave Field + Standard Transformer baseline + ablation |

### Phase 2: Competitive Benchmark (200M parameters)

| Item | Details |
|------|---------|
| **Objective** | Match or beat Mamba-200M, RWKV-200M on standard benchmarks |
| **Model config** | embed=1024, 16 layers, 16 heads, FFN=4096 |
| **Dataset** | SlimPajama or FineWeb, 1B tokens |
| **Training** | AdamW, muP hyperparameter transfer, bf16, FSDP |
| **GPUs** | 8x A100 80GB |
| **Estimated time** | 24 hours |
| **GPU-hours** | **192** |
| **Includes** | 2 runs: Wave Field + Standard Transformer baseline |

### Phase 3: Production Model (1B parameters)

| Item | Details |
|------|---------|
| **Objective** | Train usable model, publish weights, demonstrate real-world viability |
| **Model config** | embed=2048, 24 layers, 16 heads, FFN=8192 |
| **Dataset** | FineWeb / RedPajama, 10B tokens |
| **Training** | AdamW, muP, bf16, FSDP across multi-node |
| **GPUs** | 32x A100 80GB |
| **Estimated time** | 72 hours |
| **GPU-hours** | **2,304** |
| **Includes** | 1 full training run + checkpointing every 500M tokens |

### Engineering & Iteration Buffer

| Item | Details |
|------|---------|
| **Objective** | Flash-FFT kernel development, hyperparameter sweeps, debugging |
| **Activities** | CUDA kernel profiling, muP grid search, causality testing at scale |
| **GPUs** | 4x A100 80GB |
| **Estimated time** | 100 hours |
| **GPU-hours** | **400** |

---

## Total Compute Summary

| Phase | GPU-Hours | % of Total |
|-------|-----------|-----------|
| Phase 1: 55M scaling | 8 | 0.3% |
| Phase 2: 200M competitive | 192 | 6.6% |
| Phase 3: 1B production | 2,304 | 79.3% |
| Engineering buffer | 400 | 13.8% |
| **Total** | **2,904** | **100%** |

---

## Cost Estimate

| Rate | Cost |
|------|------|
| IndiaAI subsidized rate (₹65/GPU-hr) | ₹1,88,760 |
| AWS market rate (~₹330/GPU-hr) | ₹9,58,320 |
| Azure market rate (~₹590/GPU-hr) | ₹17,13,360 |
| **Subsidy requested (100%)** | **₹1,88,760** |

---

## Timeline

```
Month 1  ████████░░░░  Phase 1 (55M) + Flash-FFT kernel dev
Month 2  ████████████  Phase 1 results → Phase 2 start + muP integration
Month 3  ████████████  Phase 2 training (200M, 1B tokens)
Month 4  ████████░░░░  Phase 2 evaluation + FSDP setup for Phase 3
Month 5  ████████████  Phase 3 training (1B, 10B tokens)
Month 6  ████████░░░░  Phase 3 evaluation + report + model release
```

---

## Software Stack

| Component | Version | License |
|-----------|---------|---------|
| PyTorch | ≥2.0.0 | BSD |
| Python | 3.10+ | PSF |
| CUDA | 12.x | NVIDIA EULA |
| HuggingFace datasets | ≥2.14.0 | Apache 2.0 |
| HuggingFace tokenizers | ≥0.15.0 | Apache 2.0 |
| Docker | Latest | Apache 2.0 |
| Wave Field LLM (our code) | V4.3+ | AGPL-3.0 |

**No proprietary dependencies.** Entire stack is open-source or freely available.

---

## Outputs & Deliverables

| Deliverable | Format | Where Published |
|-------------|--------|----------------|
| Trained model weights (1B) | PyTorch .pt / SafeTensors | HuggingFace Hub |
| Training code | Python/PyTorch | GitHub (AGPL-3.0) |
| Flash-FFT kernel | CUDA C++ | GitHub (AGPL-3.0) |
| Benchmark results | JSON + Markdown | GitHub + technical report |
| Technical report | PDF | arXiv |
| Inference code | Python | GitHub (AGPL-3.0) |

---

## Notes

1. **Auto-approval eligible:** Total request (2,904 GPU-hrs) is below the 5,000-hour threshold
2. **No data storage needed:** Training data (WikiText, SlimPajama, FineWeb) is publicly available and downloaded at runtime
3. **Checkpointing:** All phases checkpoint every 500M tokens — no compute is wasted on crashes
4. **Reproducibility:** Docker Compose configuration included — any reviewer can reproduce results
5. **Monthly reporting:** Willing to submit progress reports with benchmark results at each milestone
