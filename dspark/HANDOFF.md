# DSpark vLLM Integration — Handoff Document

> **Date:** 2026-06-27
> **Session:** playground → vLLM fork
> **Branch:** `dspark-research` (pushed to `origin`)
> **Target Hardware:** 2× NVIDIA DGX Spark (GB10 / SM121, 128 GB unified each)
> **Next Session:** Pull this branch, continue research & implementation for GB10

## Quick Start (on DGX Spark Cluster)

```bash
# On each DGX Spark node:
cd ~/Documents/repos
git clone git@github.com:ishan5ain/vllm.git
cd vllm
git checkout dspark-research
cat dspark/README.md     # Orientation
cat dspark/HANDOFF.md    # This document
```

## What This Is

Complete DSpark integration research, tailored for implementation on a 2× DGX Spark
(GB10 / SM121) cluster. Any session or agent picking this up should be able to
understand the full picture and continue development.

---

## 0. DGX Spark Hardware Context (GB10 / SM121)

### Specs per node
- **GPU:** Blackwell GB10, SM121, 128 GB LPDDR5X unified memory (CPU+GPU shared)
- **Networking:** ConnectX-7 200GbE, QSFP56 direct cable for NCCL/RoCE
- **No NVLink** between nodes — all tensor-parallel communication goes over RoCE

### Known working configurations (community-validated)

**Single node (2-bit hybrid quant):**
```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --quantization deepseek_v4_hybrid_iq2 \
  --tensor-parallel-size 1 \
  --enforce-eager  # MTP/decode may need eager mode on SM121
```
Ref: `Entrpi/ds4-spark-vllm` — validated end-to-end, coherent multi-token output.
Important env: `VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0` (required on SM121).

**Dual node (official FP8, TP=2):**
```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --nnodes 2
```
Ref: `tonyd2wild/deepseek-v4-flash-dgx-spark` — TP=2, FP8 KV, 200K ctx, MTP.
Requires: `jasl/vllm` fork or `eugr/spark-vllm-docker` (PR #219).

**Docker image (recommended):** `lmxxf/vllm-deepseek-v4-dgx-spark:latest`

### GB10-specific DSpark concerns

- **Memory budget:** 128 GB per node. The base V4-Flash at 2-bit quant fits at ~100 GB.
  The DSpark draft model adds ~8 GB (3 MTP layers + 256 experts). On a single node
  with 2-bit quant, this may not fit — verify with `nvidia-smi` before loading.
- **Dual node:** With TP=2 using official FP8, each node has ~50-60 GB base model.
  Room for draft model exists but needs measurement.
- **MTP performance:** Per `Entrpi/ds4-on-spark`, MTP on single-stream decode was a
  "net throughput loss" — DSpark may show similar behavior on single-stream.
  Benchmark before committing to production deployment.
- **SM121 Triton quirk:** `VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0` required.
- **Community forks:** If vLLM main doesn't work on SM121, try `jasl/vllm` or the
  `tonyd2wild/deepseek-v4-flash-dgx-spark` recipe.

---

## 1. The Big Picture

**DSpark** is DeepSeek's production speculative decoding framework. It accelerates
LLM inference by:

1. **Better drafting** — A semi-autoregressive head (Markov W1W2, rank 256) adds
   inter-token dependencies to parallel draft generation, reducing suffix decay.
2. **Smarter verification** — A confidence head predicts per-position acceptance
   probabilities, enabling dynamic verification length trimming.
3. **Load-aware scheduling** — A hardware-aware prefix scheduler (Algorithm 1 in
   paper) optimizes verification budgets against real-time system load.

**The target model:** `deepseek-ai/DeepSeek-V4-Flash-DSpark` (284B params, 13B activated, MoE)

**The implementation target:** vLLM, in the repo at `~/Documents/repos/vllm`

---

## 2. What We've Gathered

### Sources Consulted

| Source | What It Gave Us |
|---|---|
| DSpark paper (OCR'd) | Full algorithmic description, equations, production config |
| Checkpoint `config.json` | Exact hyperparams: gamma=5, rank=256, layers=[40,41,42], mask=128799 |
| Checkpoint `model.safetensors.index.json` | Full weight structure with shapes |
| DeepSpec repo (`deepseek-ai/DeepSpec`) | PyTorch reference implementation of Markov head, confidence head, draft ops |
| vLLM `vllm/models/deepseek_v4/nvidia/mtp.py` | Existing MTP code to extend |
| vLLM PR #40860, #41404, #45953 | Pattern for dynamic SD with CUDA graph safety |
| Slack thread (Kaichao You et al.) | Practical scoping: 4-tier scheduling, start simple |
| GB10 community recipes | Dual-spark TP=2, SM121 quirks, memory budget, MTP performance |

### Key Architecture Facts

```
DSpark Draft Model (loaded from checkpoint):

  Input: [anchor_token] [mask] [mask] [mask] [mask]  (gamma=5)
  
  mtp.0: DeepseekV4DecoderLayer (attn + MoE FFN, bidirectional)
  mtp.1: DeepseekV4DecoderLayer (attn + MoE FFN, bidirectional)  
  mtp.2: DeepseekV4DecoderLayer + hc_head + shared_head  (output layer)
  
  + markov_head: W1[129280x256], W2[256x129280]  (on mtp.2)
  + confidence_head: Linear[4352->1]  (on mtp.2)
```

**Target context:** Hidden states from V4 layers 40, 41, 42 concatenated and projected
into draft hidden space via `fc: Linear[12288->4096]`.

**Inference flow (one cycle):**
1. Target forward -> extract hidden states from layers 40,41,42
2. Project context -> draft_ctx = Norm(fc(concat(h40,h41,h42)))
3. Draft backbone forward (bidirectional, is_causal=False) -> hidden [B, gamma, 4096]
4. hc_head + shared_head -> base_logits [B, gamma, 129280]
5. Markov sequential loop (k=1..5): bias = W2(W1[prev]), logits = base + bias, sample
6. Confidence: per-position c_k = sigma(w.[h_k; W1[prev]])
7. Verification: target model checks prefix, rejection sampling

---

## 3. What We've Built (on the branch)

```
dspark/
├── README.md                    Index + config metadata + design decisions
├── IMPLEMENTATION_PLAN.md       5-phase roadmap with architecture diagrams
├── ALGORITHM_REFERENCE.md       All paper equations, pseudocode, production adaptations
├── checkpoint_anatomy.md        Full weight structure with shapes and per-layer breakdown
├── dspark_model_skeleton.py     Python stubs: MarkovHead, ConfidenceHead, DSparkModel
├── sts_calibration.py           673-line STS calibration implementation (8 functions)
├── sts_calibration.md           Companion docs for STS
└── DSpark_paper_full.md         OCR'd full paper (887 lines)
```

### STS Calibration (`sts_calibration.py`)

Key functions:
- `calibrate_sts(logits, labels, gamma=5)` — Grid search T_k in [0.1, 5.0], 50 steps, minimizes ECE
- `apply_temperatures(logits, temps)` — Apply calibrated sigma(c/T) during inference
- `compute_cumulative_survival(logits, temps)` — Product sigma(./T) for prefix scheduler
- `compute_ece(probs, labels)` — ECE with equal-frequency binning
- `compute_default_temperatures()` — Identity fallback [1.0]x5

---

## 4. Design Decisions (Settled)

| Decision | Choice | Rationale |
|---|---|---|
| Integration class | New `DeepSeekV4DSpark` (parallel to MTP) | Avoids breaking existing MTP; cleaner separation |
| Layer count override | Add `dspark_num_backbone_layers: 3` config field | Cleaner than repurposing `num_nextn_predict_layers` |
| Attention mode | Bidirectional within block (`is_causal=False`) | Required by DSpark architecture |
| Scheduling phasing | Tier 1 -> Tier 2 -> Tier 3 -> Tier 4 | Slack discussion; start with fixed gamma=5 |
| STS calibration | Offline grid search; fallback to identity | Implementation exists in `sts_calibration.py` |

---

## 5. Implementation Roadmap

### Phase 1: Model Loading (research done, not implemented)
- [ ] Create `DeepSeekV4DSpark` class in `vllm/models/deepseek_v4/nvidia/dspark.py`
- [ ] Implement `load_weights()` for all 3 MTP layers + markov + confidence heads
- [ ] Load `dspark_*` config fields from checkpoint

### Phase 2: Draft Generation (core)
- [ ] Implement `forward_backbone()` with bidirectional attention
- [ ] Implement `generate_draft_block()` with Markov sequential sampling
- [ ] Integrate target context extraction from layers 40,41,42

### Phase 3: Speculative Runner Integration
- [ ] Hook DSpark into vLLM's speculative decoding pipeline
- [ ] Either: use existing multi-step runner, or create DSpark-specific worker
- [ ] Pattern from PR #45953 for dynamic SD may help

### Phase 4: Scheduling (Tier 1 first)
- [ ] Tier 1: Fixed gamma=5 verification (default, no changes needed)
- [ ] Tier 2: Per-batch averaged truncation from confidence scores
- [ ] Tier 3: Per-request variable lengths (deferred)
- [ ] Tier 4: Full Hardware-Aware Prefix Scheduler (deferred)

### Phase 5: STS Calibration
- [ ] Generate 1000-5000 calibration samples (draft + verify pairs)
- [ ] Run `calibrate_sts()` from `sts_calibration.py`
- [ ] Store temperatures in model config or separate file

---

## 6. Open Questions / Risks

| Question | Status | Mitigation |
|---|---|---|
| STS temperatures not in checkpoint | Need calibration or fallback to identity | `sts_calibration.py` ready; use identity for Tier 1 |
| `num_nextn_predict_layers=1` vs 3 MTP layers | Config conflict | Add `dspark_num_backbone_layers` field |
| Bidirectional attention support | Needs new code path | Verify `is_causal=False` works across backends |
| Memory (draft model ~8GB) | May be tight for single-GPU | FP8 quant on MTP layers; expert offloading |
| vLLM multi-step runner compatibility | DSpark produces all gamma tokens at once | May need specialized worker |
| Confidence head pruning without CUDA graph break | Tier 2 approach (average across batch) | PR #45953 as reference pattern |
| **GB10: MTP was net throughput loss on single-stream** | DSpark may show same behavior | Benchmark before committing |
| **GB10: SM121 requires special flags** | Triton MLA sparse matmul | Set `VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0` |
| **GB10: No NVLink between nodes** | TP communication latency | Profile RDMA bandwidth; may bottleneck drafting |

---

## 7. Key Files to Read in Other Sessions

### In this repo (vLLM fork):
- `dspark/IMPLEMENTATION_PLAN.md` — Full roadmap
- `dspark/dspark_model_skeleton.py` — Class structure to implement
- `vllm/models/deepseek_v4/nvidia/mtp.py` — Existing MTP code (pattern to follow/extend)

### External:
- `/tmp/pi-github-repos/deepseek-ai/DeepSpec/deepspec/modeling/dspark/markov_head.py` — Markov head reference
- `/tmp/pi-github-repos/deepseek-ai/DeepSpec/deepspec/eval/dspark/draft_ops.py` — Inference flow reference
- `/tmp/pi-github-repos/deepseek-ai/DeepSpec/deepspec/eval/dspark/evaluator.py` — Full eval orchestration
- `/Users/ishansain/Downloads/DSpark_paper_full.md` — OCR'd paper

### GB10 Community Recipes:
- `Entrpi/ds4-spark-vllm` — Single-spark 2-bit hybrid quant recipe
- `tonyd2wild/deepseek-v4-flash-dgx-spark` — Dual-spark TP=2 recipe
- `tonyd2wild/deepseek-v4-flash-dual-spark-recipe` — Alternative dual-spark recipe

---

## 8. Suggested Skills for Next Session

- **codebase-design** — For designing the `DeepSeekV4DSpark` class interface and module seams
- **tdd** — For writing tests before implementing Markov head, confidence head, draft ops
- **diagnosing-bugs** — If CUDA graph or attention backend issues arise on SM121
- **librarian** — If deeper research into vLLM internals, FlashInfer varlen, or GB10 quirks

---

## 9. How to Proceed (GB10 Focus)

1. **Set up the environment:** Pull this branch, verify the Docker image or native build
   works on GB10 with the base V4-Flash model (without DSpark). Use the community recipes.
2. **Read `dspark/README.md`** first for the quick orientation.
3. **Read `dspark/IMPLEMENTATION_PLAN.md`** for the phased roadmap.
4. **Study `dspark/dspark_model_skeleton.py`** for the target class structure.
5. **Study `vllm/models/deepseek_v4/nvidia/mtp.py`** for the existing MTP pattern.
6. **Begin Phase 1** (model loading) — this is the foundation everything else depends on.
7. **Measure memory:** Before loading the DSpark checkpoint, check available memory with
   `nvidia-smi`. The draft model adds ~8 GB to the base model.
8. **Benchmark as you go:** MTP on single-stream was a net loss on GB10. Verify DSpark
   actually improves throughput before investing in Tier 2+ scheduling.

---

*Generated by pi session in ~/Documents/repos/playground*
*Branch: dspark-research on ~/Documents/repos/vllm*
*Hardware target: 2x NVIDIA DGX Spark (GB10 / SM121)*
