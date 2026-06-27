# DSpark vLLM Integration — Handoff Document

> **Date:** 2026-06-27
> **Session:** playground → vllm fork
> **Branch:** `dspark-research` (pushed to `origin`)

## What This Is

This is a complete handoff of DSpark integration research. Any session or agent
picking this up should be able to understand the full picture: what DSpark is,
what we know, what we've built, and what still needs to be done.

---

## 1. The Big Picture

**DSpark** is DeepSeek's production speculative decoding framework. It accelerates
LLM inference by:

1. **Better drafting** — A semi-autoregressive head (Markov W₁W₂, rank 256) adds
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
| Checkpoint `config.json` | Exact hyperparams: γ=5, rank=256, layers=[40,41,42], mask=128799 |
| Checkpoint `model.safetensors.index.json` | Full weight structure with shapes |
| DeepSpec repo (`deepseek-ai/DeepSpec`) | PyTorch reference implementation of Markov head, confidence head, draft ops |
| vLLM `vllm/models/deepseek_v4/nvidia/mtp.py` | Existing MTP code to extend |
| vLLM PR #40860, #41404, #45953 | Pattern for dynamic SD with CUDA graph safety |
| Slack thread (Kaichao You et al.) | Practical scoping: 4-tier scheduling, start simple |

### Key Architecture Facts

```
DSpark Draft Model (loaded from checkpoint):

  Input: [anchor_token] [mask] [mask] [mask] [mask]  (γ=5)
  
  mtp.0: DeepseekV4DecoderLayer (attn + MoE FFN, bidirectional)
  mtp.1: DeepseekV4DecoderLayer (attn + MoE FFN, bidirectional)  
  mtp.2: DeepseekV4DecoderLayer + hc_head + shared_head  (output layer)
  
  + markov_head: W₁[129280×256], W₂[256×129280]  (on mtp.2)
  + confidence_head: Linear[4352→1]  (on mtp.2)
```

**Target context:** Hidden states from V4 layers 40, 41, 42 concatenated and projected
into draft hidden space via `fc: Linear[12288→4096]`.

**Inference flow (one cycle):**
1. Target forward → extract hidden states from layers 40,41,42
2. Project context → draft_ctx = Norm(fc(concat(h40,h41,h42)))
3. Draft backbone forward (bidirectional, is_causal=False) → hidden [B, γ, 4096]
4. hc_head + shared_head → base_logits [B, γ, 129280]
5. Markov sequential loop (k=1..5): bias = W₂(W₁[prev]), logits = base + bias, sample
6. Confidence: per-position cₖ = σ(w·[hₖ; W₁[prev]])
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
- `calibrate_sts(logits, labels, gamma=5)` — Grid search Tₖ ∈ [0.1, 5.0], 50 steps, minimizes ECE
- `apply_temperatures(logits, temps)` — Apply calibrated σ(c/T) during inference
- `compute_cumulative_survival(logits, temps)` — ∏ σ(·/T) for prefix scheduler
- `compute_ece(probs, labels)` — ECE with equal-frequency binning
- `compute_default_temperatures()` — Identity fallback [1.0]×5

---

## 4. Design Decisions (Settled)

| Decision | Choice | Rationale |
|---|---|---|
| Integration class | New `DeepSeekV4DSpark` (parallel to MTP) | Avoids breaking existing MTP; cleaner separation |
| Layer count override | Add `dspark_num_backbone_layers: 3` config field | Cleaner than repurposing `num_nextn_predict_layers` |
| Attention mode | Bidirectional within block (`is_causal=False`) | Required by DSpark architecture |
| Scheduling phasing | Tier 1 → Tier 2 → Tier 3 → Tier 4 | Slack discussion; start with fixed γ=5 |
| STS calibration | Offline grid search; fallback to identity | Implementation exists in `sts_calibration.py` |

---

## 5. Implementation Roadmap

### Phase 1: Model Loading ✓ (research done, not implemented)
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
- [ ] Tier 1: Fixed γ=5 verification (default, no changes needed)
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
| vLLM multi-step runner compatibility | DSpark produces all γ tokens at once | May need specialized worker |
| Confidence head pruning without CUDA graph break | Tier 2 approach (average across batch) | PR #45953 as reference pattern |

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

---

## 8. How to Proceed

If you're picking this up in a new session:

1. **Read `dspark/README.md`** first for the quick orientation
2. **Read `dspark/IMPLEMENTATION_PLAN.md`** for the phased roadmap
3. **Study `dspark/dspark_model_skeleton.py`** for the target class structure
4. **Study `vllm/models/deepseek_v4/nvidia/mtp.py`** for the existing MTP pattern
5. **Begin Phase 1** (model loading) — this is the foundation everything else depends on

For STS calibration: `dspark/sts_calibration.py` is self-contained and ready to use.
The companion `sts_calibration.md` explains the theory and usage.

---

*Generated by pi session in ~/Documents/repos/playground*
*Branch: dspark-research on ~/Documents/repos/vllm*
