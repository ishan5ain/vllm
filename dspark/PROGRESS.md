# DSpark vLLM Integration — Progress & State

> **Date:** 2026-06-27
> **Session 2:** Implementation (phases 0–2) + correctness review + bug fixes
> **Branch:** `dspark-research`

## Implementation Status

```
Phase 0: Prerequisites    ████████████ DONE
Phase 1: Model Loading    ████████████ DONE
Phase 2: Draft Generation ████████████ DONE (code complete, reviewed, 5 blockers fixed)
Phase 3: CUDA Graphs      ░░░░░░░░░░░░ DEFERRED
Phase 4: Scheduling       ░░░░░░░░░░░░ NOT STARTED
Phase 5: STS Calibration  ░░░░░░░░░░░░ CODE EXISTS, NOT INTEGRATED
```

## What Exists (vLLM code changes)

### vllm/models/deepseek_v4/nvidia/model.py — Target model
- **Line 1019**: `_dspark_context_buffer` allocation (shape: `[max_tokens, 3*D]`)
- **Line 1071**: Layer loop captures hidden states from layers 40, 41, 42
- **Line 1090**: DSpark context stash (before hc_head collapse)
- **Line 1398**: `get_dspark_context_hidden_states()` exposed on both `DeepseekV4Model` and `DeepseekV4ForCausalLM`

### vllm/models/deepseek_v4/nvidia/dspark.py — DSpark draft model (873 lines)
- `DeepSeekV4DSparkLayer`: Per-layer backbone (enorm/hnorm + e_proj/h_proj + mtp_block). Output layer has hc_head + shared_head.
- `DSparkInnerModel`: All logic — 3 backbone layers, Markov head (W₁: VocabParallelEmbedding, W₂: ColumnParallelLinear), confidence head (ReplicatedLinear[4352→1]), context projection fc (ReplicatedLinear[12288→4096]).
- `DeepSeekV4DSparkModel`: Thin wrapper for vLLM compatibility. Exposes `self.model` for `load_eagle_model()` (embedding sharing, topk_indices_buffer). Delegates forward/compute_logits/load_weights.
- `forward_dspark_block()`: Full DSpark cycle — context projection, draft embedding, per-layer input projections (enorm/hnorm + e_proj/h_proj), backbone loop, hc_head logits, Markov sequential sampling, confidence computation.
- `load_weights()`: Full weight loading (3 MTP layers + markov + confidence heads). `_rewrite_spec_layer_name()` correctly maps checkpoint paths to `layers.X.*` params. DSpark-specific weights (markov_w1/w2, confidence_proj, fc) are treated as shared top-level weights.
- Step-by-step `forward()` / `compute_logits()`: Backward-compatible with MTP calling pattern.

### vllm/v1/worker/gpu/spec_decode/dspark/speculator.py — DSpark speculator (398 lines)
- `DSparkSpeculator(DraftModelSpeculator)`: 
  - `set_attn()`: Sets up draft KV cache group, block tables
  - `_build_draft_attn_metadata(causal=False)`: Bidirectional attention
  - `_get_anchor_data()`: Extracts anchor tokens/positions from batch
  - `_prepare_dspark_inputs()`: Populates [anchor, mask×4] input IDs
  - `propose()`: Full flow — anchor extraction → context slicing → input prep → attention metadata → `forward_dspark_block()` within `set_forward_context`
  - Phase 2: eager mode only (no CUDA graphs)

### vllm/v1/worker/gpu/model_runner.py — Context plumbing
- Both `propose()` call sites inject DSpark context into `aux_hidden_states` when `get_dspark_context_hidden_states` is available.

### vllm/config/speculative.py — Config override
- `hf_config_override()`: Detects DSpark via `hasattr(hf_config, "dspark_block_size")`, overrides `model_type` → `"deepseek_dspark"`, `num_nextn_predict_layers` → 3, `architectures` → `["DeepSeekV4DSparkModel"]`
- `"dspark"` added to `SpeculativeMethod` Literal

### vllm/model_executor/models/registry.py
- `"DeepSeekV4DSparkModel": ("vllm.models.deepseek_v4", "DeepSeekV4DSparkModel")`

### vllm/models/deepseek_v4/__init__.py
- Re-exports `DeepSeekV4DSparkModel` from nvidia platform module

### vllm/v1/worker/gpu/spec_decode/__init__.py
- Routes `method: "dspark"` → `DSparkSpeculator`

## Architecture Flow (one target step)

```
Target Model forward(layers 0..42)
  │
  ├─→ Captures hidden_states[:,0,:] from layers 40, 41, 42
  │   → concat → [T, 3*D] in _dspark_context_buffer
  │
  ▼
model_runner._execute_model()
  │
  ├─→ get_dspark_context_hidden_states() → aux_hidden_states[0]
  │
  ▼
DSparkSpeculator.propose()
  │
  ├─→ _get_anchor_data()                → anchor tokens [B], positions [B]
  ├─→ target_context[anchor_indices]    → [B, 3*D]
  ├─→ _prepare_dspark_inputs()          → [anchor, mask×4] input_ids [B*γ]
  ├─→ _build_draft_attn_metadata(causal=False)
  ├─→ build_slot_mappings_by_layer()
  │
  ├─→ set_forward_context(attn_metadata, ...)
  │     └─→ model.forward_dspark_block(draft_ids, positions, anchors, context)
  │           │
  │           ├─→ fc(context)            → ctx [B, D]
  │           ├─→ embed([anchor,mask×4]) → [B*γ, D], inject ctx at pos 0
  │           ├─→ Backbone: 3 layers (bidirectional via causal=False)
  │           ├─→ hc_head → base_logits  → [B, γ, V]
  │           ├─→ Markov loop: bias_k = W₂(W₁[prev_k-1])
  │           │   logits_k = base_logits_k + bias_k
  │           │   token_k = argmax(logits_k)
  │           ├─→ Confidence: c_k = σ(w·[h_k; W₁[prev_k-1]])
  │           └─→ return {draft_tokens, draft_logits, confidence}
  │
  ▼
Return [num_reqs, γ] draft tokens
```

## Known Gaps (after review & fixes)

| # | Gap | Severity | Status |
|---|---|---|---|
| 1 | FlashInfer non-causal on SM121 unverified | Low | Deferred to cluster test |
| 2 | Draft KV cache group allocation | Medium | `set_attn()` wired; verify at runtime |
| 3 | Markov head TP=2 correctness | Medium | TP-aware layers follow vLLM patterns; verify at runtime |
| 4 | `fc` weight has no confirmed checkpoint source | Medium | May stay randomly initialized — verify checkpoint key at cluster test |
| 5 | No CUDA graphs | Low | Deferred to Phase 3 |
| 6 | Auto-detection won't set `method="dspark"` | Low | User must pass explicit `method: "dspark"` in spec config |

## Review History

**2026-06-27 — Correctness review:** Found 5 BLOCKERs, 4 WARNINGs, 2 NITs.
All blockers fixed (see `dspark/review-correctness.md`).

**2026-06-27 — Integration review:** Found 0 BLOCKERs, 2 WARNINGs.
Config override, registry, routing, and model_runner plumbing confirmed correct
(see `dspark/review-integration.md`).

## Next Steps (Priority Order)

### Immediate: Cluster Integration Test
1. **Build Docker image** with the dspark-research vLLM fork
2. **Start standard recipe** pointing at `deepseek-ai/DeepSeek-V4-Flash-DSpark`
3. **Speculative config**: `{"method":"dspark","num_speculative_tokens":5}`
4. **Check logs**: model loading → context buffer → draft token generation
5. **If crash**: fix based on stack trace, rebuild, retry

### After Successful Load
6. Run a single inference, verify draft tokens are non-junk
7. Measure acceptance rate (expected: better than MTP's ~2.2/2)
8. Compare throughput vs standard MTP recipe

### Post-Validation
9. **Phase 3**: Add CUDA graphs for the DSpark speculator
10. **Phase 4**: Tier 2 scheduling (batch-averaged confidence truncation)
11. **Phase 5**: STS calibration on held-out set

## Key Design Decisions (Final)

| Decision | Choice |
|---|---|
| Target recipe | **Standard** (`vllm-node`, Ray, default backends). NOT Chthonic b12x. |
| Draft TP | **TP=2** (matches target). No proposer TP mismatch. |
| Model type detection | `hasattr(hf_config, "dspark_block_size")` in `hf_config_override()` |
| Speculator | Custom `DSparkSpeculator(DraftModelSpeculator)`, not MTPSpeculator |
| Attention | `causal=False` in metadata, bidirectional within γ block |
| CUDA graphs | Deferred to Phase 3 |
| Memory | TP=1 OOM'd. TP=2 estimate: ~74–94 GB/node within 102 GB budget. Validate at cluster test. |

## Files Changed (vs upstream vLLM)

```
vllm/config/speculative.py                 (+18/-5)  DSpark detection, method routing
vllm/model_executor/models/registry.py     (+1)      Model registration
vllm/models/deepseek_v4/__init__.py        (+2)      Re-export
vllm/models/deepseek_v4/nvidia/model.py    (+46/-3)  Context capture
vllm/models/deepseek_v4/nvidia/dspark.py   (NEW, 873)  Draft model (3 classes)
vllm/v1/worker/gpu/model_runner.py         (+18/-2)  Context plumbing
vllm/v1/worker/gpu/spec_decode/__init__.py (+6)      Speculator routing
vllm/v1/worker/gpu/spec_decode/dspark/__init__.py     (NEW)
vllm/v1/worker/gpu/spec_decode/dspark/speculator.py   (NEW, 406)
```

## dspark/ Documentation

```
dspark/
├── README.md                    Quick orientation
├── HANDOFF.md                   Session handoff (high-level)
├── PROGRESS.md                  This file — implementation state
├── IMPLEMENTATION_PLAN.md       5-phase roadmap
├── ALGORITHM_REFERENCE.md       Paper equations & pseudocode
├── checkpoint_anatomy.md        Weight structure & shapes
├── dspark_model_skeleton.py     Design stubs (pre-implementation)
├── sts_calibration.py           STS calibration (ready, not integrated)
├── sts_calibration.md           STS docs
├── DSpark_paper_full.md         OCR'd paper
├── gb10_recipes_analysis.md     Cluster recipe analysis → DSpark implications
├── backend_compatibility.md     Backend matrix (standard recipe focus)
├── phase0_findings.md           Phase 0 decisions (TP, methods, config)
├── phase2_dflash_analysis.md    DFlash speculator pattern analysis
└── phase1_implementation_summary.md  (empty — worker failed)
```
