# DSpark Phase 0: Prerequisites — Findings

> **Date:** 2026-06-27
> **Status:** Investigation complete. Ready for Phase 1.

## Decision 1: Draft Model TP Strategy

**Finding:** The existing `DraftModelProposer._raise_if_draft_tp_mismatch()` in
`vllm/v1/spec_decode/draft_model.py` enforces that `draft_tp == target_tp`.
The check exists to prevent torch compile cache corruption when different TP
sizes compile on the same rank.

**Decision: DSpark draft model uses TP=2 (matches target TP=2).**

This is the simplest path:
- No code changes to the proposer or TP checks
- No torch compile cache issues
- The draft model (~8 GB) gets sharded across 2 nodes via the same TP mechanism
  as the target model
- Output all-reduce is negligible: γ=5 × vocab_size=129280 × 4 bytes = ~2.6 MB
- Standard recipe uses `DG_JIT_USE_NVRTC=0` (no NVRTC JIT), further reducing
  compile concerns

**What this means for the DSpark model class:**
- `DeepSeekV4DSpark` must be TP-aware (column-parallel linear, row-parallel linear)
- All `ReplicatedLinear` / `ColumnParallelLinear` / `RowParallelLinear` layers
  in the draft model need proper TP sharding
- Markov head W₁ (Embedding [129280×256]) and W₂ (Linear [256×129280]) need
  TP-aware layout
- Follow the same TP patterns as `DeepSeekV4MTP`

**Alternative (deferred):** TP=1 draft model would require:
- A DSpark-specific proposer subclass that overrides `_raise_if_draft_tp_mismatch()`
- Setting `draft_tensor_parallel_size: 1` in speculative config
- Ensuring torch compile caches don't collide (not currently an issue since
  the standard recipe uses `DG_JIT_USE_NVRTC=0`)

## Decision 2: Speculative Method Integration

**Finding:** The method → model class chain works as follows:

```
--speculative-config '{"method":"mtp",...}'
  ↓
vllm/v1/spec_decode/__init__.py: init_speculator()
  method == "mtp" → MTPSpeculator
  ↓
vllm/v1/worker/gpu/spec_decode/mtp/speculator.py: MTPSpeculator.load_draft_model()
  calls load_eagle_model(target_model, vllm_config)
  ↓
vllm/v1/worker/gpu/spec_decode/eagle/utils.py: load_eagle_model()
  calls get_model(vllm_config=..., model_config=draft_model_config)
  ↓
vllm/model_executor/model_loader.py: get_model()
  looks up architecture name in registry → instantiates class
```

**Registry entry for MTP:**
```python
# vllm/model_executor/models/registry.py
"DeepSeekV4MTPModel": ("vllm.models.deepseek_v4", "DeepSeekV4MTP"),
```

**Config override for MTP (vllm/config/speculative.py):**
```python
# hf_config_override() transforms the draft model's HF config:
if hf_config.model_type == "deepseek_v4":
    hf_config.model_type = "deepseek_mtp"
    n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
    hf_config.update(
        {"n_predict": n_predict, "architectures": ["DeepSeekV4MTPModel"]}
    )
```

**What we need for DSpark (3 files, minimal changes):**

### 1) `vllm/config/speculative.py` — Config override
Add `dspark_block_size` detection to the existing `deepseek_v4` branch (see
Decision 3 for code). Also add `"deepseek_dspark"` to `MTPModelTypes` Literal
(or create `DSparkModelTypes`).

### 2) `vllm/model_executor/models/registry.py` — Model registration
```python
"DeepSeekV4DSparkModel": ("vllm.models.deepseek_v4", "DeepSeekV4DSpark"),
```

### 3) `vllm/v1/worker/gpu/spec_decode/__init__.py` — Speculator routing
```python
elif speculative_config.method == "dspark":
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
    return MTPSpeculator(vllm_config, device)
```

Reuse `MTPSpeculator` (which extends `AutoRegressiveSpeculator`) — DSpark
produces γ draft tokens in one forward pass, same autoregressive flow as MTP
with `num_speculative_tokens > 1`.

## Decision 3: Model Type for DSpark Checkpoint

**Finding:** The DSpark checkpoint's `config.json` has been examined:
- `model_type: "deepseek_v4"` — **same as the base model**
- `architectures: ["DeepseekV4ForCausalLM"]` — **same as the base model**
- `num_nextn_predict_layers: 1` — same as non-DSpark MTP
- `dspark_*` fields present: `dspark_block_size: 5`, `dspark_noise_token_id: 128799`,
  `dspark_target_layer_ids: [40,41,42]`, `dspark_markov_rank: 256`

**Implication:** Auto-detection can't distinguish DSpark from the base model by
`model_type` alone. We need to check for `dspark_block_size` to differentiate.

**Plan — follow the existing pattern in `SpeculativeConfig.hf_config_override()`:**

```python
# In vllm/config/speculative.py, hf_config_override():
if hf_config.model_type == "deepseek_v4":
    if hasattr(hf_config, "dspark_block_size"):
        # DSpark checkpoint detected
        hf_config.model_type = "deepseek_dspark"
        hf_config.update({
            "n_predict": getattr(hf_config, "dspark_block_size", 5),
            "num_nextn_predict_layers": 3,  # always 3 for DSpark
            "architectures": ["DeepSeekV4DSparkModel"],
        })
    else:
        # Standard MTP (existing behavior, unchanged)
        hf_config.model_type = "deepseek_mtp"
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update({
            "n_predict": n_predict,
            "architectures": ["DeepSeekV4MTPModel"],
        })
```

This is a minimal change (add `hasattr` check + else clause to existing
`deepseek_v4` branch). No refactoring needed.

## Pending: Memory Budget Measurement

**Attempted 2026-06-27:** Tried loading base `DeepSeek-V4-Flash` at TP=1 on a
single GB10 node (128 GB unified memory) via the `vllm-node` Docker image.

**Result: OOM (out of memory).** The base model at TP=1 does not fit on a
single 128 GB GB10 node. This confirms why the cluster recipe uses TP=2 —
each node holds roughly half the model weights.

**Implication for DSpark:**
- At TP=2 across 2 nodes, each node holds ~50-60 GB for base model weights
  plus KV cache allocation
- The DSpark draft model (~8 GB total) is also TP=2 sharded → ~4 GB per node
- With `gpu_memory_utilization: 0.8` → ~102 GB usable per node of 128 GB
- Estimated: 50-60 GB (weights) + 20-30 GB (KV cache) + 4 GB (DSpark) ≈ 74-94 GB
- This should fit within the 102 GB budget, but it's worth monitoring

**Next step:** Skip standalone memory measurement. Proceed directly to
Phase 2 (draft generation) with the full 2-node cluster setup. The first
DSpark load attempt will naturally validate the memory budget — if it OOMs,
the mitigations are:
1. Reduce `gpu_memory_utilization` from 0.8 to 0.75 (frees ~6 GB)
2. Reduce `max_num_seqs` from 4 to 3 (reduces KV cache allocation)
3. Reduce `max_model_len` from 500K to a lower value

## Summary: Phase 1 Task List (Revised)

1. **[x] Draft TP strategy** — Use TP=2 (matches target). No proposer changes needed.
2. **[x] Spec method integration plan** — Register model, add method, handle auto-detection.
3. **[ ] Memory budget measurement** — Run `nvidia-smi` on GB10 with base model loaded.
4. **[x]** Check DSpark checkpoint `config.json` — done. `model_type: deepseek_v4` with `dspark_*` fields.
   Differential detection via `hasattr(hf_config, "dspark_block_size")` in `hf_config_override()`.
6. **[ ]** Register `DeepSeekV4DSpark` in model registry.
7. **[ ]** Add `"dspark"` method to `init_speculator()`.
8. **[ ]** Implement `load_weights()` (following existing `DeepSeekV4MTP` pattern).
9. **[ ]** Load `dspark_*` config fields from checkpoint.
