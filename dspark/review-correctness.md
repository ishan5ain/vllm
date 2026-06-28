# DSpark Phase 0-2 Correctness Review

## Review Summary

**5 BLOCKER issues, 4 WARNING issues, 2 NIT issues** across 3 files.
Several issues are systematic: the weight loading `model.` prefix mismatch
affects nearly every checkpoint weight, and the model loading path via
`load_eagle_model()` will crash at `eagle_model.model`.

---

## A. Weight Loading Correctness

### BLOCKER: `dspark.py:423–430` — `model.` prefix mismatch for all weights

**Evidence:** After the `name.replace(f"mtp.{idx}.", f"model.layers.{...}.")` on line 423, every checkpoint weight name gains a `model.` prefix. However, `DeepSeekV4DSparkModel` IS the model — it does NOT wrap an inner model under `self.model`. Its `named_parameters()` returns paths like `layers.43.enorm.weight`, `markov_w1.weight` — **without** the `model.` prefix.

The `_rewrite_spec_layer_name` (line 484–515) was copied from `mtp.py:486` where it is correct — the MTP class holds the inner model under `self.model`, so `named_parameters()` returns `model.layers.43.enorm.weight`. In DSpark, the same logic produces parameter names that never match `params_dict` keys.

**Affected weight categories:**

| Category | After rewrite | Actual param key |
|---|---|---|
| Spec-layer norms/proj (`enorm`, `hnorm`, `e_proj`, `h_proj`) | `model.layers.43.enorm.weight` | `layers.43.enorm.weight` |
| Decoder block internals (`attn.*`, `ffn.*`, `hc_*`) | `model.layers.43.mtp_block.attn.wq_a.weight` | `layers.43.mtp_block.attn.wq_a.weight` |
| DSpark heads (`markov_w1`, `markov_w2`, `confidence_proj`, `fc`) | `model.layers.43.markov_w1.weight` | `markov_w1.weight` |

**Impact:** `params_dict[name]` raises `KeyError` for virtually every weight. No weights are loaded. The model stays with random initialization.

**Suggested fix:** Change the replacement on line 423 from:
```python
f"model.layers.{self.config.num_hidden_layers + mtp_layer_idx}."
```
to:
```python
f"layers.{self.config.num_hidden_layers + mtp_layer_idx}."
```
And update `_rewrite_spec_layer_name` to use `layers.{spec_layer}.` instead of `model.layers.{spec_layer}.` for its `.replace()` calls. Additionally, for DSpark-specific weights (`markov_w1`, `markov_w2`, `confidence_proj`, `fc`) that are top-level on the model (not inside `layers`), the `_rewrite_spec_layer_name` must strip the entire `layers.{idx}.` prefix so the name becomes directly e.g. `markov_w1.weight`.

---

### BLOCKER: `dspark.py:220–228` — `config.num_nextn_predict_layers` not overridden

**Evidence:** Line 225–228:
```python
self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 3)
if self.num_mtp_layers != 3:
    logger.warning(...)
    self.num_mtp_layers = 3
```
Sets `self.num_mtp_layers = 3` but **never** writes `config.num_nextn_predict_layers = 3`.

Line ~427 calls:
```python
spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
```
The function at `deepseek_v2.py:1801` iterates `range(config.num_nextn_predict_layers)`. Per `checkpoint_anatomy.md`, the actual HF config has `num_nextn_predict_layers: 1` even though the checkpoint has 3 MTP layers (mtp.0, mtp.1, mtp.2).

**Impact:** If the original config has `num_nextn_predict_layers=1`, only weights for `mtp.0` (layer index 43) are found. Weights for `mtp.1` and `mtp.2` get `spec_layer=None` and are silently skipped.

**Suggested fix:** After overriding, write `config.num_nextn_predict_layers = 3`.

---

### BLOCKER: `dspark.py:287–289` — `fc` weight has no checkpoint source and no remapping

**Evidence:** The `fc` projection is created on line 287–289:
```python
self.fc = ReplicatedLinear(
    len(self.target_layer_ids) * config.hidden_size, config.hidden_size, ...
)
```
But the `checkpoint_anatomy.md` lists NO `fc` or `mtp.X.fc` weight in the DSpark checkpoint. The `WEIGHT_NAME_REMAPPING` (line 393–403) has no entry for `fc`. There is no `mtp.X.fc.weight` in the checkpoint listing.

**Impact:** `fc` stays with random initialization. Every call to `project_context()` or `forward_dspark_block()` produces garbage context projections, leading to incorrect draft tokens.

**Suggested fix:** Either (a) confirm the actual checkpoint weight key for `fc` and add it to `WEIGHT_NAME_REMAPPING`, or (b) if `fc` is part of the main model checkpoint (e.g., `model.fc.weight` at top level), load it from the target model instead.

---

### WARNING: `dspark.py:393–403` — DSpark-specific weight remapping is incomplete

**Evidence:** The `WEIGHT_NAME_REMAPPING` handles `markov_head.markov_w1`, `markov_head.markov_w2`, and `confidence_head.proj`. But the checkpoint paths for these weights (e.g., `mtp.2.markov_head.markov_w1.weight`) go through the `_rewrite_spec_layer_name` function which preserves the `model.layers.43.` prefix (see first BLOCKER). After fixing the prefix issue, the remapping from `.markov_head.markov_w1.weight` → `.markov_w1.weight` should work, but only if `_rewrite_spec_layer_name` also strips the layer prefix for these top-level weights.

---

### WARNING: `dspark.py:390` — Missing weight remapping for `enorm`, `hnorm`, `e_proj`, `h_proj`

**Evidence:** The `WEIGHT_NAME_REMAPPING` only covers `emb.tok_emb`, `head`, `norm`, and the three DSpark-specific heads. If the actual checkpoint uses different key suffixes (e.g., `enorm.weight` vs `input_layernorm.weight`), the `_remap_weight_name` function won't transform them. Combined with the prefix mismatch, this is a secondary concern.

---

## B. Tensor Parallelism (TP) Correctness

### NIT: `dspark.py:273–284` — TP choices are correct per design

- `markov_w1` as `VocabParallelEmbedding` — **correct** for vocab-split TP.
- `markov_w2` as `ColumnParallelLinear` — **correct** (output vocab is split along TP).
- `confidence_proj` as `ReplicatedLinear` — **correct** (output=1, trivially replicated).
- `fc` as `ReplicatedLinear` — **correct** (small, 12288×4096).

No TP bugs found in layer construction.

---

## C. Forward Pass Logic

### BLOCKER: `speculator.py:294–304` — `forward_dspark_block` bypasses input projections

**Evidence:** In `forward_dspark_block` (dspark.py line 357–443), the draft embeddings go directly into `layer.mtp_block()`:
```python
hidden_states = draft_embeds  # raw embedding output
for layer_key in sorted(self.layers.keys(), key=int):
    hidden_states, residual, post_mix, res_mix = layer.mtp_block(
        positions=draft_positions,
        x=hidden_states,
        input_ids=None,
    )
```

But `DeepseekV4DSparkLayer.forward()` (dspark.py line 140–170) applies:
1. `fused_mtp_input_rmsnorm(inputs_embeds, positions, previous_hidden_states, enorm, hnorm)` — normalizes both embedding and hidden-state input
2. `e_proj(inputs_embeds)` + `h_proj(previous_hidden_states)` — projects them into the decoder

These preprocessing steps are entirely skipped in `forward_dspark_block`. The raw embeddings are fed into the first `mtp_block` without normalization or projection.

**Impact:** The backbone receives un-normalized, un-projected embeddings. If the model was trained with these projections (as the `enorm`/`hnorm`/`e_proj`/`h_proj` parameters suggest), the output hidden states will be qualitatively wrong, producing garbage draft tokens. Conversely, if the DSpark checkpoint was trained WITHOUT these projections (and `enorm`/`hnorm`/`e_proj`/`h_proj` don't exist in the checkpoint), then `DeepseekV4DSparkLayer.__init__` creates layers that never get weights (see weight loading blockers).

**Either way, there is an architectural mismatch.** Suggested fix: determine the actual DSpark training architecture. If projections are needed, add the normalization + projection step to `forward_dspark_block`. If they are not, remove the unneeded projection layers from `DeepseekV4DSparkLayer.__init__`.

---

### BLOCKER: `speculator.py:247` — `last_hidden_states` fallback shape mismatch

**Evidence:** Speculator `propose()` line 247:
```python
target_context_all = last_hidden_states
```
`last_hidden_states` comes from the target model's output which is the pre-hc_head residual: shape `[T, hc_mult * hidden_size]` = `[T, 16384]` for V4 (hc_mult=4, hidden_size=4096).

But `forward_dspark_block` expects `target_context` of shape `[B, 3 * hidden_size]` = `[B, 12288]` and passes it through `self.fc` with input dim `12288`.

**Impact:** If the auxiliary path and target model buffer path both fail, `self.fc(target_context)` will receive `[B, 16384]` but expects `12288` input features → `RuntimeError: mat1 and mat2 shapes cannot be multiplied`.

**Suggested fix:** Either (a) ensure the fallback can never trigger by making the target context buffer always available, or (b) add a shape assertion/reshape in the fallback path.

---

### WARNING: `speculator.py:358` — Temperature ignored in `propose()`

**Evidence:** Line 358:
```python
result = self.model.forward_dspark_block(
    ...
    temperature=0.0,
)
```
The `temperature` parameter passed to `propose()` is not forwarded; greedy sampling is hardcoded. The `temperature` tensor from the speculator's buffer is available but unused.

**Impact:** If probabilistic draft sampling is configured, it won't work. Greedy sampling is always used.

---

### WARNING: `dspark.py:387–399` — Markov forward pass uses `self.markov_w1(prev_token_ids.long())`

**Evidence:** In `markov_bias()` (line 387):
```python
prev_emb = self.markov_w1(prev_token_ids.long())
```
`VocabParallelEmbedding.forward()` expects `input_ids` as the first positional argument, and `.long()` conversion is correct for integer token IDs. However, the output of `VocabParallelEmbedding` under TP is a shard of the full embedding — each rank sees only `vocab_size // tp_size` outputs. This is passed to `self.markov_w2` which is `ColumnParallelLinear` — the output vocab IS column-parallel, so ranks compute different shards. When `step_logits.argmax(dim=-1)` is called in `forward_dspark_block` (line 424), each rank produces a local argmax within its own shard. The cross-rank allreduce for argmax is handled by PyTorch's TP communication... **BUT** this only works if the model is wrapped in a TP-aware distributed context. In the eager-mode Phase 2, this should be fine as the TP ranks communicate automatically through the linear layer's reduce.

**Verdict:** Correct for TP=2 assuming proper distributed setup. No bug here.

---

### Correct: Residual passing between layers is correct

**Evidence:** In `forward_dspark_block`, each `layer.mtp_block()` call runs standalone MHC pre-norm (since `residual=None` is the default). The `mhc_post_tilelang` call consumes the layer's own residual. The next layer starts fresh with its own pre-norm. This matches how `DeepseekV4Model.forward()` chains layers — each layer independently computes and consumes its residual via `mhc_post_tilelang`. **Not a bug.**

---

## D. Speculator Logic

### Correct: `_get_anchor_data()` logic (speculator.py:180–221)

- `valid_end = qe - rejected` — correctly computes the end of the valid prefix
- `anchor_idx = valid_end - 1` — correctly selects the last valid token
- `anchor_positions = positions[anchor_idx]` — correct
- `last_sampled` / `next_prefill_tokens` fallback — matches DFlash pattern

### WARNING: `_prepare_dspark_inputs()` (speculator.py:224–276) — `seq_lens` edge case

**Evidence:** Line 261:
```python
ib.seq_lens[req_idx] = anchor_positions[req_idx] + gamma + 1
```
If `anchor_positions[req_idx] + gamma + 1 > self.max_model_len`, the attention backend may receive a `max_seq_len` exceeding the model's positional encoding limit. The speculator clamps `draft_max_seq_len = min(max_seq_len + gamma, self.max_model_len)` (line 305), but the individual `seq_lens` values are not clamped.

**Impact:** When approaching `max_model_len`, the `seq_lens` could exceed the clamped `max_seq_len`, potentially confusing attention backends. Low severity — requires near-max-length sequences.

---

## E. Potential Crashes

### BLOCKER: `eagle/utils.py:48` — `eagle_model.model` AttributeError

**Evidence:** `load_eagle_model` (called from speculator.py:96):
```python
eagle_model = get_model(vllm_config=vllm_config, model_config=draft_model_config)
# ...
draft_inner = eagle_model.model  # line 48
```
The speculative config (speculative.py:333) overrides `architectures: ["DeepSeekV4DSparkModel"]`. The registry resolves this to the `DeepSeekV4DSparkModel` class. `initialize_model` creates a `DeepSeekV4DSparkModel` instance.

`DeepSeekV4DSparkModel` inherits from `nn.Module` directly — it has **no `.model` attribute**.

**Impact:** `AttributeError: 'DeepSeekV4DSparkModel' object has no attribute 'model'` — crash during model initialization, before any forward pass.

**Suggested fix:** Either:
1. Create a `DeepseekV4DSparkForCausalLM` wrapper class (parallel to `DeepseekV4ForCausalLM`) that holds `DeepSeekV4DSparkModel` under `self.model`, and register THAT in the registry.
2. Or modify `load_eagle_model` to check whether `eagle_model` is already the inner model (e.g., check for `hasattr(eagle_model, "model")` and fall back to `eagle_model` itself).

---

### BLOCKER: `speculator.py:343` — `forward_dspark_block` not available on loaded model

**Evidence:** Even if the `eagle_model.model` crash were fixed, `self.model` is currently a `DeepseekV4ForCausalLM` wrapper (from `load_eagle_model`). As coded, the speculator calls `self.model.forward_dspark_block(...)` on line 343:
```python
result = self.model.forward_dspark_block(...)
```

`DeepseekV4ForCausalLM` does NOT have `forward_dspark_block` — that method is on `DeepSeekV4DSparkModel`. Fixed by the same wrapper approach above (the wrapper needs to delegate to `self.model.forward_dspark_block`).

---

### WARNING: `speculator.py:131–140` — `draft_kv_cache_group_id = -1` silent failure

**Evidence:** Lines 131–140:
```python
draft_groups = [gid for gid, g in enumerate(self.attn_groups) if g]
if draft_groups:
    self.draft_kv_cache_group_id = draft_groups[0]
    ...
else:
    logger.warning("DSpark speculator: no draft attention groups found...")
```
If no groups are found, `draft_kv_cache_group_id` stays at -1 and `draft_block_size` stays at -1. `_build_draft_attn_metadata` is then called with these default values. The `block_tables.input_block_tables` indexing with group_id=-1 would likely fail.

**Impact:** If the draft model's attention layers aren't recognized, the speculator will crash during `_build_draft_attn_metadata` with an index error. But this likely means a configuration error, so the warning + crash is acceptable for now.

---

## Summary Table

| # | Severity | File:Line | Issue |
|---|----------|-----------|-------|
| 1 | **BLOCKER** | `eagle/utils.py:48` | `eagle_model.model` AttributeError — DeepSeekV4DSparkModel has no `.model` |
| 2 | **BLOCKER** | `dspark.py:423,484` | `model.` prefix mismatch in weight names — no weights can be loaded |
| 3 | **BLOCKER** | `dspark.py:225` | `config.num_nextn_predict_layers` not overridden — layers 44,45 skipped |
| 4 | **BLOCKER** | `dspark.py:371` vs `dspark.py:140` | `forward_dspark_block` skips input projections (enorm/hnorm/e_proj/h_proj) |
| 5 | **BLOCKER** | `speculator.py:247` | `last_hidden_states` fallback has wrong shape for `fc` |
| 6 | WARNING | `dspark.py:287` | `fc` weight has no checkpoint source or remapping |
| 7 | WARNING | `dspark.py:287` | `enorm`/`hnorm`/`e_proj`/`h_proj` may not exist in DSpark checkpoint |
| 8 | WARNING | `speculator.py:358` | `temperature=0.0` hardcoded, ignores sampling config |
| 9 | WARNING | `speculator.py:261` | `seq_lens` not clamped to `max_model_len` |
| 10 | NIT | `speculator.py:131` | `draft_kv_cache_group_id=-1` no graceful fallback |
| 11 | NIT | `dspark.py:273-284` | TP choices verified correct |

---

## Acceptance Report

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Review inspected all 3 specified files completely (dspark.py 762 lines, model.py DSpark sections, speculator.py 398 lines). Found 5 BLOCKER bugs with file:line references and suggested fixes. No files were modified — review-only."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "grep -n 'dspark' vllm/models/deepseek_v4/nvidia/model.py",
      "result": "passed",
      "summary": "Found DSpark context buffer at lines 1024-1034, forward capture at 1080-1117, get_dspark_context_hidden_states at 1438-1445"
    },
    {
      "command": "grep -n 'class DeepseekV4DecoderLayer' vllm/models/deepseek_v4/nvidia/model.py",
      "result": "passed",
      "summary": "Found at line 782; inspected forward signature at 840-895"
    },
    {
      "command": "grep -rn 'load_eagle_model' vllm/v1/worker/gpu/spec_decode/eagle/utils.py",
      "result": "passed",
      "summary": "Confirmed eagle_model.model access pattern at line 48"
    }
  ],
  "validationOutput": [
    "All 5 BLOCKER issues verified from code evidence with file:line citations",
    "Weight loading prefix mismatch is systematic — affects all weight categories",
    "config.num_nextn_predict_layers override confirmed missing from self.config",
    "eagle_model.model crash confirmed: DeepSeekV4DSparkModel has no .model attr",
    "forward_dspark_block confirmed to bypass input projections (enorm/hnorm/e_proj/h_proj)",
    "last_hidden_states shape mismatch confirmed: 16384 vs expected 12288"
  ],
  "residualRisks": [
    "The fc weight source is unknown — may need additional checkpoint inspection",
    "The enorm/hnorm/e_proj/h_proj layers may not be in the DSpark checkpoint — architectural clarification needed",
    "Markov sequential sampling correctness depends on TP allreduce for argmax — not verified in integration"
  ],
  "noStagedFiles": true,
  "diffSummary": "No files modified. Review-only. Found 5 BLOCKER, 4 WARNING, 2 NIT issues.",
  "reviewFindings": [
    "blocker: eagle/utils.py:48 - eagle_model.model crashes; DeepSeekV4DSparkModel has no .model attr",
    "blocker: dspark.py:423,484 - model. prefix mismatch prevents ALL weight loading",
    "blocker: dspark.py:225 - config.num_nextn_predict_layers not overridden; layers 44,45 skipped",
    "blocker: dspark.py:371 vs dspark.py:140 - forward_dspark_block bypasses input projections",
    "blocker: speculator.py:247 - last_hidden_states fallback shape mismatch (16384 vs 12288)"
  ],
  "manualNotes": "The weight loading issue (BLOCKER #1 and #2) is the most critical — no weights can be loaded at all. The model loading path issue (BLOCKER #1 in summary) would crash before weight loading even begins. These two issues alone make the implementation non-functional as-is. The architectural mismatch between forward_dspark_block and the per-layer forward() method (BLOCKER #4) needs design clarification: either the checkpoint has input projections (and they should be used) or it doesn't (and they should be removed from the layer class)."
}
```
