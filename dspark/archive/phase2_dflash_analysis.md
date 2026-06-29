# DSpark Speculator — DFlash Pattern Analysis

> **Date:** 2026-06-27

## DFlash Speculator Flow (reference pattern)

```
propose():
  1. Copy target hidden_states → self.hidden_states buffer
  2. _copy_request_inputs() — copy temperature, seeds, idx_mapping
  3. prepare_dflash_inputs() — build draft input_ids, positions,
     query_start_loc, seq_lens, slot_mappings for context+query tokens
  4. model.precompute_and_store_context_kv() — store target hidden states
     INTO the draft model's KV cache as context (DFlash does cross-attention)
  5. _build_draft_attn_metadata(
       causal=self.dflash_causal  ← set by model config, can be False
     )
  6. build_slot_mappings_by_layer() — per-layer slot mappings
  7. _generate_draft() → _run_model() → model(input_ids, positions)
     (wrapped in set_forward_context with attention metadata)
  8. Return draft_tokens [num_reqs, num_spec_steps]
```

## What DraftModelSpeculator base class provides

- `input_buffers`: InputBuffers (input_ids, positions, query_start_loc, seq_lens)
- `idx_mapping`: [max_num_reqs] int32 on device
- `temperature`, `seeds`: [max_num_reqs] on device
- `draft_tokens`: [max_num_reqs, num_spec_steps] int64 on device
- `draft_logits`: optional [max_num_reqs, num_spec_steps, vocab_size]
- `set_attn(model_state, kv_cache_config, block_tables)`: initializes attention
  backends for draft layers, computes draft_attn_layer_names, attn_groups,
  block_tables, kv_cache_config
- `_build_draft_attn_metadata(num_reqs, num_reqs_padded, num_tokens_padded, ...)`:
  builds attention metadata with configurable `causal` flag
- `sample_draft()`: Gumbel-based or argmax sampling

## DSpark vs DFlash — Key Differences

| Aspect | DFlash | DSpark |
|---|---|---|
| Context injection | KV cache via `precompute_and_store_context_kv()` | Hidden state addition via fc projection + embedding add |
| Draft attention | Can be causal or non-causal (config-driven) | **Must be non-causal** (bidirectional within γ block) |
| Model forward shape | (bonus + γ) tokens per request in one pass | γ tokens per request in one pass |
| Target context needed | Target hidden states per token | **Target hidden states from layers 40,41,42 per anchor** |

## What DSpark speculator is missing

1. **No `set_attn()` call** — The base class sets up `draft_attn_layer_names`,
   `attn_groups`, `block_tables`, `kv_cache_config` via `set_attn()`. These are
   required for `_build_draft_attn_metadata()` and for the model to have
   a valid KV cache.

2. **No input preparation** — The draft model needs `input_ids` (anchor + masks)
   and `positions` populated in `self.input_buffers`. DFlash uses
   `prepare_dflash_inputs()` which is a Triton kernel — DSpark could use a
   simpler Python approach since it doesn't need context slot mappings.

3. **No attention metadata** — `_build_draft_attn_metadata(causal=False)` must
   be called to create the attention metadata dict for `set_forward_context()`.

4. **No forward context wrapping** — The model call is not wrapped in
   `set_forward_context(attn_metadata, ...)`, so the attention layers can't
   find KV cache or slot mappings.

5. **No KV cache for draft tokens** — The draft model's own attention KV
   cache isn't set up. Without `set_attn()`, the attention backends don't
   know about the draft layers.

6. **Wrong hidden_states shape** — `self.hidden_size` includes `hc_mult`
   scaling (×4 for DeepSeek V4), but the DSpark context is raw hidden_size.
   Need to be careful about buffer sizing.

7. **InputBatch field names need fixing** — The `propose()` code references:
   - `input_batch.token_ids_cpu` — does NOT exist. Use `input_batch` token
     access via `req_states` or `input_buffers`.
   - `input_batch.seq_lens` — EXISTS (torch.Tensor)
   - `input_batch.query_start_loc` — EXISTS (torch.Tensor)
   - `input_batch.num_reqs` — EXISTS (int)
   - `input_batch.num_tokens` — EXISTS (int)
   - `input_batch.positions` — EXISTS (torch.Tensor)
   - `input_batch.idx_mapping` — EXISTS (torch.Tensor)
   - `input_batch.seq_lens_cpu_upper_bound` — EXISTS

8. **Token ID access** — DSpark needs the anchor token ID for each request.
   In the DFlash kernel, it reads `target_positions[valid_ctx_end-1]` to get
   positions, and tokens come from `last_sampled` or `next_prefill_tokens`.
   DSpark should follow the same pattern.

## Revised DSpark speculator plan

### Model loading (load_draft_model)
- Use `load_eagle_model()` (same as MTP) — it wires up embedding sharing
  and topk_indices_buffer. DSpark model doesn't need DFlash's non-causal
  attention config override since we set `causal=False` in metadata.

### Attention setup (set_attn)
- Call `super().set_attn()` — this initializes `attn_groups`, `block_tables`,
  `kv_cache_config`, and identifies `draft_kv_cache_group_id`.
- DSpark needs exactly 1 KV cache group for its draft layers (same as DFlash).

### Input preparation (new method: prepare_dspark_inputs)
Python-only (no Triton kernel needed since DSpark doesn't need per-token
context slot mappings). For each request:
  - draft_input_ids = [anchor_token, mask, mask, mask, mask]
  - draft_positions = [anchor_pos+1, anchor_pos+2, ..., anchor_pos+γ]
  - draft_query_start_loc = [0, γ, 2γ, ..., R*γ]
  - draft_seq_lens = anchor_pos + γ + 1 (context + draft)

### propose() flow
```
propose():
  1. Copy target hidden states → self.hidden_states (for model if needed)
  2. _copy_request_inputs() — copy temperature, seeds
  3. Extract anchor tokens: use last_sampled/next_prefill_tokens
     (same as DFlash kernel logic, but in Python)
  4. Get DSpark context from aux_hidden_states[0] — [T, 3*D]
     Index by anchor positions → [B, 3*D]
  5. prepare_dspark_inputs() — populate input_buffers
  6. _build_draft_attn_metadata(causal=False, num_query_per_req=γ)
  7. build_slot_mappings_by_layer()
  8. Call model.forward_dspark_block(anchors, anchor_positions, context)
     wrapped in set_forward_context(attn_metadata, ...)
  9. Return draft_tokens [num_reqs, γ]
```

### Attention backend for DSpark
DFlash model doesn't set `use_non_causal` in the vllm_config. Instead,
it passes `causal=dflash_causal` to `_build_draft_attn_metadata()`.
DSpark should do the same: `_build_draft_attn_metadata(causal=False)`.

BUT: the standard recipe uses `VLLM_USE_BREAKABLE_CUDAGRAPH=0` and
default attention backends (FlashInfer). FlashInfer supports non-causal
attention. The `causal=False` flag in the attention metadata should
be sufficient.
