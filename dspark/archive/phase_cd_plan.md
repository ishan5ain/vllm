# DSpark fixes C & D — design & status

> Companion to `dspark/checkpoint_anatomy.md` → "✅ VERIFIED MAPPING".
> Records the deep architecture fixes that are NOT pure weight-loading edits.

## Status

| Fix | What | Status |
|---|---|---|
| **D** | Target-context capture = mean over hc_mult of the full post-mHC layer output | ✅ DONE (untested on cluster) |
| **C** | Target context enters via DSparkAttention cross-attention (per-stage `main_kv` in draft KV cache) | ⏳ INTERIM in place; proper version designed below |

A/B/E/F (model definition + weight loading) are done in `dspark.py`; the
`load_weights` completeness assertion now passes only when every parameter has
a checkpoint source.

## Fix D (done) — what changed and the risk

`vllm/models/deepseek_v4/nvidia/model.py`, DSpark context capture in
`DeepseekV4Model.forward`:

- **Before:** captured `hidden_states[:, 0, :]` — the first hc stream of the
  *pre-`mhc_post`* FFN output.
- **After:** `full = mhc_post_tilelang(hidden_states, residual, post_mix,
  res_mix)` then `full.mean(dim=1)` — the full per-layer output averaged over
  `hc_mult`, matching reference `Transformer.forward` (`h.mean(dim=2)`).

Applied non-destructively (separate variable; the main loop continues with the
original deferred-residual tensors). **Risk to verify on cluster:** calling
`mhc_post_tilelang` at layers 40/41/42 adds 3 extra fused-kernel invocations per
forward; confirm the kernel is side-effect free when called mid-stack (it is a
pure combiner at loop end, so this should hold).

## Fix C — proper cross-attention (designed, not yet coded)

### Reference behavior (`inference/model.py`)

`DSparkAttention.forward(x, start_pos, main_x)`:
- **Prefill (start_pos==0):** `main_kv = kv_norm(wkv(main_x))`; write into the
  attention KV cache window; return x unchanged.
- **Decode (start_pos>0):** compute q/kv for the γ draft tokens; `kv =
  cat([kv_cache_window, block_kv])`; `sparse_attn(q, kv, attn_sink, topk_idxs)`.

So the projected context `main_x` (one vector per request, from
`mtp.0.main_proj`+`main_norm`) is consumed **only** as additional KV that every
draft stage attends to. `main_x` is the same for all 3 stages, but each stage
projects it through its *own* `attn.wkv` → a per-stage `main_kv`.

### Current INTERIM (in `forward_dspark_block`)

`main_x` is added to the anchor (position-0) draft embedding. This injects some
context but is NOT equivalent to cross-attention and will under-perform.

### Target implementation in vLLM

The draft uses vLLM's paged attention via `mtp_block.attn`, with the KV cache
and slot mappings built by `DSparkSpeculator`. To make the draft attend to the
main context:

1. **Reserve a context slot per request in the draft KV cache.** In
   `DSparkSpeculator._prepare_dspark_inputs` / `_build_draft_attn_metadata`,
   the draft block already declares `seq_lens = anchor_pos + γ + 1`, implying
   history positions `0..anchor_pos`. Repurpose the anchor position (or a
   dedicated slot) to hold the projected main context KV.
2. **Write per-stage `main_kv` into the draft KV cache** before the block
   attention runs. Two options:
   - (a) Add a one-token "prefill" pass per stage that runs
     `mtp_block.attn` on `main_x` (so the backend writes its KV via the slot
     mapping for the context slot). Cleanest reuse of existing kernels.
   - (b) Compute `main_kv = stage.mtp_block.attn.<wkv/kv_norm>(main_x)` directly
     and scatter into the KV cache tensor at the context slot. Lower-level;
     must match the backend's KV layout exactly.
3. **Have the γ draft tokens attend to [context slot + own block].** With (a),
   the existing `seq_lens`/block-table machinery already includes the context
   position; ensure `topk`/window covers it and `causal=False` keeps the block
   bidirectional.
4. **Remove the interim embedding addition** in `forward_dspark_block`.

### Files to touch for C

- `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` — KV cache slot for
  context, per-stage main_kv write, attn metadata.
- `vllm/models/deepseek_v4/nvidia/dspark.py` — drop the interim add; thread
  `main_x` into the per-stage attention path.

### Recommended cluster validation order

1. Build with A/B/D/E/F + interim C. Confirm it loads (assertion passes) and
   serves. Measure acceptance — expect low but possibly >0 (interim context).
2. Implement C option (a). Re-measure; target >3/5 acceptance.
3. Only then revisit CUDA graphs (Phase 3) and scheduling (Phase 4).
