# DSpark Phase 3 — CUDA Graphs Plan

> **Date:** 2026-06-28
> **Status:** PLANNING
> **Depends on:** f0b84ed07 (cooperative_topk Blackwell fix)

## Goal

Move DSpark from O0 eager mode (~15 tok/s decode) to O1+ with CUDA graphs
(target: 30–45 tok/s). The DFlash speculator already supports full CUDA graphs
on the same hardware; DSpark needs equivalent support.

## Prerequisites (already in place)

- `f0b84ed07`: cooperative_topk disabled on Blackwell (≥sm_100), falls back to
  persistent_topk. This was the blocker for both graph capture (larger batch
  sizes trigger cooperative_topk) and FlashInfer autotune warmup.

## Step 1: Enable O1 and verify graph capture (recipe change)

O1 provides:
- `cudagraph_mode: PIECEWISE` — piecewise CUDA graphs (attention runs eagerly,
  FFN + other ops captured)
- `enable_flashinfer_autotune: True` — should no longer crash since cooperative_topk
  is bypassed via persistent_topk fallback
- Norm/act quant fusion passes

**Change in recipe:**
```yaml
# Remove -O0, replace with -O1
command: |
    vllm serve ... \
        -O1 \
        ...
```

**Test:** Run recipe, verify:
1. FlashInfer autotune warmup completes without `cooperative_topk` crash
2. "Skipping CUDA graph capture" message is NOT shown
3. Model starts serving

If autotune still crashes: may need to keep `enable_flashinfer_autotune: False`
while enabling CUDA graphs via `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`.

## Step 2: Verify persistent_topk correctness under graph capture

The `persistent_topk` path in `sparse_attn_indexer.py` (line 493) uses a
persistent kernel that may have different numerical behavior. During graph
capture with larger batch sizes (>32 rows), the `cooperative_topk` path is
already bypassed (condition `num_rows <= 32` fails). So graph capture was
likely already using persistent_topk, and our Blackwell guard just extends
this to all batch sizes.

**Risk:** Low. The persistent kernel is already exercised on Hopper at larger
batch sizes. No new code path.

## Step 3: Build DSparkCudaGraphManager

Model after DFlash (`vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py`).

### New file: `vllm/v1/worker/gpu/spec_decode/dspark/cudagraph.py`

```python
class DSparkCudaGraphManager(CudaGraphManager):
    """DSpark CudaGraphManager for block-generation forward.

    DSpark produces all γ draft tokens in one forward pass. The draft batch
    has exactly `num_reqs * (1 + γ)` tokens (anchor + γ draft positions),
    which is uniform per batch — ideal for CUDA graphs.
    """

    def __init__(self, *args, causal: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal = causal  # False for bidirectional draft attention

    def capture(self, forward_fn, input_buffers, block_tables,
                attn_groups, kv_cache_config, max_model_len,
                progress_bar_desc="Capturing DSpark CUDA graphs"):
        # Build dummy attention metadata with causal=False
        def create_forward_fn(desc, warmup):
            # Same pattern as DFlash: build dummy inputs, metadata, slot mappings
            ...
        super().capture(create_forward_fn, progress_bar_desc)
```

Helper: `_prepare_dspark_inputs_to_capture()` — builds attention metadata with
`causal=False` and uniform batch layout `[num_reqs, 1+γ]`.

**Files to create:**
- `vllm/v1/worker/gpu/spec_decode/dspark/cudagraph.py` (~80 lines)

**Files to modify:**
- `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` — wire `init_cudagraph_manager`

## Step 4: Wire CUDA graph support into DSparkSpeculator

### 4a. `init_cudagraph_manager()` (line 74)

Replace stub with real initialization (pattern from DFlash):
```python
def init_cudagraph_manager(self, cudagraph_mode):
    if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
        cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    elif cudagraph_mode != CUDAGraphMode.NONE:
        cudagraph_mode = CUDAGraphMode.NONE
    self.query_cudagraph_manager = DSparkCudaGraphManager(
        vllm_config=self.vllm_config,
        device=self.device,
        cudagraph_mode=cudagraph_mode,
        decode_query_len=self.num_query_per_req,  # 1 + γ
        causal=False,
    )
```

### 4b. `capture()` method

Replace stub with real capture (pattern from DFlash):
```python
def capture(self, attn_states=None):
    if self.query_cudagraph_manager is None:
        return
    self.query_cudagraph_manager.capture(
        forward_fn=self._cudagraph_forward,
        input_buffers=self.input_buffers,
        block_tables=self.block_tables,
        attn_groups=self.attn_groups,
        kv_cache_config=self.kv_cache_config,
        max_model_len=self.max_model_len,
    )
```

### 4c. `_cudagraph_forward()` — new method

Wraps `forward_dspark_block()` with CUDA graph-compatible input handling:
```python
def _cudagraph_forward(self, num_reqs, num_tokens, attn_metadata,
                       slot_mappings, num_tokens_across_dp, cg_mode):
    # Build input batch from buffers
    # Call forward_dspark_block with graph-compatible inputs
    ...
```

### 4d. `propose()` — use resolved CUDA graph mode

Change line 385 from:
```python
cudagraph_runtime_mode=CUDAGraphMode.NONE,
```
To:
```python
cudagraph_runtime_mode=self.cudagraph_dispatcher.dispatch(...),
```

### 4e. Input buffer preparation

`DSparkSpeculator` needs `InputBuffers` and `BlockTables` for graph capture
dummy input construction (same pattern as DFlash at speculator.py:204-216).

**Files to modify:**
- `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` (~100 lines changed)

## Step 5: Update `num_query_per_req`

Currently `self.num_query_per_req = self.num_speculative_steps` (= γ = 5).
For CUDA graphs, this should be `1 + self.num_speculative_steps` (= 6)
to include the anchor token position, matching DFlash's pattern of
`1 + num_speculative_steps`.

Check: the draft batch size in `propose()` is `num_reqs * (1 + γ)`.
CUDA graph capture needs `decode_query_len` to match this per-request
token count for uniform batch sizing.

## Step 6: Benchmark

After all wiring:
1. Remove `-O0` from recipe, use `-O1` (or `-O2` if graphs support FULL_AND_PIECEWISE)
2. Run benchmark comparing O0 vs O1:
   ```bash
   python vllm/benchmarks/serve.py \
     --backend openai --base-url http://<ip>:8000 \
     --model deepseek-ai/DeepSeek-V4-Flash-DSpark \
     --dataset-name sharegpt --num-prompts 100
   ```
3. Expected improvement: 15 tok/s → 30–45 tok/s (2–3× from CUDA graphs)
4. Compare acceptance rate: should be unchanged (same computation, different execution)

## Fallback Plan

If Step 1 (O1 enable) fails due to FlashInfer autotune still crashing:
- Keep O0-level kernel settings but enable only CUDA graphs:
  ```bash
  --compilation-config '{"cudagraph_mode":"PIECEWISE","custom_ops":[]}'
  ```
- This skips autotune + optimizations but gets graph capture benefits

If Steps 3–4 (DSparkCudaGraphManager) are too complex for a single session:
- Phase 3a: enable main model CUDA graphs (O1) — immediate ~2× speedup from
  target model graph capture alone
- Phase 3b: add DSpark speculator graphs — additional speedup from draft
  forward graph capture

## Estimated Effort

| Step | Effort | Risk |
|---|---|---|
| Step 1 (O1 enable) | Recipe change only | Low — if autotune crashes, use fallback |
| Step 2 (persistent_topk verify) | No code | Low |
| Step 3 (CudaGraphManager) | ~80 lines new file | Medium — follows DFlash pattern closely |
| Step 4 (wire into speculator) | ~100 lines modified | Medium — requires careful input buffer handling |
| Step 5 (num_query_per_req) | 3 lines | Low |
| Step 6 (benchmark) | bash command | Low |
| **Total** | **~180 lines, 2 new/modified files** | **Medium** |
