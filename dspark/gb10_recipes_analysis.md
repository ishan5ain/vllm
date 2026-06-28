# GB10 DGX Spark Recipes Analysis — DSpark Implications

> **Date:** 2026-06-27
> **Sources:** `~/repos/spark-vllm-docker/recipes/deepseek-v4-flash.yaml` and
> `deepseek-v4-flash-chthonic.yaml`
> **Purpose:** Understand how the 2× DGX Spark cluster currently serves
> `deepseek-ai/DeepSeek-V4-Flash`, and identify what must change for DSpark.

## 0. Two Operating Modes

Both recipes run the same target model on the same 2× DGX Spark (GB10 / SM121)
cluster with TP=2 and achieve similar performance. They differ substantially in
*how*:

| Aspect | Standard Recipe | Chthonic Recipe |
|---|---|---|
| Container | `vllm-node` | `aidendle94/sparkrun-vllm-ds4-gb10:production-2.9` |
| Orchestration | Ray (`--distributed-executor-backend ray`) | `sparkrun` (custom launcher, no Ray) |
| Attention backend | Default (FlashInfer/FlashAttn) | `B12X_MLA_SPARSE` (b12x sparse MLA) |
| MoE backend | Default | `flashinfer_cutlass` |
| Linear backend | Default | `b12x` |
| Compilation | Standard | AOT compile (`VLLM_USE_AOT_COMPILE=1`, mega artifact) |
| Loading | `instanttensor` | Standard safetensors |
| CUDA graphs | Breakable disabled | FULL_AND_PIECEWISE cudagraph mode |
| Num spec tokens | 2 | 2 |
| Draft method | MTP | MTP (`draft_sample_method: greedy`) |
| MTP accept rate | Not specified | ~2.2–2.4 tokens accepted per step |

## 1. Target Recipe: Standard (vllm-node)

**DSpark only needs to work with ONE recipe — the simpler, easier one.**
The standard recipe (`deepseek-v4-flash.yaml`) is the clear choice:
- Default vLLM backends (FlashInfer/FlashAttn) — no b12x compatibility concerns
- Standard Ray executor — vLLM's mainline distributed path
- `instanttensor` model loading
- Only 3 env vars needed (vs 20+ for Chthonic)

The Chthonic b12x stack is a performance-optimized fork; DSpark can be ported to
it later if profiling shows benefit.

### 1.1 Current recipe uses `num_speculative_tokens: 2`

DSpark's γ=5 is 2.5× more speculative tokens. This has implications:

- **CUDA graph memory**: The speculative decode metadata, draft token buffers, and
  attention metadata scale with num_speculative_tokens. With fixed γ=5 (Tier 1), CUDA
  graphs remain compatible but capture size grows.
- **Acceptance behavior**: MTP currently accepts ~2.2-2.4 of 2 proposed tokens (~110%
  accept rate — at least one token almost always accepted). DSpark with 5 tokens will
  likely see lower per-position acceptance at positions 4-5, but higher total accepted
  tokens due to better draft quality.
- **Speculative config field**: When DSpark is deployed, `num_speculative_tokens`
  must be set to 5 in the `--speculative-config` JSON. The existing `DeepSeekV4MTP`
  code handles variable `num_nextn_predict_layers`; DSpark will need its own handling.

### 1.2 Standard recipe backend stack (the DSpark target)

```
Container: vllm-node
Orchestration: Ray (--distributed-executor-backend ray)
Attention: Default (FlashInfer/FlashAttn)
MoE: Default (flashinfer_cutlass)
Linear: Default
Compilation: Standard (DG_JIT_USE_NVRTC=0)
CUDA Graphs: Breakable disabled (VLLM_USE_BREAKABLE_CUDAGRAPH=0)
```

**DSpark compatibility assessment (standard recipe):**

| Component | DSpark Needs | Risk |
|---|---|---|
| MLA attention | Bidirectional (is_causal=False) | **Low** — FlashInfer supports non-causal attention |
| MoE (256 experts) | Same expert structure in draft | Low — same backend |
| mHC | Draft layers use hyperconnections | Low — Python mHC path available |
| hc_head | Hypercompressed LM head | Low — same kernel as existing MTP |
| CUDA graphs | γ=5 draft block forward | Low for Tier 1 (fixed γ) |
| KV cache | FP8, draft layers need KV | Low — same KV format as target |

### 1.3 Memory budget

GB10 has 128 GB **unified** memory (CPU+GPU shared via LPDDR5X).

**Current usage (Chthonic recipe):**
```
gpu_memory_utilization: 0.8  → ~102 GB for model + KV cache
max_num_seqs: 4, max_num_batched_tokens: 8192
max_model_len: 500000 (but KV cache allocation not linear — block_size: 256)
```

**DSpark adds:** ~8 GB for the draft model (3 MTP layers × MoE 256 experts, Markov W₁W₂,
confidence head). This is ~8% of total memory.

**Impact:** If the base model already uses ~90-100 GB, the draft model may not fit
without reducing `gpu_memory_utilization`, `max_num_seqs`, or `max_model_len`.
On the Chthonic stack specifically, the b12x optimizations trade memory for speed
— memory pressure may already be higher than the standard recipe.

**Mitigation options:**
1. Expert offloading for draft model MoE layers (expensive at inference)
2. FP8 quant on draft model layers (checkpoint already uses fp4/fp8 for experts)
3. Reduce max_model_len or max_num_seqs
4. Verify with `nvidia-smi` and add explicit memory profiling to Phase 1

### 1.4 No NVLink — TP communication over RoCE

Both recipes run TP=2 across two DGX Spark nodes connected via ConnectX-7
200GbE (RoCE v2). Key constraints:

- **ASYNC_SCHED=0** is critical (Chthonic recipe explicitly warns: "desyncs TP
  collectives on 2-node")
- **NCCL_CROSS_NIC=1** — use both ConnectX-7 ports
- **No PCIe allreduce env vars** — Chthonic recipe unsets them (they crash on 2-node)
- Dynamic RoCE GID probing (GID index re-enumerates across reboots)

**DSpark impact:**
- **Context extraction from layers 40,41,42**: These hidden states are on each TP rank.
  The context projection (`fc: Linear[12288→4096]`) runs independently per rank — no
  cross-node communication needed.
- **Draft forward pass**: Each rank runs its draft model shard. Draft attention KV
  would need all-reduce if using TP>1 for the draft model, but DSpark draft model
  is small enough to run TP=1.
- **Draft model TP**: If the draft model uses TP=1 (likely, given ~8 GB size), all
  ranks run identical computation — just don't all-reduce. Need to verify
  `DraftModelProposer._raise_if_draft_tp_mismatch()` behavior.
- **Verification**: Standard speculative decode verification runs through the target
  model, which is already TP=2 via RoCE. No change needed.

### 1.5 Ray executor

The standard recipe uses `--distributed-executor-backend ray`. DSpark model
loading and forward passes must work with Ray's distributed setup. vLLM's
model loading infrastructure (`get_model()`) handles this transparently for
standard model classes — no DSpark-specific changes expected.

## 2. Plan Revisions

### 2.1 Focus on the standard recipe

Since DSpark only needs to work with one recipe, target the simpler standard
recipe (`vllm-node` container, Ray executor, default backends). This removes
all b12x-specific concerns (bidirectional attention compatibility, mHC kernel
shape assumptions, AOT mega-artifact, sparkrun orchestration).

The Chthonic b12x stack can be revisited later if profiling shows DSpark
performance is bottlenecked on the standard backend.

### 2.2 Add prerequisite tasks to Phase 1

Before writing `dspark.py`, these checks remain important:

1. **Memory budget measurement** (prerequisite):
   - The standard recipe uses `gpu_memory_utilization: 0.8` on 128 GB GB10
   - With the base V4-Flash model loaded, check available memory via `nvidia-smi`
   - The DSpark draft model adds ~8 GB
   - If tight, reduce `max_num_seqs` (currently 4) or `max_model_len` (currently 500K)

2. **Draft model TP strategy** (decision):
   - Draft model runs TP=1 on each rank (identical computation, no all-reduce)
   - Need to verify `DraftModelProposer._raise_if_draft_tp_mismatch()` handles
     draft TP=1 with target TP=2 (or add override)

### 2.3 Approach: Separate class confirmed

The current plan (new `DeepSeekV4DSpark` parallel class) is the right choice.
No need to work around b12x custom kernels — the standard recipe uses default
backends where bidirectional attention, the mHC Python path, and standard
linear ops all work without special handling.

### 2.4 Updated Phase 1 checklist

- [ ] **Prereq: Memory budget measurement** — measure available memory on GB10
  with base V4-Flash loaded via standard recipe
- [ ] **Prereq: Draft model TP strategy** — verify TP=1 draft + TP=2 target setup
- [ ] Create `DeepSeekV4DSpark` class in `vllm/models/deepseek_v4/nvidia/dspark.py`
- [ ] Implement `load_weights()` for all 3 MTP layers + markov + confidence heads
- [ ] Load `dspark_*` config fields from checkpoint
- [ ] Register DSpark model in vLLM model registry
- [ ] Add DSpark to spec decode proposer lookup (`method: "dspark"`)
- [ ] Add DSpark-specific config fields (`dspark_block_size`, etc.)

## 3. Infrastructure Notes

### 3.1 Model download

The checkpoint is being downloaded via `~/repos/spark-vllm-docker/hf-download.sh`:
```bash
./hf-download.sh deepseek-ai/DeepSeek-V4-Flash-DSpark --copy-to <worker-ip>
```
This uses `uvx hf download` and optionally `rsync`s the model to the worker node.
The model lands in `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/`.

### 3.2 Running on the cluster

Once the checkpoint is downloaded, DSpark will be served via the same recipe
system, with modifications:
- Updated `--speculative-config` with `method: "dspark"` and `num_speculative_tokens: 5`
- Possible adjustments to `gpu_memory_utilization` to accommodate the draft model
- If bidirectional attention requires a different backend, add `ATTN_BACKEND` override
  for draft layers only

## 4. Risks Noted (added/updated from original plan)

| Risk | New? | Mitigation |
|---|---|---|
| B12X_MLA_SPARSE doesn't support `is_causal=False` | **NEW** | Fall back to FlashInfer or Triton attention for draft layers only |
| b12x mHC incompatible with draft layer dimensions | **NEW** | Disable b12x mHC for draft layers (use Python fallback) |
| Draft model ~8GB doesn't fit alongside base model on 128GB GB10 | **NEW** | Memory profiling prerequisite; reduce max_num_seqs or model_len |
| Draft TP=1 / target TP=2 mismatch rejected by proposer | **NEW** | Check `_raise_if_draft_tp_mismatch()` — may need override |
| AOT compile time increases with DSpark compute graph | **NEW** | Acceptable for cold boot; persistent JIT cache mitigates |
| DSpark MTP was net throughput loss on single-stream (per Entrpi/ds4-on-spark) | Existing | Benchmark Phase 2 before deploying to production |
| num_speculative_tokens: 5 vs current 2 — CUDA graph memory scaling | **NEW** | Fixed γ=5 is compatible; measure actual memory increase |
