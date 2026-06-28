# DSpark vLLM Integration — Progress & State

> **Date:** 2026-06-28
> **Branch:** `dspark-research` (fork: `github.com/ishan5ain/vllm`)
> **Latest commit:** `d4d66dfee` — hc_head fix (3D backbone, pending cluster test)
> **Sessions:** research → implementation → review → cluster testing → serving → debugging acceptance

## Implementation Status

```
Phase 0: Prerequisites    ████████████ DONE
Phase 1: Model Loading    ████████████ DONE (4 cluster-test bugs fixed)
Phase 2: Draft Generation ████████████ DONE (serving, acceptance bug found)
Phase 3: CUDA Graphs      ██████████░░ IN PROGRESS (O1 enabled, DSpark graphs pending)
Phase 4: Scheduling       ░░░░░░░░░░░░ NOT STARTED
Phase 5: STS Calibration  ░░░░░░░░░░░░ CODE EXISTS, NOT INTEGRATED
```

## Current Status (2026-06-28)

**DSpark is serving on 2× DGX Spark GB10 cluster at O1 with PIECEWISE CUDA graphs.**
A critical draft acceptance bug was found: **0% acceptance across all 5 positions**.
The hc_head kernel was receiving wrong input — the backbone produced 2D hidden states
which were replicated into 4 identical copies instead of 4 genuinely unique mhc-encoded
streams. Fix committed (`d4d66dfee`), **pending cluster test**.

### Acceptance Rate Bug (found 2026-06-28)

```
SpecDecoding metrics: Mean acceptance length: 1.00, Accepted: 0 tokens,
Drafted: 670 tokens, Per-position acceptance rate: 0.000 across all 5 positions
```

Root cause: `forward_dspark_block` ran the backbone with 2D input `[T, D]` (hc_mult=1),
then called `.repeat(1, hc_mult, 1)` to create 4 identical copies for the hc_head kernel.
The hc_head kernel's 4×4 mhc mixing matrix expects genuinely unique streams produced
by the decoder layer's mhc encoding. With identical copies, the mixing produced wrong
logits → all draft tokens rejected.

Fix: expand draft embeddings to 3D `[T, hc_mult, D]` (hc_mult=4) before the backbone loop.
The decoder layer's `mhc_pre_tilelang` / `mhc_fused_post_pre_tilelang` kernels then
create proper 4-stream mhc encoding, matching what the target model's hc_head expects.

| Commit | Description |
|---|---|
| `55451307d` | First attempt: pass hc_mult=1 (insufficient — 1-stream vs 4-stream fundamentally different) |
| `d4d66dfee` | Proper fix: 3D backbone input → genuine 4-stream mhc encoding |

### Benchmark Results (O1, PIECEWISE CUDA graphs, before hc_head fix)

| Test | t/s | Notes |
|---|---|---|
| `tg128` (decode) | 6.9 | Lower than O0 due to draft overhead without acceptance |
| Drafted throughput | 34.5 tok/s | 670 tokens drafted over test window |
| Accepted throughput | 0.00 tok/s | All draft tokens rejected |

Expected after fix: decode speed should improve when draft tokens are accepted
(effective speed = decode_t/s × (1 + acceptance_rate × acceptance_length)).

### O1 Upgrade (2026-06-28)

Successfully moved from O0 to O1:
- `cudagraph_mode: PIECEWISE` — attention runs eagerly, FFN captured in piecewise graphs
- `enable_flashinfer_autotune: False` — autotune skipped (would crash cooperative_topk)
- Recipe uses `-O1` with `--compilation-config '{"cudagraph_mode":"PIECEWISE","custom_ops":[]}'`

### Memory Footprint (O1)

| Component | Size |
|---|---|
| Target model (DeepSeek V4 Flash, TP=2) | ~71 GiB |
| Draft model (DSpark 3-layer, TP=2) | ~8.5 GiB |
| KV cache (gpu_memory_utilization=0.85) | ~9.7 GiB (620K tokens) |
| **Total** | **~89 GiB** / 96 GiB unified memory |

### Recipe Configuration

```yaml
gpu_memory_utilization: 0.85
max_model_len: 256000
max_num_seqs: 1
num_speculative_tokens: 5
# O1 via compilation-config (not -O0):
# cudagraph_mode: PIECEWISE, custom_ops: []
```

## Cluster-Test Bugs Fixed

### Weight Loading (4 fixes, commits `c646e6965`–`90ead3aea`)

| Bug | Symptom | Fix |
|---|---|---|
| `model.` prefix mismatch | `KeyError` on all param lookups | Changed mtp→layer prefix from `model.layers.{idx}.` to `layers.{idx}.` |
| `.norm.weight` remap too broad | Corrupted norm names | Guard: exclude attn_norm, ffn_norm, kv_norm, q_norm |
| `main_norm`/`main_proj` missing | `KeyError` — MTP weights not in DSpark | Added to weight_name list + continue guard |
| `attn_sink` not in params | `KeyError` — may be buffer | Added `name not in params_dict` guard |
| Guard placement wrong | Guard after attn_sink/experts checks | Moved to top of outer else branch |

### Integration (3 fixes, pre-cluster-test)

| Bug | Symptom | Fix |
|---|---|---|
| `num_speculative_tokens` without model | ValidationError | Added `"dspark"` to MTP-like auto-set path |
| `NotImplementedError` for `"dspark"` | Auto-detection chain missing | Added model_type→method auto-detection |
| `Unknown speculative decoding method` | model_runner drafter chain missing | Added DSparkProposer + routing |

### Runtime & Correctness (5 fixes)

| Bug | Symptom | Fix | Commit |
|---|---|---|---|
| EAGLE3 interface required | `Model does not support EAGLE3` | Removed `use_aux_hidden_state_outputs` | `cbaa4ad2a` |
| 2D/3D hidden_state IndexError | `[:, 0, :]` on 2D tensor | Check `hidden_states.dim()` | `90ead3aea` |
| cooperative_topk crash (warmup) | `invalid argument` in autotune | Added cooperative_topk Blackwell guard | `f0b84ed07` |
| cooperative_topk crash (inference) | Same error at decode | Disabled on ≥sm_100, fallback persistent_topk | `f0b84ed07` |
| **0% draft acceptance** | hc_head receives 1 stream × 4 copies | **3D backbone input for proper 4-stream mhc encoding** | **`d4d66dfee`** |

## Architecture (files changed vs upstream)

```
vllm/config/speculative.py                 DSpark detection + method routing
vllm/model_executor/models/registry.py     DeepSeekV4DSparkModel registration
vllm/models/deepseek_v4/__init__.py        Re-export (NVIDIA only)
vllm/models/deepseek_v4/nvidia/model.py    Target context capture + 2D/3D fix
vllm/models/deepseek_v4/nvidia/dspark.py   Draft model + forward_dspark_block + hc_head fix
vllm/v1/worker/gpu/model_runner.py         Context plumbing + EAGLE3 bypass
vllm/v1/worker/gpu/spec_decode/__init__.py Speculator routing
vllm/v1/worker/gpu/spec_decode/dspark/     DSparkSpeculator (KV cache, attention, propose)
vllm/v1/spec_decode/dspark_proposer.py     DSparkProposer (SpecDecodeBaseProposer)
vllm/model_executor/layers/sparse_attn_indexer.py  cooperative_topk Blackwell fix
```

## Known Gaps

| Gap | Severity | Notes |
|---|---|---|
| Acceptance rate unverified | **HIGH** | hc_head fix pending cluster test; expected to resolve 0% acceptance |
| Markov head correctness | **HIGH** | Verified against paper — correct; confounded by hc_head bug |
| Confidence head unused | High | Scores computed but Phase 4 scheduling not implemented |
| No DSpark CUDA graphs | Medium | Main model has PIECEWISE graphs; speculator runs eagerly |
| `fc` weight source unknown | Medium | Context projection may stay random |
| Bidirectional attention | Medium | `causal=False` set, not validated at runtime |
| Draft KV cache | Medium | `set_attn()` wired; verify correctness |
| cooperative_topk on Blackwell | Medium | Fallback to persistent_topk; slower on sm_100 |
| O1 memory tight | Medium | 0.85 util needed; leaves only 9.7 GiB KV cache |

## Build & Run Commands

```bash
# Build
cd ~/repos/spark-vllm-docker
./build-and-copy.sh --vllm-ref dspark-research \
  --vllm-repo https://github.com/ishan5ain/vllm.git \
  --rebuild-vllm --copy-to 192.168.0.183
docker tag vllm-node:latest vllm-node:dspark
ssh 192.168.0.183 "docker tag vllm-node:latest vllm-node:dspark"

# Run
./run-recipe.sh deepseek-v4-flash-dspark --no-ray

# Stop
./launch-cluster.sh stop
```

## Next Steps (Priority Order)

1. **Cluster test hc_head fix** (`d4d66dfee`) — expected to resolve 0% acceptance
2. **Benchmark acceptance rate** — compare vs MTP baseline (~2.2/2); target: >3/5
3. **Verify Markov head end-to-end** — confirm W₁W₂ sampling produces correct tokens
4. **Phase 3b: DSpark CUDA graphs** — build `DSparkCudaGraphManager` (see `phase3_cudagraph_plan.md`)
5. **Phase 4: Confidence-based scheduling** — integrate confidence head
6. **Phase 5: STS calibration** — run `dspark/sts_calibration.py`
