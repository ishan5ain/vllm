# DSpark vLLM Integration — Progress & State

> **Date:** 2026-06-28
> **Branch:** `dspark-research` (fork: `github.com/ishan5ain/vllm`)
> **Sessions:** research → implementation → review → cluster testing → serving

## Implementation Status

```
Phase 0: Prerequisites    ████████████ DONE
Phase 1: Model Loading    ████████████ DONE (4 cluster-test bugs fixed)
Phase 2: Draft Generation ████████████ DONE (serving on 2× GB10 cluster)
Phase 3: CUDA Graphs      ░░░░░░░░░░░░ DEFERRED (O0 eager mode for now)
Phase 4: Scheduling       ░░░░░░░░░░░░ NOT STARTED
Phase 5: STS Calibration  ░░░░░░░░░░░░ CODE EXISTS, NOT INTEGRATED
```

## Current Status (2026-06-28)

**DSpark is serving on 2× DGX Spark GB10 cluster.** All weight loading bugs resolved,
model initializes, and inference produces tokens. Benchmarking of acceptance rate and
draft quality is the next step.

### Benchmark Results (eager mode, O0)

| Test | t/s | Notes |
|---|---|---|
| `tg128` (decode, no context) | 15.25 ± 0.07 | Decode tokens/sec |
| `ctx_pp @ d4096` (prefill) | 13359 ± 5102 tok/s | High variance prefill |
| `ctx_tg @ d4096` (decode w/ context) | 14.97 ± 0.19 | Similar to clean decode |
| `tg128 @ d4096` (decode w/ context) | 14.94 ± 0.06 | Consistent decode speed |

For reference: Chthonic b12x + MTP baseline achieves ~52 tok/s decode on the same
hardware. The ~3.5× gap is attributable to: (a) no CUDA graphs (O0 eager mode),
(b) standard kernels (not B12X), (c) DSpark draft model overhead (~8 GiB leaving
less memory for KV cache).

### Memory Footprint

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
optimization_level: 0  # -O0 (eager mode, no FlashInfer autotune)
```

## Cluster-Test Bugs Fixed

### Weight Loading (4 fixes, commits `c646e6965`–`90ead3aea`)

| Bug | Symptom | Fix |
|---|---|---|
| `model.` prefix mismatch | `KeyError` on all param lookups — inner model `named_parameters()` doesn't include `model.` prefix | Changed mtp→layer prefix rewrite from `model.layers.{idx}.` to `layers.{idx}.` |
| `.norm.weight` remap too broad | Corrupted `kv_norm`, `q_norm`, `attn_norm`, `ffn_norm` names | Guard: exclude names containing these patterns |
| `main_norm`/`main_proj` missing | `KeyError` — MTP weights not in DSpark model | Added to weight_name list + continue guard |
| `attn_sink` not in params | `KeyError` — may be buffer, not parameter | Added `name not in params_dict` guard |
| Guard placement wrong | Guard after attn_sink/experts checks | Moved to top of outer else branch |

### Integration (3 fixes, pre-cluster-test)

| Bug | Symptom | Fix |
|---|---|---|
| `num_speculative_tokens` without model | ValidationError | Added `"dspark"` to MTP-like auto-set path |
| `NotImplementedError` for `"dspark"` | Auto-detection chain missing | Added model_type→method auto-detection |
| `Unknown speculative decoding method` | model_runner's drafter chain missing | Added DSparkProposer + routing |

### Runtime (4 fixes, during cluster test)

| Bug | Symptom | Fix | Commit |
|---|---|---|---|
| EAGLE3 interface required | `Model does not support EAGLE3 interface` | Removed `use_aux_hidden_state_outputs` for DSpark (uses custom `_dspark_context_buffer`) | `cbaa4ad2a` |
| 2D/3D hidden_state IndexError | `[:, 0, :]` on 2D tensor in context capture | Check `hidden_states.dim()` before indexing | `90ead3aea` |
| cooperative_topk crash (warmup) | `invalid argument` in FlashInfer autotune | Added `-O0` to recipe (disables autotune + CUDA graphs) | recipe |
| cooperative_topk crash (inference) | Same error during decode with small batches | Disabled cooperative_topk on ≥sm_100 (Blackwell); falls back to persistent_topk | `f0b84ed07` |

## Architecture (files changed vs upstream)

```
vllm/config/speculative.py                 DSpark detection + method routing + model auto-set
vllm/model_executor/models/registry.py     DeepSeekV4DSparkModel registration
vllm/models/deepseek_v4/__init__.py        Re-export (NVIDIA only)
vllm/models/deepseek_v4/nvidia/model.py    Target context capture (layers 40,41,42) + 2D/3D fix
vllm/models/deepseek_v4/nvidia/dspark.py   Draft model (3 classes, ~870 lines, prefix fix)
vllm/v1/worker/gpu/model_runner.py         Context plumbing + DSparkProposer routing + EAGLE3 bypass
vllm/v1/worker/gpu/spec_decode/__init__.py Speculator routing
vllm/v1/worker/gpu/spec_decode/dspark/     DSparkSpeculator (KV cache, attention, propose)
vllm/v1/spec_decode/dspark_proposer.py     DSparkProposer (SpecDecodeBaseProposer wrapper)
vllm/model_executor/layers/sparse_attn_indexer.py  cooperative_topk Blackwell fix
```

## Class Structure (dspark.py)

```
DeepSeekV4DSparkLayer    — Per-layer backbone (enorm/hnorm, e_proj/h_proj, mtp_block)
                           Output layer (idx=45) has hc_head + shared_head
DSparkInnerModel          — All logic: 3 layers, Markov head (W₁/W₂), confidence head,
                           fc context projection, forward_dspark_block, load_weights
DeepSeekV4DSparkModel     — Wrapper for vLLM compat. Exposes self.model for
                           load_eagle_model (embedding sharing, topk_indices_buffer)
```

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Target recipe | Standard (`vllm-node`, Ray, default backends) | Simpler than Chthonic b12x |
| Draft TP | TP=2 (matches target) | Avoids proposer TP mismatch checks |
| Detection | `hasattr(hf_config, "dspark_block_size")` | Differentiates DSpark from standard MTP |
| Speculator | Custom `DSparkSpeculator` (not MTPSpeculator) | DSpark produces all γ tokens in one pass |
| Proposer | `DSparkProposer(SpecDecodeBaseProposer)` | Satisfies model_runner's drafter chain |
| Model path | `DSparkProposer._get_model()` loads via registry | Separate from speculator's `load_eagle_model` |
| Attention | `causal=False` in metadata | Bidirectional within γ block |
| CUDA graphs | Deferred (eager mode, O0) | cooperative_topk + memory constraints |
| Optimization level | O0 | Bypasses FlashInfer autotune + CUDA graph crashes on SM120a |
| Memory | `gpu_memory_utilization=0.85`, `max_num_seqs=1` | Conservative for DSpark's ~8 GB draft model |

## Known Gaps

| Gap | Severity | Notes |
|---|---|---|
| Acceptance rate unmeasured | **HIGH** | Need to benchmark DSpark acceptance vs MTP baseline (~2.2/2) |
| Markov head correctness | **HIGH** | W₁W₂ sampling not verified with real prompts |
| Confidence head unused | High | Confidence scores computed but scheduling (Phase 4) not implemented |
| No CUDA graphs | Medium | ~3.5× slower than Chthonic baseline; O0 eager mode |
| `fc` weight source unknown | Medium | Context projection may stay random — verify checkpoint key |
| Bidirectional attention untested | Medium | `causal=False` set but not yet validated at runtime |
| Draft KV cache allocation | Medium | `set_attn()` wired; verify correctness |
| cooperative_topk disabled on Blackwell | Medium | Fallback to persistent_topk; may be slower on sm_100 (B200) |
| Subagent async broken | Low | Only affects dev workflow, not runtime |

## Build & Run Commands

```bash
# Build (from ~/repos/spark-vllm-docker)
./build-and-copy.sh \
  --vllm-ref dspark-research \
  --vllm-repo https://github.com/ishan5ain/vllm.git \
  --rebuild-vllm \
  --copy-to 192.168.0.183

# Tag (after build)
docker tag vllm-node:latest vllm-node:dspark
ssh 192.168.0.183 "docker tag vllm-node:latest vllm-node:dspark"

# Run
./run-recipe.sh deepseek-v4-flash-dspark --no-ray

# Stop
./launch-cluster.sh stop
```

## Next Steps (Priority Order)

1. **Benchmark acceptance rate** — compare DSpark's acceptance rate vs standard MTP (~2.2/2). This is the key metric — if DSpark achieves >4/5 acceptance, the spec overhead is worth it despite slower raw decode.
2. **Verify Markov head** — send known prompts and compare draft tokens against reference implementation; confirm W₁W₂ sampling is correct.
3. **Profile draft overhead** — measure time spent in DSpark forward vs target model forward to quantify the spec decode cost.
4. **Phase 3: Re-enable CUDA graphs** — once cooperative_topk is fixed upstream or B12X is available, move from O0 to O2 with CUDA graphs. Expected improvement: +2-3× decode speed.
5. **Phase 4: Tier 2 scheduling** — integrate confidence head with batch-averaged truncation for adaptive draft lengths.
6. **Phase 5: STS calibration** — run `sts_calibration.py` on held-out set to calibrate acceptance thresholds.
