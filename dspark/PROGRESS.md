# DSpark vLLM Integration — Progress & State

> **Date:** 2026-06-28
> **Branch:** `dspark-research` (fork: `github.com/ishan5ain/vllm`)
> **Sessions:** research → implementation → review → cluster testing

## Implementation Status

```
Phase 0: Prerequisites    ████████████ DONE
Phase 1: Model Loading    ████████████ DONE (weight loading bugs fixed iteratively on cluster)
Phase 2: Draft Generation ████████████ DONE (code complete, cluster test in progress)
Phase 3: CUDA Graphs      ░░░░░░░░░░░░ DEFERRED (eager mode for now)
Phase 4: Scheduling       ░░░░░░░░░░░░ NOT STARTED
Phase 5: STS Calibration  ░░░░░░░░░░░░ CODE EXISTS, NOT INTEGRATED
```

## Current Cluster Test Status (2026-06-28)

Model loads but **weight loading fails** at init. The `DSparkProposer` path loads the
draft model via `SpecDecodeBaseProposer._get_model()` → `get_model()` → which calls
`DeepSeekV4DSparkModel.load_weights()`.

### Weight loading bugs found and fixed on cluster

| Bug | Symptom | Fix |
|---|---|---|
| `model.` prefix mismatch | `KeyError` on all param lookups — inner model has `prefix="model"` | Restored `model.layers.X.` prefix in `_rewrite_spec_layer_name` |
| `.norm.weight` remap too broad | Corrupted `kv_norm`, `q_norm`, `attn_norm`, `ffn_norm` names → silently skipped | Guard: exclude names containing `attn_norm`, `ffn_norm`, `kv_norm`, `q_norm` |
| `main_norm`/`main_proj` missing params | `KeyError` — these MTP weights don't exist in DSpark model | Added to `spec_layer_weight_names` + `continue` guard |
| `attn_sink` not in params | `KeyError` — `attn_sink` might be a buffer, not parameter | Added `name not in params_dict` guard before `attn_sink` handler |
| Guard placement wrong | Guard was after `attn_sink`/experts checks, before shared_experts only | Moved guard to top of outer `else` branch |

**Latest commit:** `f1cdf686f` — guard and attn_sink fixes. Pending rebuild and cluster test.

### Integration bugs found and fixed

| Bug | Symptom | Fix |
|---|---|---|
| `num_speculative_tokens` without model | ValidationError | Added `"dspark"` to MTP-like model auto-set path |
| `NotImplementedError` for `"dspark"` | Auto-detection chain missing `deepseek_dspark` | Added model_type→method auto-detection |
| `Unknown speculative decoding method` | model_runner's drafter chain missing DSpark | Added `DSparkProposer` + routing + isinstance checks |

## Architecture (files changed vs upstream)

```
vllm/config/speculative.py                 DSpark detection + method routing + model auto-set
vllm/model_executor/models/registry.py     DeepSeekV4DSparkModel registration
vllm/models/deepseek_v4/__init__.py        Re-export (NVIDIA only)
vllm/models/deepseek_v4/nvidia/model.py    Target context capture (layers 40,41,42)
vllm/models/deepseek_v4/nvidia/dspark.py   Draft model (3 classes, ~870 lines)
vllm/v1/worker/gpu/model_runner.py         Context plumbing + DSparkProposer routing
vllm/v1/worker/gpu/spec_decode/__init__.py Speculator routing
vllm/v1/worker/gpu/spec_decode/dspark/     DSparkSpeculator (KV cache, attention, propose)
vllm/v1/spec_decode/dspark_proposer.py     DSparkProposer (SpecDecodeBaseProposer wrapper)
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
| CUDA graphs | Deferred (eager mode) | Reduce scope for initial integration |
| Memory | `gpu_memory_utilization=0.75`, `max_num_seqs=3` | Conservative for DSpark's ~8 GB draft model |

## Known Gaps

| Gap | Severity | Notes |
|---|---|---|
| Weight loading not fully verified | **HIGH** | Pending cluster test with latest fix (attn_sink guard) |
| `fc` weight source unknown | Medium | Context projection may stay random — verify checkpoint key |
| Bidirectional attention untested | Medium | `causal=False` set in metadata but not yet validated at runtime |
| Draft KV cache allocation | Medium | `set_attn()` wired; verify at runtime |
| Markov head TP=2 correctness | Medium | TP-aware layers; verify shapes at runtime |
| No CUDA graphs | Low | ~5-12 min cold boot; optimize later |
| Subagent async broken | Low | Only affects dev workflow, not runtime |

## dspark/ Documentation

```
dspark/
├── README.md                    Quick orientation
├── PROGRESS.md                  This file — implementation state
├── HANDOFF.md                   Session handoff
├── IMPLEMENTATION_PLAN.md       5-phase roadmap
├── ALGORITHM_REFERENCE.md       Paper equations & pseudocode
├── checkpoint_anatomy.md        Weight structure & shapes
├── dspark_model_skeleton.py     Design stubs
├── sts_calibration.py           STS calibration (ready, not integrated)
├── DSpark_paper_full.md         OCR'd paper
├── gb10_recipes_analysis.md     Cluster recipe analysis
├── backend_compatibility.md     Backend matrix
├── phase0_findings.md           Phase 0 decisions
├── phase2_dflash_analysis.md    DFlash speculator pattern
├── review-correctness.md        Review findings (5 blockers fixed)
└── review-integration.md        Integration review (0 blockers)
```

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

1. **Rebuild & test** with latest weight loading fixes (commit `f1cdf686f`)
2. **Verify model loads** — target model (~83s) + draft model (~70s) should complete without errors
3. **First inference** — send a simple prompt, verify draft tokens are produced
4. **Benchmark acceptance rate** — compare DSpark's acceptance rate vs standard MTP (~2.2/2)
5. **Phase 3**: Add CUDA graphs for DSpark speculator
6. **Phase 4**: Tier 2 scheduling (batch-averaged confidence truncation)
7. **Phase 5**: STS calibration on held-out set
