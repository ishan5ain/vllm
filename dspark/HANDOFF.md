# DSpark vLLM Integration — Handoff

> **Date:** 2026-06-28
> **Status:** 🔧 DEBUGGING — 0% draft acceptance, hc_head fix pending cluster test
> **Branch:** `dspark-research` on `github.com/ishan5ain/vllm`
> **Latest commit:** `d4d66dfee` — run backbone with 3D input for proper hc_head
> **Read first:** `dspark/PROGRESS.md` — full state, all bugs fixed, gaps

## What This Is

Integration of DeepSeek's DSpark speculative decoding into vLLM for
`deepseek-ai/DeepSeek-V4-Flash-DSpark` on a 2× DGX Spark GB10 cluster.

## Current State

**Phase 2 (Draft Generation) code complete. DSpark is serving at O1 with PIECEWISE
CUDA graphs on the main model.** However, draft acceptance is **0% across all 5
positions** — the hc_head was receiving wrong input (1 stream × 4 identical copies
instead of 4 genuine mhc-encoded streams). Fix committed in `d4d66dfee`, pending
cluster test.

9 bugs fixed across weight loading, EAGLE3 interface, tensor dimensionality,
kernel compatibility, and draft correctness.

## Architecture at a Glance

```
Target model forward → layers 40,41,42 → _dspark_context_buffer [T, 3×D]
  ↓
model_runner → DSparkProposer.propose() → DSparkSpeculator
  ↓
DSparkSpeculator → _prepare_dspark_inputs → forward_dspark_block()
  ↓
forward_dspark_block:
  fc context proj → embed [anchor,noise×4] → 3D expand (hc_mult=4)
  → backbone(3 layers, mhc encoding) → hc_head (4-stream)
  → Markov W₁W₂ sampling → confidence head
  ↓
Return [num_reqs, γ=5] draft tokens + logits + confidence
```

## Key Decisions

- Standard `vllm-node` container (not Chthonic b12x)
- Draft TP=2, O1 with PIECEWISE CUDA graphs on main model
- `DSparkProposer(SpecDecodeBaseProposer)` — model_runner integration
- `causal=False` in draft attention (bidirectional within block)
- cooperative_topk disabled on Blackwell (≥sm_100), fallback to persistent_topk
- Memory: `gpu_memory_utilization=0.85`, `max_num_seqs=1`

## Resolved Issues

| # | Bug | Commit |
|---|---|---|
| 1 | `model.` prefix mismatch in weight lookups | `0346cbd7b` |
| 2 | `.norm.weight` / `main_norm` / `attn_sink` weight loading | `f1cdf686f` |
| 3 | Integration: method routing, model auto-set, proposer chain | pre-cluster |
| 4 | EAGLE3 interface during init | `cbaa4ad2a` |
| 5 | 2D/3D hidden_state IndexError | `90ead3aea` |
| 6 | cooperative_topk crash (warmup + inference) | `f0b84ed07` + O1 recipe |
| 7 | **0% draft acceptance** — hc_head wrong input | `d4d66dfee` (pending test) |

## Quick Start

```bash
cd ~/repos/spark-vllm-docker
./build-and-copy.sh --vllm-ref dspark-research \
  --vllm-repo https://github.com/ishan5ain/vllm.git \
  --rebuild-vllm --copy-to 192.168.0.183
docker tag vllm-node:latest vllm-node:dspark
ssh 192.168.0.183 "docker tag vllm-node:latest vllm-node:dspark"
./run-recipe.sh deepseek-v4-flash-dspark --no-ray
```

## Immediate Next Steps

1. **Cluster test hc_head fix** — if acceptance >0%, benchmark
2. **Phase 3b: DSpark CUDA graphs** — build DSparkCudaGraphManager (see `phase3_cudagraph_plan.md`)
3. **Phase 4: Confidence scheduling** — integrate confidence head
4. **Phase 5: STS calibration** — calibrate acceptance thresholds
