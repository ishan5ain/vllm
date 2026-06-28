# DSpark vLLM Integration — Handoff

> **Date:** 2026-06-28
> **Status:** ✅ SERVING on 2× DGX Spark GB10 cluster
> **Branch:** `dspark-research` on `github.com/ishan5ain/vllm`
> **Read first:** `dspark/PROGRESS.md` — full state, gaps, build/run commands
> **Latest commit:** `f0b84ed07` — sparse_attn: disable cooperative_topk on Blackwell+

## What This Is

Integration of DeepSeek's DSpark speculative decoding into vLLM for
`deepseek-ai/DeepSeek-V4-Flash-DSpark` on a 2× DGX Spark GB10 cluster.

## Current State

**Phases 0–2 complete. Model is serving at ~15 tok/s decode, ~13K tok/s prefill.**

All 8+ bugs across weight loading, EAGLE3 interface, tensor dimensionality, and
kernel compatibility have been resolved. The server accepts `/v1/chat/completions`
requests and generates tokens with DSpark speculative decoding enabled.

Performance is capped by O0 eager mode (no CUDA graphs, no FlashInfer autotune)
due to cooperative_topk kernel incompatibility on SM120a (GB10's Blackwell GPU).
The standard Chthonic b12x + MTP path achieves ~52 tok/s on the same hardware.

## Architecture at a Glance

```
Target model forward → captures layers 40,41,42 → _dspark_context_buffer
  ↓
model_runner → DSparkProposer.propose() → DSparkSpeculator
  ↓
DSparkSpeculator → _prepare_dspark_inputs() → forward_dspark_block()
  ↓
forward_dspark_block: fc proj → embed [anchor,mask×4] → backbone(3 layers)
  → hc_head → Markov W₁W₂ sampling → confidence head
  ↓
Return [num_reqs, γ=5] draft tokens
```

## Key Decisions

- Standard recipe (`vllm-node`, default backends), NOT Chthonic b12x
- Draft TP=2 (matches target)
- `DSparkProposer(SpecDecodeBaseProposer)` for model_runner integration
- `causal=False` in attention metadata (bidirectional within block)
- CUDA graphs deferred (Phase 3) — O0 eager mode currently
- cooperative_topk disabled on Blackwell (≥sm_100), falls back to persistent_topk
- Memory: `gpu_memory_utilization=0.85`, `max_num_seqs=1` on GB10

## Resolved Cluster-Test Issues

| # | Bug | Fix Commit |
|---|---|---|
| 1 | `model.` prefix mismatch in weight lookups | `0346cbd7b` |
| 2 | EAGLE3 interface requirement during init | `cbaa4ad2a` |
| 3 | 2D/3D hidden_state IndexError in context capture | `90ead3aea` |
| 4 | cooperative_topk crash (warmup + inference) | recipe `-O0` + `f0b84ed07` |

## Quick Start (Cluster)

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

1. **Benchmark acceptance rate** — is DSpark achieving >4/5 accepted tokens vs MTP's ~2.2/2?
2. **Verify Markov head** — compare draft token sequences against reference implementation
3. **Profile overhead** — quantify time in DSpark forward vs target forward
4. **Phase 3** — re-enable CUDA graphs once cooperative_topk is fixed on SM120a
5. **Phase 4** — integrate confidence head for adaptive draft truncation
6. **Phase 5** — STS calibration on held-out set
