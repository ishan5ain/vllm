# DSpark vLLM Integration — Handoff

> **Date:** 2026-06-28
> **Branch:** `dspark-research` on `github.com/ishan5ain/vllm`
> **Read first:** `dspark/PROGRESS.md` — full state, gaps, build/run commands

## What This Is

Integration of DeepSeek's DSpark speculative decoding into vLLM for
`deepseek-ai/DeepSeek-V4-Flash-DSpark` on a 2× DGX Spark GB10 cluster.

## Current State

**Phases 0–2 code complete. Cluster testing in progress — weight loading
bugs found and fixed, pending rebuild and verification.**

The draft model loads weights from the DSpark checkpoint alongside the
target model. Integration spans 9 files across the vLLM codebase (config,
registry, model, speculator, proposer, model_runner).

Weight loading was the primary cluster-testing challenge — multiple
naming and remapping bugs required iterative fixes. The latest fix
(guard placement + attn_sink handler) is pending rebuild.

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
- CUDA graphs deferred to Phase 3

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
