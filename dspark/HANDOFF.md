# DSpark vLLM Integration — Handoff Document

> **Date:** 2026-06-27
> **Sessions:** playground research → implementation (phases 0–2)
> **Branch:** `dspark-research` (pushed to `origin`)
> **Target Hardware:** 2× NVIDIA DGX Spark (GB10 / SM121, 128 GB unified each)

## Quick Start

```bash
git checkout dspark-research
cat dspark/PROGRESS.md    # Current state, gaps, next steps ← READ THIS FIRST
cat dspark/README.md      # Quick orientation
```

## What This Is

Integration of DeepSeek's **DSpark** speculative decoding into vLLM for
`deepseek-ai/DeepSeek-V4-Flash-DSpark` on a 2× DGX Spark GB10 cluster.

DSpark accelerates inference via:
1. **Semi-autoregressive drafting** — Markov head (W₁W₂, rank 256) adds
   inter-token dependencies within a γ=5 parallel draft block
2. **Confidence head** — Predicts per-position acceptance probabilities
3. **Bidirectional block attention** — `is_causal=False` within the draft block

## Current State (2026-06-27)

**Phases 0–2 code-complete. Not yet tested on cluster.**

| Layer | Files | Status |
|---|---|---|
| Target model context capture | `model.py` | Layers 40,41,42 hidden states captured during forward |
| DSpark draft model | `dspark.py` (762 lines) | Weights, Markov head, confidence head, block generation |
| DSpark speculator | `dspark/speculator.py` (398 lines) | KV cache, attention metadata (causal=False), draft inputs |
| Config & routing | `speculative.py`, `registry.py`, `__init__.py` | Auto-detection, method routing |
| Context plumbing | `model_runner.py` | DSpark context passed via aux_hidden_states |

See `dspark/PROGRESS.md` for full architecture flow, known gaps, and next steps.

## Key Design Decisions

| Decision | Choice |
|---|---|
| Target recipe | Standard (`vllm-node`, Ray, default backends). NOT Chthonic b12x. |
| Draft TP | TP=2 (matches target). Markov W₁/W₂ are TP-aware. |
| Detection | `hasattr(hf_config, "dspark_block_size")` in config override |
| Speculator | Custom `DSparkSpeculator`, not MTPSpeculator |
| CUDA graphs | Deferred to Phase 3 (eager mode for now) |

## Next Step

**Cluster integration test** — build Docker image with this fork, start standard
recipe pointing at DSpark checkpoint with `--speculative-config '{"method":"dspark","num_speculative_tokens":5}'`.

## Key Docs

| Doc | Purpose |
|---|---|
| `PROGRESS.md` | Current state, architecture flow, gaps, task list |
| `IMPLEMENTATION_PLAN.md` | 5-phase roadmap with Slack discussion context |
| `phase0_findings.md` | TP strategy, method integration, config analysis |
| `phase2_dflash_analysis.md` | DFlash speculator pattern (reference for DSpark) |
| `gb10_recipes_analysis.md` | Cluster recipe analysis → standard recipe chosen |
| `checkpoint_anatomy.md` | Weight key mappings from checkpoint |
| `ALGORITHM_REFERENCE.md` | Paper equations and inference flow |
