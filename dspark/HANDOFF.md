# DSpark vLLM Integration — Handoff

> **Date:** 2026-06-28
> **Status:** 🔧 ARCHITECTURE REWIRED to match checkpoint (A/B/D/E/F done,
>   uncommitted, untested on cluster). Fix C (cross-attention) is INTERIM —
>   proper version designed in `dspark/phase_cd_plan.md`.
> **Branch:** `dspark-research` on `github.com/ishan5ain/vllm`
> **Latest commit:** `d4d66dfee` (code edits below are uncommitted)
> **Read first:** `dspark/checkpoint_anatomy.md` → "✅ VERIFIED MAPPING", then
>   `dspark/phase_cd_plan.md`, then `dspark/PROGRESS.md`.

## What This Is

Integration of DeepSeek's DSpark speculative decoding into vLLM for
`deepseek-ai/DeepSeek-V4-Flash-DSpark` on a 2× DGX Spark GB10 cluster.

## Current State

**Phase 2 code runs end-to-end and serves at O1 with PIECEWISE CUDA graphs, but
draft acceptance is 0% across all 5 positions.** The hc_head 2D→3D fix
(`d4d66dfee`) was necessary but is **not** sufficient.

**Root cause (verified 2026-06-28 against the local HF checkpoint + DeepSeek's
shipped `inference/model.py`):** the vLLM draft model invents weights that do
not exist in the checkpoint and leaves them randomly initialized — which
guarantees garbage drafts and 0% acceptance regardless of the hc_head fix:

- `self.fc` (context projection) has **no checkpoint source**. The real context
  projection is `mtp.0.main_proj` + `mtp.0.main_norm` — which `load_weights`
  currently **skips**.
- `enorm` / `hnorm` / `e_proj` / `h_proj` (per-layer input projections) **do not
  exist** in DSpark at all. The reference feeds token embeddings directly into
  the blocks; context enters via cross-attention (`DSparkAttention` takes
  `main_x`), not via an MTP-style embed/hidden merge.
- Head/norm remaps point at a non-existent `shared_head`. The draft shares the
  top-level `head.weight` and uses `mtp.2.norm.weight` as its pre-head norm.
- Target-context capture uses `hidden[:, 0, :]` (first hc stream); the reference
  uses the **mean over hc_mult streams** (`h.mean(dim=2)`).

See `dspark/checkpoint_anatomy.md` → "✅ VERIFIED MAPPING" for the exact weight
list, data flow, and the A–F fix table.

**Guardrail (`load_weights`):** hard-fails listing every parameter with no
checkpoint source (token embedding + tied head exempted). This now PASSES with
the rewire below — every draft parameter has a real checkpoint weight.

## Architecture rewire done this session (A/B/D/E/F)

Edits in `vllm/models/deepseek_v4/nvidia/dspark.py` and `.../model.py`
(uncommitted; lint-clean via ruff; not yet built/tested on cluster):

- **A** — `self.fc` removed; context projection is now `mtp.0.main_proj`
  (fp8, quant_config) + `mtp.0.main_norm` on the input stage. These load
  (were previously skipped).
- **B** — `enorm`/`hnorm`/`e_proj`/`h_proj` removed everywhere; token
  embeddings feed the decoder blocks directly (matches reference).
- **D** — target context capture is now `mhc_post`-applied + mean over
  `hc_mult` (reference `h.mean(dim=2)`), not first-stream pre-`mhc_post`.
- **E/F** — dead `.emb.tok_emb`/`.head.weight` remaps removed; `mtp.2.norm`
  → `shared_head.norm`; embedding + LM head shared via `load_eagle_model`
  (confirmed in eagle/utils.py:67–85).
- **C (INTERIM)** — `main_x` is added to the anchor embedding. Proper
  cross-attention (per-stage `main_kv` in the draft KV cache) is designed
  in `dspark/phase_cd_plan.md` but NOT implemented (needs speculator/KV
  changes; untestable from here).

9 prior bugs fixed across weight loading, EAGLE3 interface, tensor
dimensionality, kernel compatibility, and draft correctness.

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

1. **Build & cluster-test the rewire (A/B/D/E/F)** — confirm the model loads
   (the completeness assertion passes), serves, and measure acceptance. With
   interim C, expect low-but-possibly-nonzero acceptance.
2. **Implement proper C** (cross-attention) per `dspark/phase_cd_plan.md`
   (option (a): per-stage one-token main_kv prefill into the draft KV cache).
   Remove the interim embedding addition. Target >3/5 acceptance.
3. **Verify Markov head end-to-end** against the reference `forward_head` loop.
4. **Phase 3b: DSpark CUDA graphs** — build DSparkCudaGraphManager (see `phase3_cudagraph_plan.md`)
5. **Phase 4: Confidence scheduling** — integrate confidence head
6. **Phase 5: STS calibration** — calibrate acceptance thresholds

## Risks to watch on first cluster build

- `main_proj` fp8 scale: expects `main_proj.weight_scale_inv` param (created by
  the fp8 quant method). If quant_config differs, the `.scale` load may need a
  different suffix.
- Fix D adds 3 extra `mhc_post_tilelang` calls (layers 40/41/42) per forward.
- Interim C: acceptance may stay low until proper cross-attention lands.
