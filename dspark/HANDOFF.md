# DSpark vLLM Integration — Handoff

> **Date:** 2026-06-29
> **Status:** 🔧 LOADS & SERVES with all-real weights and coherent output, but
>   draft acceptance is **0%** (greedy: 0/795 tokens). Diagnostic build committed;
>   **pending cluster rebuild with `DSPARK_DEBUG=1`** to localize the cause before
>   implementing fix C (cross-attention; designed in `dspark/phase_cd_plan.md`).
> **Branch:** `dspark-research` on `github.com/ishan5ain/vllm`
> **Latest commit:** `461fe59d3` — diagnostics enable via TOGGLE FILE (env didn't
>   reach worker); on `2a4c5ae85` file-sink, `ecd5e8db4` diagnostics, `b7aff3407`
>   loader fix, `9c83c59cb` rewire
> **Read first:** `dspark/PROGRESS.md` → "Current Status (2026-06-29)", then
>   `dspark/checkpoint_anatomy.md` → "✅ VERIFIED MAPPING" and `dspark/phase_cd_plan.md`.

## What This Is

Integration of DeepSeek's DSpark speculative decoding into vLLM for
`deepseek-ai/DeepSeek-V4-Flash-DSpark` on a 2× DGX Spark GB10 cluster.

## Current State

**The rewired model now loads (79.4 GiB, all draft params real — the completeness
assertion passes), serves at `http://spark10.local:8000/v1`, and generates
coherent output. But draft acceptance is still 0%.** Fresh greedy request: 159
drafts → 795 draft tokens → **0 accepted** (acceptance 0.00%, length 1.000). The
target path is correct; the draft is produced but 100% rejected.

0/795 *exact* is stronger than "weak context" alone, so a diagnostic build
(`ecd5e8db4`, `DSPARK_DEBUG=1`) was added to localize the cause (context not
reaching the draft / anchor-position misalignment / Markov bias swamping the base
logits / backbone) before committing to the larger cross-attention change.

### Run the diagnostic build

1. Rebuild from `dspark-research` @ `461fe59d3`; deploy to **both** nodes (keep
   images in sync — image skew caused the prior multi-node startup failure).
2. Relaunch, then **enable diagnostics via the toggle file** (env doesn't reach
   the worker — see gotcha #2):
   ```
   docker exec vllm_node touch /tmp/dspark_debug_on
   ssh 192.168.0.183 "docker exec vllm_node touch /tmp/dspark_debug_on"
   ```
3. Send one short greedy request, then read the diagnostics **from the file**:
   ```
   docker exec vllm_node cat /tmp/dspark_debug.log
   ssh 192.168.0.183 "docker exec vllm_node cat /tmp/dspark_debug.log"
   ```
   (Cap `DSPARK_DEBUG_CALLS`, default 3, resets per process start.)

> **⚠️ Gotcha #1 — do NOT use `docker logs`.** vLLM runs on the container pty
> (`/dev/pts/0`), not PID 1's stdout, so `docker logs vllm_node` is empty (0-byte
> json.log) and the pty master reader is detached (can't `cat` it). That's why
> `2a4c5ae85` writes diagnostics to `/tmp/dspark_debug.log` (`DSPARK_DEBUG_FILE`).
>
> **⚠️ Gotcha #2 — env doesn't reach the worker.** PID 1 is `sleep infinity`;
> vLLM is `docker exec`-ed in. `-e DSPARK_DEBUG=1` lands in the container
> `Config.Env` but NOT in the vLLM worker's environ, so the env gate stayed off
> (empty file despite drafts). That's why `461fe59d3` adds the `/tmp/dspark_debug_on`
> toggle (`DSPARK_DEBUG_TOGGLE`), checked per call — enable with `docker exec
> touch`, no restart. Acceptance itself is readable any time from `/metrics`.

The last measured 0% (and the earlier per-position 0%) is consistent across the
pre- and post-rewire builds — the rewire fixed *loading*, not yet acceptance.

**Root cause (verified 2026-06-28 against the local HF checkpoint + DeepSeek's
shipped `inference/model.py`):** the vLLM draft model invented weights that do
not exist in the checkpoint and left them randomly initialized — which
guaranteed garbage drafts and 0% acceptance regardless of the hc_head fix:

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

## Architecture rewire + loader fix (committed `9c83c59cb`, `b7aff3407`)

Edits in `vllm/models/deepseek_v4/nvidia/dspark.py` and `.../model.py`
(lint-clean via ruff; not yet built/tested on cluster):

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

**Loader fix (`b7aff3407`)** — the completeness assertion fired on the first
build (25 unloaded params). Fixes:
- Removed the `if name not in params_dict: continue` guard that pre-empted the
  expert loader + the `shared_experts.w2→down_proj` / `gate.bias→
  e_score_correction_bias` renames (MoE experts, shared down_proj, gate bias
  were silently random). Now matches the proven `mtp.py` loader.
- markov/confidence load explicitly before the stacked loop (`markov_w1`
  collides with the `"w1"` substring; `spec_layer!=mtp_start` had skipped them).
- `confidence_proj` created with `bias=False` (checkpoint has no bias).

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
  main_proj+main_norm → main_x;  embed [anchor,noise×4] → 3D expand (hc_mult=4)
  → [INTERIM: main_x added to anchor embed]  → backbone(3 stages, mhc encoding)
  → hc_head (4-stream) → Markov W₁W₂ sampling → confidence head
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
| 7 | **0% draft acceptance** — hc_head wrong input | `d4d66dfee` |
| 8 | Draft model architecture mismatch vs checkpoint (A/B/D/E/F) | `9c83c59cb` |
| 9 | Weight loading: experts/down_proj/gate bias/markov/confidence unloaded | `b7aff3407` |

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

1. **Diagnostic build** (`461fe59d3`) — rebuild both nodes; `docker exec
   vllm_node touch /tmp/dspark_debug_on` (both nodes); one greedy request; read
   `/tmp/dspark_debug.log` via `docker exec` (not `docker logs`, not env-only),
   localize the 0% cause.
2. **Fix what the diagnostics reveal** — a smaller bug (alignment / Markov
   scaling / context) and/or confirmation that fix C is the lever.
3. **Implement proper C** (cross-attention) per `dspark/phase_cd_plan.md`
   (option (a): per-stage one-token main_kv prefill into the draft KV cache).
   Remove the interim embedding addition. Target >3/5 acceptance.
4. **Verify Markov head end-to-end** against the reference `forward_head` loop.
5. **Phase 3b: DSpark CUDA graphs** — build DSparkCudaGraphManager (see `phase3_cudagraph_plan.md`)
6. **Phase 4: Confidence scheduling** — integrate confidence head
7. **Phase 5: STS calibration** — calibrate acceptance thresholds

## Risks / open questions

- `main_proj` fp8 scale: RESOLVED — the model loaded without a `KeyError` on
  `main_proj.weight_scale_inv`, so the scale resolved. Sanity-check the projected
  context magnitude via `ctx_norm` in the diagnostics.
- Fix D adds 3 extra `mhc_post_tilelang` calls (layers 40/41/42) per forward —
  watch for correctness/perf; confirm the captured context looks sane.
- Interim C (context added to anchor embedding, not cross-attention) is the
  leading suspect for 0% acceptance — but the diagnostics will confirm before we
  invest in the cross-attention rewrite.
- Keep both nodes on the SAME image — image skew caused the multi-node startup
  failure (`Connection closed by peer`) on a prior build.
