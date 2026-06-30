# DSpark vLLM Integration — Handoff

> **Date:** 2026-06-29
> **Status:** 🔁 **PIVOTING to upstream PR #46995.** Our hand-rolled integration
>   loads & serves but is stuck at 0% acceptance because of two wrong
>   architectural choices (custom context buffer; hand-rolled non-causal
>   attention). An official DSpark PR by the DFlash author solves both with
>   existing machinery. Decision: adopt the PR; keep our (confirmed) weight
>   anatomy and GB10 fixes.
> **Branch:** `dspark-research` on `github.com/ishan5ain/vllm`
> **Latest commit:** `eb2377bee` (`c5d4dd2a6`) — diagnostics era (superseded by pivot)
> **PR to adopt:** [vLLM #46995](https://github.com/vllm-project/vllm/pull/46995),
>   branch `benchislett:dspark` @ `8b82d11`, base `main`, MERGEABLE.
>
> **Read first (in order):**
> 1. `dspark/PR_46995_COMPARISON.md` — what we missed / got right / GB10 insights.
> 2. `dspark/MIGRATION_PLAN.md` — step-by-step adoption (keep/replace/delete/add).
> 3. `dspark/checkpoint_anatomy.md` → "✅ VERIFIED MAPPING" — confirmed by the PR.
>
> ⚠️ Do NOT continue the diagnostic / "context is None" investigation — that path
> is being deleted. The early-return we were probing does not exist in the PR.
>
> ---
> **Historical status:** 🔧 LOADS & SERVES with all-real weights and coherent
> output, but draft acceptance is **0%** (greedy: 0/795 tokens) because
> `propose()` early-returns (`target_context_all is None`) — the draft never
> executes. The pivot resolves this structurally.

## What This Is

Integration of DeepSeek's DSpark speculative decoding into vLLM for
`deepseek-ai/DeepSeek-V4-Flash-DSpark` on a 2× DGX Spark GB10 cluster.

## Current State

**The rewired model loads (79.4 GiB, all draft params real — the completeness
assertion passes), serves at `http://spark10.local:8000/v1`, and generates
coherent output. Acceptance is 0% because THE DRAFT MODEL NEVER RUNS.**

🔴 **Real root cause (2026-06-29):** `propose()` early-returns because
`target_context_all is None`, returning zero draft tokens that are 100% rejected.
Established by elimination: file-based diagnostics are enabled (worker sees
`/tmp/dspark_debug_on`, shares `/tmp`), yet `/tmp/dspark_debug.log` is never
created on either node despite drafts being produced — so `forward_dspark_block`
is never reached. **The target DSpark context isn't reaching the speculator; the
draft doesn't execute.** All the weight/architecture work fixed *loading* only.

The context *should* be present (`_dspark_context_buffer` allocated on last PP
rank; getter returns it; `model_runner.py:611-614` injects it into
`aux_hidden_states`), so `c5d4dd2a6` adds a **propose-entry probe** (before the
early return) dumping: propose-called, `aux` len, `_target_model` type, getter
presence, getter buffer shape, `ctx_is_none` — to localize the loss in one run.

### Run the diagnostic build

1. Rebuild from `dspark-research` @ `c5d4dd2a6`; deploy to **both** nodes (keep
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
   (Cap `DSPARK_DEBUG_CALLS`, default 3, resets per process start.) The
   `DSPARK_DEBUG propose-entry:` line shows `ctx_is_none` + getter buffer shape
   → tells you where the context is lost (model_runner / getter / buffer /
   `_target_model`). If `forward_dspark_block` runs, its `block`/`k=` lines also
   appear (separate counter).

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

## Immediate Next Steps — POST-PIVOT (see `dspark/MIGRATION_PLAN.md`)

1. **✅ DONE — Phase M0: adopt PR #46995** (commit `fe8afd81c`). Cherry-picked the
   PR feature commit `8b82d11` (not a branch merge — see plan), resolved 4
   conflicts, reconciled auto-merged files, deleted `dspark_proposer.py` + V1
   wiring, kept our cooperative_topk fallback. All files `py_compile`-clean.
   ⚠️ **dspark now needs an explicit `method: dspark`** in the speculative config
   — verify the cluster recipe passes it before M2.
2. **➡️ NEXT — Phase M1: validate Sparse-MLA on GB10 (sm_121) — BLOCKER.** Run
   `tests/v1/attention/test_dspark_noncausal_sparse_mla.py` on the box; confirm a
   backend (FlashMLA / FlashInfer TRTLLM) passes on sm_121 or arrange a fallback.
3. **Phase M2: build & serve** from this branch (both nodes, same image). Run
   `pre-commit`/`ruff` on the build (not done on the dev box).
4. **Phase M3: measure acceptance** (greedy, `/metrics`). Target AL ≈ 5.
5. **Phase M4: reconcile Flash config** field names (`n_mtp_layers`, `dspark_*`,
   `hc_*`).

Deferred (out of scope upstream too): confidence scheduling, dynamic drafting,
STS calibration.

### Historical next-steps (hand-rolled path — superseded)

~~Diagnostic build → fix context plumbing → implement cross-attention (fix C) →
verify Markov head → DSpark CUDA graphs.~~ Replaced by adopting the PR, whose
EAGLE3 aux path + Sparse-MLA non-causal attention + DFlash-based speculator cover
all of these.

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
