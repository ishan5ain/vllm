# DSpark — Migration Plan: Adopt Upstream PR #46995

> **Date:** 2026-06-29
> **Author:** research session (Claude)
> **Decision:** **Pivot from our hand-rolled integration to the upstream
>   implementation in [vLLM PR #46995](https://github.com/vllm-project/vllm/pull/46995)**
>   ("[Spec Decode] DSpark", by Benjamin Chislett — author of the DFlash
>   speculator). See `dspark/PR_46995_COMPARISON.md` for the full why.
> **PR ref:** branch `dspark` on `benchislett` fork, single commit
>   `8b82d11f1115979ddbe3e6379dda81dac9775aec`, base `main`, state MERGEABLE.

## Why pivot (one paragraph)

Our 0%-acceptance "context is None" blocker is **not a bug to fix** — it is a
symptom of building on the wrong foundation. We invented a custom
`_dspark_context_buffer` + getter + manual `model_runner` injection for target
context, and planned a hand-rolled cross-attention path for non-causal block
attention. PR #46995 solves both with existing vLLM machinery:

1. **Context** flows through the standard **EAGLE3 `aux_hidden_states`** path
   (target model implements `SupportsEagle3`/`EagleModelMixin`, returns aux
   states, speculator receives them as a `propose()` argument). No custom buffer.
2. **Non-causal attention** reuses the **Sparse-MLA** kernels by expanding each
   query's top-k index list to include the trailing context window **plus all
   block tokens (incl. future ones)** — index-driven, no causal mask. No new
   attention kernel.
3. The whole thing is a **thin subclass of DFlash** (`DSparkSpeculator(
   DFlashSpeculator)`), reusing parallel drafting, context-KV precompute, the
   prepare-inputs kernel, and CUDA-graph capture.

Our weight-mapping archaeology (A/B/D/E/F in `checkpoint_anatomy.md`) was
**correct** and is independently confirmed by the PR — it transfers directly.

## Prerequisites already satisfied in our branch

Verified 2026-06-29 on `dspark-research` @ `eb2377bee`:

- ✅ DFlash present: `vllm/v1/worker/gpu/spec_decode/dflash/{speculator,cudagraph,utils}.py`
  and `vllm/model_executor/models/qwen3_dflash.py`.
- ✅ `DFlashSpeculator` exposes the exact interface the PR subclasses
  (`_run_model`, `_generate_draft`, `_build_draft_attn_metadata`, `set_attn`,
  `capture`, `init_cudagraph_manager`, `load_draft_model`, `propose`) and attrs
  (`sample_indices`, `sample_pos`, `sample_idx_mapping`, `num_query_per_req`,
  `context_positions`, `parallel_drafting_token_id`).
- ✅ `init_speculator` already routes `"dspark"` → `DSparkSpeculator`.
- ✅ `EagleModelMixin` (interfaces.py:1322) + `SupportsEagle3` (interfaces.py:1372).
- ✅ `sparse_swa.py` exists (but **lacks** the non-causal path — must port).
- ✅ Our GB10 cooperative_topk fallback lives in
  `sparse_attn_indexer.py:468-499` — **keep it** (independent of DSpark).

## Recommended strategy: MERGE the PR, then re-apply GB10 deltas

The PR is one mergeable commit on the same `main` base. Cleanest path:

```bash
# from ~/repos/vllm on dspark-research
git fetch https://github.com/benchislett/vllm.git dspark:pr-46995
# Option A (preferred): reset our DSpark integration to the PR, keep GB10 fixes
#   1. create a safety tag of our current work
git tag dspark-handrolled-archive
#   2. merge the PR (expect conflicts ONLY in the files we also touched)
git merge pr-46995
```

Conflicts will appear **only** in files we modified for our hand-rolled path.
For every such file, **take the PR side** unless it is one of the GB10-specific
keeps below. Then delete the files the PR makes obsolete.

If the merge is messy, fall back to the **file-by-file port** in the table below
(cherry-pick the PR's version of each file).

## File-by-file: keep / replace / delete / add

| File | Our current state | Action | Notes |
|---|---|---|---|
| `vllm/v1/spec_decode/dspark_proposer.py` | `DSparkProposer(SpecDecodeBaseProposer)` | **DELETE** | PR has no separate proposer; `init_speculator` returns `DSparkSpeculator` directly. Remove any import of it. |
| `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` | `DSparkSpeculator(DraftModelSpeculator)` + custom KV/attn + hidden-state add + diagnostics | **REPLACE** with PR's `DSparkSpeculator(DFlashSpeculator)` (530 lines) | Drop ALL our diagnostics scaffolding (env/toggle/file-sink). The early-return we were debugging no longer exists. |
| `vllm/v1/worker/gpu/spec_decode/dspark/utils.py` | (none) | **ADD** PR's `load_dspark_model` | Aliases target embed/lm_head for the DSV4 draft; sets `use_non_causal` on the draft attention config. |
| `vllm/models/deepseek_v4/nvidia/dspark.py` | our draft model (fc removed, interim `main_x` add) | **REPLACE** with PR's (483 lines) | Same weight anatomy we found, plus `precompute_and_store_context_kv` (the real "fix C") and `_remap_dspark_name`. |
| `vllm/models/deepseek_v4/nvidia/model.py` | custom `_dspark_context_buffer` (1027) + capture + `get_dspark_context_hidden_states` (1443) | **REPLACE context path**: add `EagleModelMixin`+`SupportsEagle3`, return `(hidden, aux_hidden_states)`, capture via `idx+1 in aux_hidden_state_layers` w/ `mean(dim=1)`. **DELETE** buffer + getter. | Our `mhc_post`+mean (fix D) was right; PR does `aux_recon.mean(dim=1)`. |
| `vllm/v1/worker/gpu/model_runner.py` | manual context injection (~611-614) + EAGLE3 bypass | **REPLACE** with PR's one-liner: add `"dspark"` to the `use_aux_hidden_state_outputs` method set | Remove our injection + bypass entirely. |
| `vllm/v1/attention/backends/mla/sparse_swa.py` | no non-causal path | **PORT** PR's `_compute_dspark_noncausal_swa_indices_kernel` + `is_dspark`/`noncausal_index_width` + `common_attn_metadata.causal` branch | **GB10 RISK — see below.** |
| `vllm/config/speculative.py` | our dspark detect/route | **RECONCILE** to PR: `method=="dspark"` model from target ckpt; `model_type→deepseek_v4` + arch `DSparkDraftModel`; `parallel_drafting=True`; `use_dspark()`; `use_eagle()` incl. dspark | Take PR side. |
| `vllm/config/vllm.py` | (likely unmodified) | **ADD** PR's force-V2 for dspark + allow dspark in `_get_v2_model_runner_unsupported_features` | |
| `vllm/v1/core/sched/scheduler.py` | (likely unmodified) | **ADD** `if use_dspark(): num_lookahead_tokens = num_spec_tokens` | Anchor-as-first-prediction (N, not N+1). |
| `vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py` | (likely unmodified) | **ADD** `dspark_target_layer_ids` → aux layers (`i+1`) | Registers which target layers to capture. |
| `vllm/v1/worker/gpu/spec_decode/utils.py` | (likely unmodified) | **ADD** `dspark_noise_token_id` to `get_parallel_drafting_token_id` | |
| `vllm/model_executor/models/registry.py` | our `DeepSeekV4DSparkModel` reg | **RECONCILE** to PR: `DSparkDraftModel` → `deepseek_v4:DSparkDeepseekV4ForCausalLM`; (optional) `Qwen3DSparkModel` | |
| `vllm/model_executor/models/qwen3_dspark.py` | (none) | **ADD (optional/defer)** | Qwen3 dense DSpark — not needed for Flash; harmless to include. |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | our cooperative_topk Blackwell fallback | **KEEP** | Independent of DSpark; still required on GB10 (sm_121). |
| `tests/v1/attention/test_dspark_noncausal_sparse_mla.py` | (none) | **ADD** | 529-line correctness suite for the non-causal sparse path. Run it (see risks). |

### Docs (this directory) — keep, but re-frame
`checkpoint_anatomy.md` stays valid (weight anatomy confirmed). `phase_cd_plan.md`
(our hand-rolled cross-attention design) is **superseded** by the PR's
sparse-index + KV-precompute approach — mark it historical, don't delete.

## 🚨 Top risk to validate FIRST: GB10 (sm_121) Sparse-MLA support

The PR's non-causal attention depends on a Sparse-MLA backend. The PR tests on
**B300 (sm_100)** with **FlashMLA (SM90/SM100)** and **FlashInfer TRTLLM MLA
Sparse (SM100/SM120)**. **DGX Spark GB10 is sm_121.** Before trusting the
attention path:

1. Determine which sparse-MLA backend (if any) is selected/available on sm_121.
   The PR's launch uses `--kv-cache-dtype fp8`; the test guards FlashInfer TRTLLM
   behind `supports_compute_capability(cap)` for SM 10.x and FlashMLA behind
   `is_flashmla_sparse_supported()`.
2. Run `tests/v1/attention/test_dspark_noncausal_sparse_mla.py` **on the GB10
   box** (not just CI) to confirm a backend passes there. If both skip on
   sm_121, the elegant approach won't run for us as-is and we must either:
   (a) get FlashInfer TRTLLM MLA Sparse enabled for sm_121, or
   (b) fall back to an eager/dense non-causal path for the draft block.
3. This interacts with our existing cooperative_topk sm_100 fallback — verify
   the sparse indexer path is consistent on sm_121.

## Phased execution

- **Phase M0 — Validate foundation (no cluster):** merge the PR into a scratch
  branch; resolve conflicts per the table; `ruff`/import-check; run the new
  non-causal test on any available SM100/SM120 box if the GB10 is busy.
- **Phase M1 — GB10 kernel validation:** run `test_dspark_noncausal_sparse_mla.py`
  on the GB10 cluster. Decide backend (FlashMLA / FlashInfer TRTLLM / fallback).
  **Gate:** a sparse-MLA backend passes on sm_121, or a fallback is in place.
- **Phase M2 — Build & serve:** rebuild image from the migrated branch, deploy to
  **both** nodes (keep images in sync — image skew caused prior startup failures),
  `./run-recipe.sh deepseek-v4-flash-dspark --no-ray`. Confirm it loads & serves.
- **Phase M3 — Measure acceptance:** greedy request; read `/metrics`
  (`spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total`). Target AL ≈ 5
  (PR reports >350 TPS @ BS1, AL~5 on 8×B300). This is the real test of the pivot.
- **Phase M4 — Flash-config reconciliation:** confirm Flash's `config.json` field
  names match what the PR reads (`n_mtp_layers`, `dspark_target_layer_ids`,
  `dspark_markov_rank`, `dspark_noise_token_id`, `hc_mult`, `hc_eps`,
  `expert_dtype`). The PR was authored against DSV4-**Pro**; we run **Flash**.

## Known limitations inherited from the PR (set expectations)

- **Probabilistic drafting is broken upstream** (completions degrade into
  junk/loops; acceptance inflated; suspected rejection-sampling bug). **Greedy is
  the validated path** — matches our Tier-1 plan. Don't expect sampling to work.
- **Qwen3 DSpark has an IMA under CUDA graphs** (irrelevant to Flash, but a sign
  the CUDA-graph buffer wiring is still settling).
- Dynamic drafting / confidence-based scheduling are **out of scope** in the PR
  (tracked in PR #45953). Our Phase 4/5 ambitions remain future work.

## What we keep from our own work

- Weight-mapping knowledge (`checkpoint_anatomy.md` A/B/D/E/F) — confirmed.
- cooperative_topk Blackwell/sm_121 fallback (`sparse_attn_indexer.py`).
- Build/run recipe + cluster gotchas (pty log capture, env-to-worker, image
  sync) in `PROGRESS.md`/`HANDOFF.md`.
- The `dspark-handrolled-archive` tag, in case we need to reference our path.
