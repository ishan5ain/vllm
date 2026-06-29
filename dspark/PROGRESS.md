# DSpark vLLM Integration — Progress & State

> **Date:** 2026-06-28
> **Branch:** `dspark-research` (fork: `github.com/ishan5ain/vllm`)
> **Latest commit:** `c5d4dd2a6` — propose-entry probe to localize the early return;
>   on `461fe59d3` toggle, `2a4c5ae85` file-sink, `ecd5e8db4` diagnostics,
>   `b7aff3407` loader fix, `9c83c59cb` rewire
> **State:** model LOADS & SERVES with all-real weights, output coherent, but draft
>   acceptance is still **0%**. **REAL ROOT CAUSE FOUND (2026-06-29): the draft model
>   never executes** — `propose()` early-returns because `target_context_all is None`,
>   returning zero draft tokens that are 100% rejected. All the weight/architecture
>   work fixed *loading*; the draft never even runs. **Pending rebuild @ `c5d4dd2a6`**
>   to localize where the context is lost (model_runner / getter / buffer).
> **Sessions:** research → implementation → review → cluster testing → serving → debugging acceptance → checkpoint-verified root cause → architecture rewire → weight-loading fix → loads-but-0%-acceptance → diagnostics

## Implementation Status

```
Phase 0: Prerequisites    ████████████ DONE
Phase 1: Model Loading    ████████████ DONE (4 cluster-test bugs fixed)
Phase 2: Draft Generation ████████████ DONE (serving, acceptance bug found)
Phase 3: CUDA Graphs      ██████████░░ IN PROGRESS (O1 enabled, DSpark graphs pending)
Phase 4: Scheduling       ░░░░░░░░░░░░ NOT STARTED
Phase 5: STS Calibration  ░░░░░░░░░░░░ CODE EXISTS, NOT INTEGRATED
```

## Current Status (2026-06-29)

**The rewired model loads (79.4 GiB, all draft params real — assertion passes),
serves at `http://spark10.local:8000/v1`, generates coherent output — but draft
acceptance is 0% because THE DRAFT MODEL NEVER RUNS.**

### 🔴 REAL ROOT CAUSE — `propose()` early-returns (context is None)

Established by elimination with the file-based diagnostics + toggle:
- Diagnostics are enabled (worker sees `/tmp/dspark_debug_on`, shares `/tmp` —
  verified via `/proc/<worker>/root/tmp`), yet **no `/tmp/dspark_debug.log` is
  created on either node** despite drafts being produced.
- The only way drafts are produced without writing the debug file is the
  `propose()` early return — `forward_dspark_block` is never reached:
  ```python
  if target_context_all is None:
      logger.warning_once("DSpark speculator: no target context available...")
      return torch.zeros(...)   # zeros counted as drafts → 100% rejected
  ```
- So **the target DSpark context never reaches the speculator; the draft model
  does not execute.** This — not weak drafts — is the 0% cause, and likely
  explains every prior session's 0%. The weight rewire fixed *loading* only.

### The puzzle + the probe (`c5d4dd2a6`)

Context *should* be present: `model.py` allocates `_dspark_context_buffer`
unconditionally on the last PP rank (PP=1 here), `get_dspark_context_hidden_states`
returns it, and `model_runner.py:611-614` inserts it into `aux_hidden_states` for
`propose()`. All three read correct, yet context is None. Code-reading can't
resolve it, so `c5d4dd2a6` adds a **propose-entry dump** (before the early return)
recording: is `propose` called, `aux` length, `_target_model` type, getter
presence, **getter buffer shape**, `ctx_is_none`. One run localizes the loss to
model_runner plumbing vs getter vs buffer vs `_target_model` wiring.

### Original (now superseded) plan: localize via base_logits diagnostics

### Acceptance re-measured after the rewire (2026-06-29)

Fresh **greedy** request via `/v1/chat/completions` (greedy ⇒ a draft token is
accepted iff it equals the target argmax), deltas from `/metrics`:

| Metric | Δ |
|---|---|
| `spec_decode_num_drafts` | 159 |
| `spec_decode_num_draft_tokens` | 795 (= 159 × γ=5) |
| `spec_decode_num_accepted_tokens` | **0** |
| acceptance rate / length | **0.00% / 1.000** |

DSpark *is* proposing all 5 tokens/step; the target produces correct text (so the
0% is purely the draft being rejected, not a serving failure). 0/795 *exact* is
stronger than "weak context" alone → likely a systematic bug (anchor/position
misalignment, Markov bias swamping base logits, or context not reaching the
draft) in addition to the missing cross-attention. Hence diagnostics before fix C.

### Diagnostics — COMMITTED `ecd5e8db4` + `2a4c5ae85` + `461fe59d3` (no-op unless enabled)

Enable with `DSPARK_DEBUG=1` **or** `touch /tmp/dspark_debug_on` (cap via
`DSPARK_DEBUG_CALLS`, default 3). Dumps:
- `forward_dspark_block` (`dspark.py`): anchor token, sampled draft tokens,
  `target_context` norm/nan, and per-step `base_logits` top-5 vs post-Markov
  top-5 with base/bias magnitudes (detects Markov bias swamping base logits).
- `DSparkSpeculator.propose` (`speculator.py`): anchor tokens/positions/indices,
  context shape/norm/nan, produced draft tokens (anchor/alignment sanity).

**⚠️ Two cluster gotchas this took to nail down:**

1. **Log capture (why `2a4c5ae85` exists):** vLLM runs on a container **pty**
   (`/dev/pts/0`), NOT PID 1's stdout. So `docker logs vllm_node` is empty
   (0-byte json.log), the pty master reader is detached (a concurrent
   `cat /dev/pts/0` got 0 bytes), and no tmux pane has it in scrollback.
   → diagnostics now ALSO append to a file (`DSPARK_DEBUG_FILE`, default
   `/tmp/dspark_debug.log`), readable via `docker exec`.

2. **Env doesn't reach the worker (why `461fe59d3` exists):** PID 1 in the
   container is `sleep infinity`; vLLM is launched by a separate `docker exec`.
   The launcher's `-e DSPARK_DEBUG=1` lands in the container `Config.Env` (and a
   fresh `docker exec printenv` shows it), but it does **NOT** reach the actual
   vLLM worker subprocess (`/proc/<worker>/environ` had no `DSPARK_*`). So the
   env-at-import gate stayed `False` and the first toggle run produced an EMPTY
   file even though 47 drafts were generated. (This means we still don't know
   whether the draft early-returns or runs-but-wrong — the next run reveals it.)
   → diagnostics are now enabled if `DSPARK_DEBUG=1` **OR** the file
   `/tmp/dspark_debug_on` exists (`DSPARK_DEBUG_TOGGLE`), checked per call.

**How to capture (after rebuild @ `461fe59d3`, relaunch):**
```bash
docker exec vllm_node touch /tmp/dspark_debug_on            # enable (both nodes)
ssh 192.168.0.183 "docker exec vllm_node touch /tmp/dspark_debug_on"
# send ONE greedy request, then read the file:
docker exec vllm_node cat /tmp/dspark_debug.log
ssh 192.168.0.183 "docker exec vllm_node cat /tmp/dspark_debug.log"
```
The cap counter (`DSPARK_DEBUG_CALLS`, default 3) resets per process start; for a
fresh 3 dumps after it's exhausted, restart vLLM. No-rebuild shortcut to try
first: relaunch, then `docker exec vllm_node sh -c 'tr "\0" "\n" <
/proc/$(pgrep -f "vllm serve"|head -1)/environ | grep DSPARK'` — if it prints
`DSPARK_DEBUG=1`, the current image already logs.

Interpretation guide:
- `ctx_norm≈0` / `ctx_nan=True` → context not reaching draft (capture/main_proj).
- `bias_max ≫ base_max` & post-top5 unrelated to base-top5 → Markov head swamping.
- `base_top5` gibberish/constant → backbone/hc_head bug; plausible → fix C is the lever.
- anchor token ≠ last real token, or positions off → alignment bug.

(Also removed pre-existing unused imports/locals in `speculator.py` so the
touched file is ruff-clean.)

### (Historical) hc_head fix — necessary but not sufficient

The hc_head 2D→3D fix (`d4d66dfee`) is necessary but NOT sufficient.

### ROOT CAUSE — verified against the local checkpoint (2026-06-28)

The HF checkpoint is local; its `model.safetensors.index.json` and DeepSeek's
shipped reference code (`inference/model.py`: `DSparkBlock`, `DSparkAttention`,
`Transformer.forward_spec`) were inspected directly. **The vLLM draft model
definition does not match the checkpoint** — several forward-path weights are
fabricated and stay randomly initialized, which forces 0% acceptance:

| vLLM (current) | Checkpoint reality | Fix |
|---|---|---|
| `self.fc` — random, no ckpt source | context proj = `mtp.0.main_proj` + `mtp.0.main_norm` (currently **skipped** in load) | A |
| `enorm`/`hnorm`/`e_proj`/`h_proj` — random | **do not exist**; embeddings go straight into blocks, context enters via `DSparkAttention(main_x)` cross-attn | B, C |
| context = `hidden[:, 0, :]` (1st hc stream) | `h.mean(dim=2)` (mean over hc_mult) | D |
| `.head/.norm → shared_head.*` (no such param) | shared top-level `head.weight`; `mtp.2.norm.weight` is pre-head norm | E |
| `emb.tok_emb` remap (dead) | shared top-level `embed.weight` | F |

Full verified mapping, data flow, and the A–F fix table: see
`dspark/checkpoint_anatomy.md` → "✅ VERIFIED MAPPING".

**Guardrail added (`load_weights`):** a completeness assertion hard-fails and
lists every parameter with no checkpoint source (token embedding + tied head
exempt). After the rewire below it PASSES — every draft param has a real weight.

### Architecture rewire — COMMITTED `9c83c59cb` (A/B/D/E/F), pending rebuild

| Fix | Change | File |
|---|---|---|
| A | Removed fabricated `fc`; context proj = `mtp.0.main_proj` (fp8) + `mtp.0.main_norm` (now load) | `dspark.py` |
| B | Removed `enorm`/`hnorm`/`e_proj`/`h_proj`; embeddings feed blocks directly | `dspark.py` |
| D | Context capture = `mhc_post` + mean over hc_mult (ref `h.mean(dim=2)`) | `model.py` |
| E/F | Dropped dead `.emb.tok_emb`/`.head.weight` remaps; `mtp.2.norm`→`shared_head.norm`; embed+head shared | `dspark.py` |
| C | **INTERIM** — `main_x` added to anchor embedding. Proper cross-attention designed in `dspark/phase_cd_plan.md`, NOT implemented | `dspark.py` |

Lint-clean (ruff check + format). Not built/tested — cluster build is the test.

#### Loader fixes — COMMITTED `b7aff3407` (assertion caught 25 unloaded params)

The first cluster build with the rewire hit the completeness assertion (25
unloaded params) and surfaced bugs (most pre-existing, silently random before):

- **Removed the `if name not in params_dict: continue` guard** in the weight
  loop (added a prior session to skip main_norm/main_proj). It fired *before*
  the expert loader and the `shared_experts.w2→down_proj` / `gate.bias→
  e_score_correction_bias` renames — so MoE experts (non-mega path uses
  `experts.routed_experts.*`), shared-expert down_proj, and the gate bias were
  never loading. Now matches the proven `mtp.py` loader. main_proj/main_norm
  load directly (they are real params now).
- **markov/confidence load explicitly before the stacked-params loop** —
  `markov_w1` collides with the `"w1"` stacked weight-name substring; also the
  `spec_layer != mtp_start` guard had skipped these mtp.2-only heads.
- **`confidence_proj` created with `bias=False`** (checkpoint has no bias).

After these, every draft param has a checkpoint source (embedding + tied head
shared) and the assertion is expected to pass. **Not yet rebuilt/re-run** — the
0% acceptance below predates the rewire+loader fixes and should be re-measured.

Prior hypothesis (hc_head identical-copies bug) remains valid but was only one of
several issues; it could not have raised acceptance above 0% on its own.

### Acceptance Rate Bug (found 2026-06-28)

```
SpecDecoding metrics: Mean acceptance length: 1.00, Accepted: 0 tokens,
Drafted: 670 tokens, Per-position acceptance rate: 0.000 across all 5 positions
```

Root cause: `forward_dspark_block` ran the backbone with 2D input `[T, D]` (hc_mult=1),
then called `.repeat(1, hc_mult, 1)` to create 4 identical copies for the hc_head kernel.
The hc_head kernel's 4×4 mhc mixing matrix expects genuinely unique streams produced
by the decoder layer's mhc encoding. With identical copies, the mixing produced wrong
logits → all draft tokens rejected.

Fix: expand draft embeddings to 3D `[T, hc_mult, D]` (hc_mult=4) before the backbone loop.
The decoder layer's `mhc_pre_tilelang` / `mhc_fused_post_pre_tilelang` kernels then
create proper 4-stream mhc encoding, matching what the target model's hc_head expects.

| Commit | Description |
|---|---|
| `55451307d` | First attempt: pass hc_mult=1 (insufficient — 1-stream vs 4-stream fundamentally different) |
| `d4d66dfee` | Proper fix: 3D backbone input → genuine 4-stream mhc encoding |

### Benchmark Results (O1, PIECEWISE CUDA graphs, before hc_head fix)

| Test | t/s | Notes |
|---|---|---|
| `tg128` (decode) | 6.9 | Lower than O0 due to draft overhead without acceptance |
| Drafted throughput | 34.5 tok/s | 670 tokens drafted over test window |
| Accepted throughput | 0.00 tok/s | All draft tokens rejected |

Expected after fix: decode speed should improve when draft tokens are accepted
(effective speed = decode_t/s × (1 + acceptance_rate × acceptance_length)).

### O1 Upgrade (2026-06-28)

Successfully moved from O0 to O1:
- `cudagraph_mode: PIECEWISE` — attention runs eagerly, FFN captured in piecewise graphs
- `enable_flashinfer_autotune: False` — autotune skipped (would crash cooperative_topk)
- Recipe uses `-O1` with `--compilation-config '{"cudagraph_mode":"PIECEWISE","custom_ops":[]}'`

### Memory Footprint (O1)

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
# O1 via compilation-config (not -O0):
# cudagraph_mode: PIECEWISE, custom_ops: []
```

## Cluster-Test Bugs Fixed

### Weight Loading (4 fixes, commits `c646e6965`–`90ead3aea`)

| Bug | Symptom | Fix |
|---|---|---|
| `model.` prefix mismatch | `KeyError` on all param lookups | Changed mtp→layer prefix from `model.layers.{idx}.` to `layers.{idx}.` |
| `.norm.weight` remap too broad | Corrupted norm names | Guard: exclude attn_norm, ffn_norm, kv_norm, q_norm |
| `main_norm`/`main_proj` missing | `KeyError` — MTP weights not in DSpark | Added to weight_name list + continue guard |
| `attn_sink` not in params | `KeyError` — may be buffer | Added `name not in params_dict` guard |
| Guard placement wrong | Guard after attn_sink/experts checks | Moved to top of outer else branch |

### Integration (3 fixes, pre-cluster-test)

| Bug | Symptom | Fix |
|---|---|---|
| `num_speculative_tokens` without model | ValidationError | Added `"dspark"` to MTP-like auto-set path |
| `NotImplementedError` for `"dspark"` | Auto-detection chain missing | Added model_type→method auto-detection |
| `Unknown speculative decoding method` | model_runner drafter chain missing | Added DSparkProposer + routing |

### Runtime & Correctness (5 fixes)

| Bug | Symptom | Fix | Commit |
|---|---|---|---|
| EAGLE3 interface required | `Model does not support EAGLE3` | Removed `use_aux_hidden_state_outputs` | `cbaa4ad2a` |
| 2D/3D hidden_state IndexError | `[:, 0, :]` on 2D tensor | Check `hidden_states.dim()` | `90ead3aea` |
| cooperative_topk crash (warmup) | `invalid argument` in autotune | Added cooperative_topk Blackwell guard | `f0b84ed07` |
| cooperative_topk crash (inference) | Same error at decode | Disabled on ≥sm_100, fallback persistent_topk | `f0b84ed07` |
| **0% draft acceptance** | hc_head receives 1 stream × 4 copies | **3D backbone input for proper 4-stream mhc encoding** | **`d4d66dfee`** |

## Architecture (files changed vs upstream)

```
vllm/config/speculative.py                 DSpark detection + method routing
vllm/model_executor/models/registry.py     DeepSeekV4DSparkModel registration
vllm/models/deepseek_v4/__init__.py        Re-export (NVIDIA only)
vllm/models/deepseek_v4/nvidia/model.py    Target context capture + 2D/3D fix
vllm/models/deepseek_v4/nvidia/dspark.py   Draft model + forward_dspark_block + hc_head fix
vllm/v1/worker/gpu/model_runner.py         Context plumbing + EAGLE3 bypass
vllm/v1/worker/gpu/spec_decode/__init__.py Speculator routing
vllm/v1/worker/gpu/spec_decode/dspark/     DSparkSpeculator (KV cache, attention, propose)
vllm/v1/spec_decode/dspark_proposer.py     DSparkProposer (SpecDecodeBaseProposer)
vllm/model_executor/layers/sparse_attn_indexer.py  cooperative_topk Blackwell fix
```

## Known Gaps

| Gap | Severity | Notes |
|---|---|---|
| Architecture mismatch vs checkpoint | RESOLVED (untested) | A/B/D/E/F rewired this session; assertion passes. Needs cluster build. |
| **Draft never runs (context is None)** | **BLOCKER** | `propose()` early-returns (`target_context_all is None`) → zero drafts → 0% acceptance. Probe `c5d4dd2a6` pending to localize where the context is lost. THIS is the active 0% cause. |
| Context via cross-attention (fix C) | HIGH (later) | Interim embedding-add; proper per-stage `main_kv` cross-attention still pending. Only relevant AFTER the draft actually runs. |
| `main_proj` fp8 scale loading | Medium-RESOLVED | Model loaded without KeyError on `main_proj.weight_scale_inv`, so the fp8 scale resolved. Confirm value sanity via `ctx_norm` in diagnostics. |
| Markov head correctness | Medium | Verified against paper + reference `forward_head` — logic correct; depends on correct inputs (now wired) |
| Confidence head unused | Medium | Scores computed but Phase 4 scheduling not implemented |
| No DSpark CUDA graphs | Medium | Main model has PIECEWISE graphs; speculator runs eagerly |
| Bidirectional attention | Medium | `causal=False` set, not validated at runtime |
| Draft KV cache | Medium | `set_attn()` wired; verify correctness |
| cooperative_topk on Blackwell | Medium | Fallback to persistent_topk; slower on sm_100 |
| O1 memory tight | Medium | 0.85 util needed; leaves only 9.7 GiB KV cache |

## Build & Run Commands

```bash
# Build
cd ~/repos/spark-vllm-docker
./build-and-copy.sh --vllm-ref dspark-research \
  --vllm-repo https://github.com/ishan5ain/vllm.git \
  --rebuild-vllm --copy-to 192.168.0.183
docker tag vllm-node:latest vllm-node:dspark
ssh 192.168.0.183 "docker tag vllm-node:latest vllm-node:dspark"

# Run
./run-recipe.sh deepseek-v4-flash-dspark --no-ray

# Stop
./launch-cluster.sh stop
```

## Next Steps (Priority Order)

1. **Diagnostic build** (`c5d4dd2a6`) — rebuild both nodes; `docker exec
   vllm_node touch /tmp/dspark_debug_on` (both nodes), send one short greedy
   request, then read `/tmp/dspark_debug.log` via `docker exec` (NOT `docker
   logs`; NOT just the env var — see gotchas above). Read the
   `DSPARK_DEBUG propose-entry:` line to localize where the context is lost.
2. **Fix the context plumbing** so `target_context_all` is non-None and
   `forward_dspark_block` actually runs (the current blocker). THEN re-measure
   acceptance; if still low, use the base_logits/anchor diagnostics (context /
   alignment / Markov-swamp / backbone) to localize.
3. **Implement proper fix C** (cross-attention) per `dspark/phase_cd_plan.md`
   — per-stage one-token `main_kv` prefill into the draft KV cache; remove the
   interim embedding add. Target >3/5 acceptance.
4. **Verify Markov head end-to-end** vs reference `forward_head`.
5. **Phase 3b: DSpark CUDA graphs** — build `DSparkCudaGraphManager` (see `phase3_cudagraph_plan.md`)
6. **Phase 4: Confidence-based scheduling** — integrate confidence head
7. **Phase 5: STS calibration** — run `dspark/sts_calibration.py`

### How to run the diagnostic build

```bash
# rebuild from dspark-research @ c5d4dd2a6, deploy to BOTH nodes (keep in sync)
# relaunch, then enable diagnostics WITHOUT env/restart by touching the toggle:
docker exec vllm_node touch /tmp/dspark_debug_on
ssh 192.168.0.183 "docker exec vllm_node touch /tmp/dspark_debug_on"
# send one greedy request, then read the FILE (docker logs does NOT work — pty):
docker exec vllm_node cat /tmp/dspark_debug.log
ssh 192.168.0.183 "docker exec vllm_node cat /tmp/dspark_debug.log"
```

Acceptance can be measured without logs straight off `/metrics` (deltas of
`vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total`).
