# DSpark Forward Path — Profiling & Instrumentation Scout

## 1. Full Call Chain: `DSparkSpeculator.propose()` → `forward_dspark_block()`

### Entry: `DSparkSpeculator.propose()`
**File:** `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` (lines 331–428)

```
propose()
 ├── _get_anchor_data()                              # 1. extract anchor tokens/positions
 ├── target_context_all[anchor_indices]              # 2. slice context
 ├── _prepare_dspark_inputs()                        # 3. build [B*γ] draft inputs
 ├── _build_draft_attn_metadata()                    # 4. build causal=False attn metadata
 ├── build_slot_mappings_by_layer()                  # 5. build slot mappings
 └── [set_forward_context] self.model.forward_dspark_block()  # 6. DSpark block forward
      ├── 1. fc(context)                             # context projection
      ├── 2. embed_tokens(input_ids)                 # token embedding
      ├── 3. inject context at position 0            # embeds[:,0,:] += ctx
      ├── 4. for layer in layers:                     # backbone loop (3 layers)
      │    ├── enorm(norm) + hnorm(norm)             # dual RMSNorm (fused)
      │    ├── h_proj + e_proj                       # merge projections
      │    ├── mtp_block.forward():                   # V4 decoder block
      │    │    ├── mhc_fused_post_pre_tilelang       # mHC post+pre (post residual mapping + pre-norm GEMM)
      │    │    ├── DeepseekV4Attention               # MLA attention
      │    │    ├── mhc_fused_post_pre_tilelang       # mHC post+pre (FFN side)
      │    │    └── DeepseekV4MoE                     # MoE FFN
      │    └── mhc_post_tilelang                     # L2 residual mapping
      ├── 5. hc_head_fused_kernel_tilelang           # hypercompressed LM head
      │    ├── mtp_shared_head_rmsnorm               # output RMSNorm
      │    └── logits_processor(head, hidden)         # LM head GEMM + TP gather
      ├── 6. Markov sequential sampling (γ iterations):
      │    ├── markov_w1(prev)                       # W₁ embedding lookup
      │    ├── markov_w2(prev_emb)                   # W₂ linear projection
      │    ├── base_logits[:,k,:] + bias              # combine base + Markov bias
      │    ├── argmax / multinomial                   # token selection
      │    └── confidence_proj([h_k; prev_emb])       # confidence score
      └── return {draft_tokens, draft_logits, confidence}
```

### Step-by-step `forward_dspark_block()` (lines 484–555 in dspark.py)

| Step | Operation | Line | Description |
|------|-----------|------|-------------|
| 1 | `self.fc(target_context)` | 504 | Context projection: [B, 3×D] → [B, D] via ReplicatedLinear |
| 2 | `self.embed_tokens(draft_input_ids)` | 507 | Embedding lookup: [B×γ] → [B×γ, D] vocab-parallel |
| 3 | Inject context | 512–514 | Add ctx to position 0 embedding |
| 4 | Backbone loop (3 layers) | 517–531 | Per-layer: dual RMSNorm → merge projections → decoder block → mHC post |
| 5 | hc_head on output layer | 534–547 | Hypercompressed LM head (fused kernel + RMSNorm + logits processor) |
| 6 | Markov loop (γ=5 iters) | 554–573 | W₁ lookup → W₂ bias → combine → sample → confidence |

---

## 2. Operation-by-Operation Analysis

### Op A: Context Projection (`fc`)
- **File:** `vllm/models/deepseek_v4/nvidia/dspark.py`, line 504
- **Code:** `ctx = self.fc(target_context)` — `ReplicatedLinear([13056] → [4352])`
- **Compute intensity:** **Low** — small GEMM: B × (3D × D) = very few FLOPs. Memory-bound.
- **Timing breakpoint:** Simple `time.time()` before/after. No cudaEvent needed for this tiny op.

### Op B: Token Embedding Lookup
- **File:** `vllm/models/deepseek_v4/nvidia/dspark.py`, line 507
- **Code:** `draft_embeds = self.embed_tokens(draft_input_ids)`
- **Compute intensity:** **Memory-bound** — pure lookup, no compute. (B×γ) index → (B×γ×D) gather.
- **Timing breakpoint:** `time.time()` before/after, or merge with Op A.

### Op C: Context Injection
- **File:** `vllm/models/deepseek_v4/nvidia/dspark.py`, lines 512–514
- **Code:** `embeds_3d[:, 0, :] += ctx`
- **Compute intensity:** **Memory-bound** — pointwise add on [B, D]. Negligible FLOPs.
- **Timing breakpoint:** Negligible; not worth standalone timing.

### Op D: Backbone Layers (×3, per-layer)
- **File:** `vllm/models/deepseek_v4/nvidia/dspark.py`, lines 517–531

Each layer breaks down into:

#### D1. Fused Dual RMSNorm
- **File:** `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py`, lines 153–203
- **Code:** `layer.enorm(hidden_states)` + `layer.hnorm(hidden_states)` (called separately in `forward_dspark_block`)
  - Note: In `DSparkLayer.forward()` (used by step-by-step), these are fused via `fused_mtp_input_rmsnorm`. In `forward_dspark_block()` they are separate `enorm`/`hnorm` calls (lines 524–525).
- **Compute intensity:** **Memory-bound** — RMSNorm is O(T×D) with element-wise ops. FLOPs ≈ 5×T×D per norm.
- **Timing breakpoint:** torch.cuda.Event pair before `enorm(hn)` and after `hnorm(nh)`.

#### D2. Input Projections
- **File:** `vllm/models/deepseek_v4/nvidia/dspark.py`, line 526
- **Code:** `projected = layer.h_proj(norm_hid) + layer.e_proj(norm_emb)`
  - Two `ReplicatedLinear([D] → [D])` GEMMs + pointwise add
  - Each GEMM: (B×γ) × D × D FLOPs ≈ 2×(B×γ)×D² MACs
- **Compute intensity:** **Medium** — GEMM on [T, D] × [D, D]. For T=B×γ=50 (B=10, γ=5), D=4352 → ~1.9 GMACs per projection.
- **Timing breakpoint:** torch.cuda.Event pair before `h_proj` and after the add.

#### D3. Decoder Block (`mtp_block.forward`)
- **File:** `vllm/models/deepseek_v4/nvidia/model.py`, lines 862–923

##### D3a. mHC Fused Post-Pre (Attn side)
- **File:** `vllm/model_executor/kernels/mhc/tilelang.py`, line 306 (`mhc_fused_post_pre_tilelang`)
- **File:** `vllm/model_executor/kernels/mhc/tilelang_kernels.py`, line 359 (`mhc_fused_tilelang`)
- **What:** Residual mapping + pre-norm GEMM fused into one kernel:
  1. Post-mix mapping: `new_r[j] = pm[j] * x_in[h] + Σ_k cm[k,j] * residual_in[k,h]`
  2. Squared sum accumulation for RMSNorm denominator
  3. FMA: `acc[n] += weight[n,j,h] * new_r[j]`
  
  **Note:** When `norm_weight` provided (always in DSpark), RMSNorm is fused into the kernel output path.
- **Compute intensity:** **High** (compute-bound) — The GEMM dominates: T × hc_mult × (2×hc_mult + 1) × hidden ≈ T × 4 × 9 × 4352 per-pre FLOPs. The fused post-mix adds O(T × hc_mult² × hidden). Total ~75 GMACs for T=50.
- **Timing breakpoint:** torch.cuda.Event pair wrapping the `mhc_fused_post_pre_tilelang` call.

##### D3b. DeepseekV4 Attention (MLA)
- **File:** `vllm/models/deepseek_v4/nvidia/model.py`, line 905
- **Code:** `x = self.attn(positions, x, None)`
- **What:** MLA attention with KV compression & indexer. For DSpark: `is_causal=False` (bidirectional within draft block).
- **Compute intensity:** **Medium-high** — Attention compute grows with O(T²×d_head) for bidirectional (T=B×γ up to 50). But small T makes this more memory-bound than FLOP-bound. MLA compression: O(T×D×d_compressed) GEMMs.
- **Timing breakpoint:** torch.cuda.Event pair wrapping `self.attn(...)`.

##### D3c. mHC Fused Post-Pre (FFN side)
- **Same kernel as D3a** but with `hc_ffn_fn` parameters.
- **File:** `vllm/models/deepseek_v4/nvidia/model.py`, lines 909–923
- **Timing breakpoint:** torch.cuda.Event pair wrapping after the FFN-side mHC call, before the MoE.

##### D3d. DeepseekV4MoE (FFN)
- **File:** `vllm/models/deepseek_v4/nvidia/model.py`, line 924
- **Code:** `x = self.ffn(x, input_ids)`
- **What:** MoE FFN — router gating + selected expert GEMMs. With Megablocks/DeepGEMM.
- **Compute intensity:** **Very high (compute-bound)** — The dominant compute op. Each token goes through gate + top-K experts. ~2×D×D_ffn FLOPs per expert per token × K experts.
- **Timing breakpoint:** torch.cuda.Event pair wrapping `self.ffn(...)`.

#### D4. mHC Post Mapping
- **File:** `vllm/model_executor/kernels/mhc/tilelang.py`, lines 303–317
- **File:** `vllm/model_executor/kernels/mhc/tilelang_kernels.py`, lines 495–520+
- **Code:** `hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)`
- **What:** L2 residual combination: `x_out[j,h] = pm[j] * d[h] + Σ_k cm[k,j] * b[k,h]` — purely memory-bound pointwise operations.
- **Compute intensity:** **Low (memory-bound)** — O(T × hc_mult² × hidden) FLOPs but all element-wise. ~8.7 MMACs for T=50.
- **Timing breakpoint:** torch.cuda.Event pair wrapping `mhc_post_tilelang`.

### Op E: hc_head (Hypercompressed LM Head)
- **File:** `vllm/models/deepseek_v4/nvidia/dspark.py`, lines 534–547

#### E1. Input preparation
- **File:** `dspark.py`, lines 536–538
- **Code:** `hc_input = hidden_states.reshape(-1, 1, D).repeat(1, hc_mult, 1)`
- **Compute intensity:** **Memory-bound** — reshape + broadcast. No real compute.

#### E2. hc_head_fused_kernel_tilelang
- **File:** `vllm/model_executor/kernels/mhc/tilelang.py`, lines 613–644
- **File:** `vllm/model_executor/kernels/mhc/tilelang_kernels.py`, lines 718–817
- **What:** Two-pass fused kernel: (1) per-channel dot-product + RMSNorm sum-of-squares accumulation, (2) sigmoid-gated weighted sum over channels.
- FLOPs: ~T × hc_mult × (2×hidden_size) for dot products + T × hc_mult × hidden_size for weighted sum ≈ T × 3 × hc_mult × D.
- **Compute intensity:** **Medium** — ~5.7 GMACs for T=50, hc_mult=1, D=4352. But the kernel avoids materializing intermediates; likely memory-bandwidth bound for small T.
- **Timing breakpoint:** torch.cuda.Event pair wrapping `hc_head_fused_kernel_tilelang`.

#### E3. mtp_shared_head_rmsnorm
- **File:** `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py`, lines 124–150
- **Code:** `hc_output = mtp_shared_head_rmsnorm(hc_output, weight, eps)`
- **Compute intensity:** **Memory-bound** — RMSNorm on [T, D].
- **Timing breakpoint:** Merge with E2 or E4.

#### E4. LogitsProcessor (LM head GEMM)
- **File:** `vllm/model_executor/layers/logits_processor.py`, lines 54–70, 89–98
- **Code:** `base_logits = self.logits_processor(output_layer.shared_head.head, hc_output)`
- **What:** VocabParallel GEMM [T, D] × [D, V/tp] → [T, V/tp] + TP all-gather → [T, V]
- FLOPs: ~2 × T × D × V for the GEMM (V=129280). ~56.2 GMACs for T=50.
- **Compute intensity:** **High (compute-bound)** — Large vocabulary GEMM dominates.
- **Timing breakpoint:** torch.cuda.Event pair wrapping `logits_processor(...)`.

### Op F: Markov Sequential Sampling (γ iterations)
- **File:** `vllm/models/deepseek_v4/nvidia/dspark.py`, lines 552–573

#### F1. markov_w1 lookup + markov_w2 bias
- **File:** `dspark.py`, lines 555–556
- **Code:** `prev_emb = markov_w1(prev)` + `bias = markov_w2(prev_emb)`
- **F1a:** W₁: VocabParallelEmbedding lookup [B] → [B, rank=256]
- **F1b:** W₂: ColumnParallelLinear [B, 256] × [256, V/tp] → [B, V/tp] + TP gather → [B, V]
- FLOPs: ~2 × B × 256 × V. ~66 MMACs for B=10.
- **Compute intensity:** **Low (memory-bound)** — Very small GEMM, dominated by gather overhead.
- **Timing breakpoint:** torch.cuda.Event pair per iteration (or aggregate).

#### F2. Logit combination
- **File:** `dspark.py`, line 557
- **Code:** `step_logits = base_logits[:, k, :] + bias`
- **Compute intensity:** **Memory-bound** — pointwise add on [B, V].

#### F3. Token sampling
- **File:** `dspark.py`, lines 559–563
- **Code:** `argmax` or `softmax + multinomial`
- **Compute intensity:** **Memory-bound** — argmax is O(B×V) comparisons, softmax adds O(B×V) FLOPs.

#### F4. Confidence score
- **File:** `dspark.py`, lines 565–567
- **Code:** `confidence_proj([h_k; prev_emb])` — ReplicatedLinear [D+256] → [1]
- **Compute intensity:** **Low** — tiny GEMM [B, D+256] × [D+256, 1].

---

## 3. Natural Timing Breakpoints

### Recommended instrumentation points (torch.cuda.Event pairs):

| ID | Name | File | Lines | Granularity |
|----|------|------|-------|-------------|
| **T0** | `forward_dspark_block` total | dspark.py | 484–575 | Whole block |
| **T1** | Context projection (fc) | dspark.py | 504 | Single op |
| **T2** | Embedding lookup | dspark.py | 507 | Single op |
| **T3** | Backbone total (3 layers) | dspark.py | 517–531 | Aggregate |
| **T3a** | Per-layer: dual norm + projections | dspark.py | 524–526 | Per layer |
| **T3b** | Per-layer: Attn mHC post+pre | model.py | 902–904 | Per kernel |
| **T3c** | Per-layer: MLA attention | model.py | 905 | Per op |
| **T3d** | Per-layer: FFN mHC post+pre | model.py | 909–923 | Per kernel |
| **T3e** | Per-layer: MoE FFN | model.py | 924 | Per op |
| **T3f** | Per-layer: mHC post | dspark.py | 528–530 | Per kernel |
| **T4** | hc_head total | dspark.py | 534–547 | Aggregate |
| **T4a** | hc_head fused kernel | dspark.py | 539–546 | Single kernel |
| **T4b** | shared_head RMSNorm | dspark.py | 547 | Single op |
| **T4c** | LM head GEMM (logits_processor) | dspark.py | 548–550 | Single op |
| **T5** | Markov loop total | dspark.py | 552–573 | Aggregate (γ=5 iters) |
| **T5a** | Markov W₁+W₂ (per iter) | dspark.py | 555–556 | Sub-iter op |
| **T5b** | Sampling (per iter) | dspark.py | 559–563 | Sub-iter op |
| **T5c** | Confidence (per iter) | dspark.py | 565–567 | Sub-iter op |

### Where to insert torch.cuda.Event measurements:

In `forward_dspark_block()` (dspark.py):

```python
# At method top (line 484):
start_ev = torch.cuda.Event(enable_timing=True)
end_ev = torch.cuda.Event(enable_timing=True)

# Before each breakpoint, add a new pair:
# T1: Wrap line 504
# T2: Wrap line 507
# T3: Wrap lines 517-531
#   T3a-T3f: wrap individual lines within
# T4: Wrap lines 534-550
# T5: Wrap lines 552-573

# At method end (line 575):
# Aggregate all events and return in result dict
```

In `DeepseekV4DecoderLayer.forward()` (model.py lines 862–924):
- Lines 902–904: mHC fused post-pre (attn side)
- Line 905: attention
- Lines 909–923: mHC fused post-pre (FFN side)
- Line 924: MoE FFN

In `hc_head_fused_kernel_tilelang` (tilelang.py line 613):
- The kernel itself is opaque (TileLang JIT). Time the Python-side call.

In `logits_processor` forward (logits_processor.py line 54):
- The `_get_logits` call (line 64) triggers GEMM + TP gather.

---

## 4. Compute Intensity Summary

| Operation | Type | FLOPs estimate | Bound |
|-----------|------|---------------|-------|
| fc(context) | Small GEMM | ~B × 13056 × 4352 | Memory |
| embed_tokens | Lookup | 0 | Memory |
| context injection | Pointwise add | B × D | Memory |
| Dual RMSNorm (per layer) | Norm | ~10 × T × D | Memory |
| e_proj + h_proj (per layer) | Medium GEMM | ~4 × T × D² | Medium |
| mHC post-pre (per side, per layer) | Fused GEMM+norm | ~T × hc_mult × (2hc_mult+1) × D | Compute |
| MLA attention (per layer) | Attention | ~2 × T² × d_head + compression GEMMs | Mixed |
| MoE FFN (per layer) | Expert GEMMs | ~2 × T × D × D_ffn × K | Compute |
| mHC post (per layer) | Fused pointwise | ~T × hc_mult² × D | Memory |
| hc_head (output) | Fused dot+gate | ~3 × T × hc_mult × D | Memory |
| LM head GEMM | Large GEMM | ~2 × T × D × V | Compute |
| Markov W₁ lookup | Lookup | 0 | Memory |
| Markov W₂ bias | Small GEMM | ~2 × B × 256 × V | Memory |
| Confidence proj | Tiny GEMM | ~2 × B × (D+256) | Memory |
| Sampling | argmax/softmax | B × V | Memory |

Key: B = num_reqs, γ = block_size (5), T = B×γ, D = 4352, V = 129280, hc_mult = 1

---

## 5. Architecture Notes

### Two forward paths exist:
1. **Step-by-step** (MTPSpeculator-compatible): `DSparkLayer.forward()` (used for verification loop)
   - Called 3 times per target step (once per mtp layer)
   - Uses `fused_mtp_input_rmsnorm` for dual norm fusion
   - Path: `DSparkSpeculator._run_model()` → `self.model(...)` → `DSparkInnerModel.forward()` → `mtp_layer(...)`
   
2. **Block generation** (Phase 2): `DSparkInnerModel.forward_dspark_block()` (used by `propose()`)
   - Single call producing all γ tokens
   - Uses separate `enorm`/`hnorm` calls (not fused)
   - Adds Markov sequential sampling and confidence head

### Key architectural decisions affecting profiling:
- **Bidirectional attention** (is_causal=False): Attention complexity O(T²) instead of O(T) for causal, but T=γ×B is small (≤~50).
- **mHC** (Modified Hyper-Compression): Replaces standard residual+norm with learned mixing matrices — adds compute but eliminates norm overhead.
- **TileLang JIT kernels**: `mhc_post_tilelang`, `hc_head_fused_kernel_tilelang`, `mhc_fused_post_pre_tilelang` are TileLang JIT-compiled — instrumenting inside the kernel requires TileLang-level profiling.
- **Aux streams**: 3 CUDA streams for parallelizing attention GEMMs in the backbone (compressor kv_score, indexer weights_proj, indexer compressor kv_score).

---

## 6. Start Here

**First file to open for another agent:** `vllm/models/deepseek_v4/nvidia/dspark.py` at line 484 (`forward_dspark_block` method).

This is the single entry point for the DSpark block generation path. All profiling breakpoints should be inserted here, with subordinate breakpoints in `model.py` (decoder layer) and tilelang kernels as needed.

### Risk: `hc_head_fused_kernel_tilelang` n_splits
In the `forward_dspark_block()` method at lines 536-538, the hc_input is created as `[B*γ, hc_mult=1, D]`. However, in `compute_logits()` (line 305-306), the input is reshaped to match `hc_mult=4` from the actual output layer. The `forward_dspark_block` path may need `n_splits` computation for deep_gemm support if hc_mult > 1 — but currently hc_mult=1 for DSpark (hc_head uses a different structure), so this is fine.

### Risk: Separate enorm/hnorm in block path
In `forward_dspark_block()` (lines 524-525), `enorm` and `hnorm` are called separately, unlike the fused path. This means two kernel launches instead of one — a possible optimization target.

---

## 7. Files Referenced

1. `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` (lines 331–428) — `DSparkSpeculator.propose()` entry
2. `vllm/models/deepseek_v4/nvidia/dspark.py` (lines 484–575) — `forward_dspark_block()`, all DSpark heads
3. `vllm/models/deepseek_v4/nvidia/model.py` (lines 782–924) — `DeepseekV4DecoderLayer` and its forward
4. `vllm/model_executor/kernels/mhc/tilelang.py` (lines 90–230, 303–317, 613–644) — mHC kernels (pre, post, hc_head)
5. `vllm/model_executor/kernels/mhc/tilelang_kernels.py` (lines 359–519, 495–520+, 718–817) — TileLang JIT kernels
6. `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py` (lines 124–203) — Fused RMSNorm
7. `vllm/model_executor/layers/logits_processor.py` (lines 54–70, 89–98) — LM head GEMM
