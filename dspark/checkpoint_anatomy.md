# DSpark Checkpoint Weight Anatomy

> Source: `deepseek-ai/DeepSeek-V4-Flash-DSpark` model.safetensors.index.json

---

## ✅ VERIFIED MAPPING (2026-06-28, from local checkpoint + reference code)

> Verified against the **actual** `model.safetensors.index.json` and DeepSeek's
> shipped reference modeling code at `inference/model.py` in the checkpoint
> (classes `DSparkBlock`, `DSparkAttention`, `Transformer.forward_spec`). This
> supersedes any conflicting detail later in this file. The earlier draft of
> this doc invented `mtp.2.shared_head.*` and an `fc` projection — neither
> exists in the checkpoint.

### Exact mtp.* weight names present (experts collapsed)

```
# Every stage (mtp.0, mtp.1, mtp.2) — a standard DeepSeek-V4 HC Block:
mtp.{s}.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}
mtp.{s}.attn.{q_norm,kv_norm}.weight
mtp.{s}.attn.attn_sink
mtp.{s}.attn_norm.weight
mtp.{s}.ffn.gate.{weight,bias}
mtp.{s}.ffn.experts.{0..255}.{w1,w2,w3}.{weight,scale}
mtp.{s}.ffn.shared_experts.{w1,w2,w3}.{weight,scale}
mtp.{s}.ffn_norm.weight
mtp.{s}.{hc_attn_fn,hc_attn_scale,hc_attn_base}
mtp.{s}.{hc_ffn_fn,hc_ffn_scale,hc_ffn_base}

# mtp.0 ONLY — the CONTEXT PROJECTION:
mtp.0.main_proj.{weight,scale}     # Linear[3*dim -> dim]  (dim*len(target_layer_ids))
mtp.0.main_norm.weight             # RMSNorm applied after main_proj

# mtp.2 ONLY — output head + DSpark heads:
mtp.2.{hc_head_fn,hc_head_base,hc_head_scale}   # draft hc_head output proj
mtp.2.norm.weight                               # pre-LM-head RMSNorm
mtp.2.markov_head.markov_w1.weight  [129280, 256]
mtp.2.markov_head.markov_w2.weight  [256 -> 129280 via ParallelHead]
mtp.2.confidence_head.proj.weight   [1, 4352]   # dim(4096)+markov_rank(256)

# Shared from the TARGET model (top-level, NO mtp prefix):
embed.weight        # token embedding — shared into draft
head.weight         # tied LM head    — shared into draft
# (top-level norm.weight + hc_head_* belong to the MAIN model, not the draft)
```

### Weights that DO NOT EXIST (vLLM currently fabricates these → random init)

`fc`, `enorm`, `hnorm`, `e_proj`, `h_proj`, `eh_proj`, `shared_head`,
`emb.tok_emb`. The draft has **no** MTP-style embed/hidden input projection.

### How context actually flows (reference `forward_spec`)

1. `main_x = mtp.0.main_norm(mtp.0.main_proj(main_hidden))`, where
   `main_hidden = cat([h_40, h_41, h_42], dim=-1)` and each `h_i` is the target
   layer output **averaged over the hc_mult streams**: `h.mean(dim=2)` → `[b,s,d]`.
   ⚠️ vLLM captures `hidden_states[:, 0, :]` (first stream only) — **wrong**.
2. `draft_ids = [anchor, noise_token_id × (block_size-1)]`;
   `x = embed(draft_ids)`; `x = x.unsqueeze(2).repeat(1,1,hc_mult,1)` → `[b, γ, hc_mult, d]`.
3. `for block in mtp: x = block(x, start_pos, input_ids, main_x)`.
   `DSparkAttention` cross-attends the draft block to `main_x`'s KV **plus** its
   own block KV (bidirectional within block). The **same** `main_x` feeds all 3
   stages — it is NOT added into the position-0 embedding.
4. `mtp.2.forward_head`: `hc_head(x)` (3D→2D) → `norm` → shared `head` → base
   logits; then the Markov sampling loop; then `confidence_head(x, markov_embed)`.

### Required vLLM code fixes (root cause of 0% acceptance)

| # | Fix | Where |
|---|---|---|
| A | Replace random `fc` with `mtp.0.main_proj` + `mtp.0.main_norm`; stop skipping them in `load_weights` | `dspark.py` |
| B | Remove `enorm`/`hnorm`/`e_proj`/`h_proj`; feed embeddings directly to blocks | `dspark.py` `DSparkLayer`, `forward_dspark_block` |
| C | Pass `main_x` as cross-attention context to every block (DSparkAttention), not added to embedding[:,0] | `forward_dspark_block` + attention |
| D | Capture target context as **mean over hc_mult** (`h.mean(dim=2)`), not `[:,0,:]` | `model.py` context capture |
| E | `mtp.2.norm.weight` → draft pre-head norm; use shared top-level `head.weight`; drop all `shared_head` remaps | `load_weights` `WEIGHT_NAME_REMAPPING` |
| F | Embedding is shared top-level `embed.weight`; drop the dead `emb.tok_emb` remap | `load_weights` |

The completeness assertion added to `load_weights` (2026-06-28) will hard-fail
at load until A/B/E are done — it lists every parameter with no checkpoint
source. That is intentional: serving with random projections wastes cluster time
at guaranteed 0% acceptance.

---


## MTP Layer Structure (3 layers)

### mtp.0 — Backbone Layer 1
```
mtp.0.attn.wq_a.weight, .scale           Q projection (lora compressed)
mtp.0.attn.wq_b.weight, .scale           Q projection (lora expanded)
mtp.0.attn.wkv.weight, .scale            KV combined projection
mtp.0.attn.wo_a.weight, .scale           O projection (lora compressed)
mtp.0.attn.wo_b.weight, .scale           O projection (lora expanded)
mtp.0.attn.q_norm.weight                 Q RMSNorm
mtp.0.attn.kv_norm.weight                KV RMSNorm
mtp.0.attn.attn_sink                     Attention sink values
mtp.0.attn_norm.weight                   Pre-attention RMSNorm
mtp.0.ffn_norm.weight                    Pre-FFN RMSNorm
mtp.0.ffn.gate.weight, .bias            Router gate
mtp.0.ffn.experts.{0..255}.w1.*         256 expert gate projections
mtp.0.ffn.experts.{0..255}.w2.*         256 expert down projections
mtp.0.ffn.experts.{0..255}.w3.*         256 expert up projections
mtp.0.ffn.shared_experts.w1,w2,w3       Shared expert weights
mtp.0.hc_attn_fn, hc_attn_scale, hc_attn_base   mHC attention residual
mtp.0.hc_ffn_fn,  hc_ffn_scale,  hc_ffn_base    mHC FFN residual
mtp.0.main_norm.weight                  Output RMSNorm
mtp.0.main_proj.weight, .scale          Output projection (to hidden_size)
```

### mtp.1 — Backbone Layer 2
```
(Same structure as mtp.0, minus main_norm and main_proj)
mtp.1.attn.*                            MLA attention
mtp.1.attn_norm.weight
mtp.1.ffn_norm.weight
mtp.1.ffn.gate.*
mtp.1.ffn.experts.{0..255}.*
mtp.1.ffn.shared_experts.*
mtp.1.hc_attn_fn, hc_attn_scale, hc_attn_base
mtp.1.hc_ffn_fn,  hc_ffn_scale,  hc_ffn_base
```

### mtp.2 — Output Layer + DSpark Heads
```
(Same attention + FFN structure as mtp.1, PLUS:)
mtp.2.hc_head_fn                        [4, 16384] Hypercompressed head function
mtp.2.hc_head_base                      [4]       HC head bias
mtp.2.hc_head_scale                     [1]       HC head scale
mtp.2.norm.weight                       Output RMSNorm
mtp.2.shared_head.head.weight           Vocab projection (shared with target)
mtp.2.shared_head.norm.weight           Pre-head RMSNorm

═══════════════════════════════════════════════════════════════════
DSPARK-SPECIFIC WEIGHTS (mtp.2 only)
═══════════════════════════════════════════════════════════════════

mtp.2.markov_head.markov_w1.weight      [129280, 256]   W₁ embedding (Eq 5)
mtp.2.markov_head.markov_w2.weight      [256, 129280]   W₂ projection (Eq 5)
mtp.2.confidence_head.proj.weight       [1, 4352]       Confidence head (Eq 7)
                                       ^^^^  ^^^^
                                        |     └── 4096 (hidden) + 256 (markov_rank)
                                        └── scalar output → sigmoid
```

## Config.json DSpark Fields

```json
{
  "dspark_block_size": 5,
  "dspark_noise_token_id": 128799,
  "dspark_target_layer_ids": [40, 41, 42],
  "dspark_markov_rank": 256,
  "num_nextn_predict_layers": 1
}
```

**Note on `num_nextn_predict_layers`:** The config says 1, but the checkpoint has 3 MTP layers (mtp.0, mtp.1, mtp.2). The existing vLLM MTP loader uses this to determine how many layers to create. For DSpark, this MUST be overridden to 3, or DSpark needs its own config field (`dspark_num_layers`).

The base V4-Flash checkpoint (non-DSpark) uses `num_nextn_predict_layers: 1` and only has mtp.0.

## How Target Context Flows

```
Target model layers 0..42
         │
         ├── Layer 40 hidden states ──┐
         ├── Layer 41 hidden states ──┼── concat → fc projection → draft context
         ├── Layer 42 hidden states ──┘   (W_c: [4096, 3×4096] → [4096])
         │
         ▼
    Draft backbone (mtp.0, mtp.1)
         │
         ▼
    Draft output (mtp.2) → hc_head → base_logits
         │
         ├── base_logits + markov bias → draft_logits
         └── hidden + markov_emb → confidence_head → confidence[5]
```

## Per-Layer Hidden Dimension Trace

```
Target hidden:         4096
Target context (3 layers concat):  3 × 4096 = 12288
After fc projection:   4096         (hidden_norm applied)
Draft backbone hidden: 4096         (all backbone layers)
After hc_head:         4 × 4096     (hypercompressed, 4-way)
After shared_head:     vocab_size = 129280
```

## Memory Estimates (Draft Model Only)

| Component | Approx Size |
|---|---|
| 3 × MTP layers (MLA attn) | ~2 GB |
| 3 × MTP layers (MoE, 256 experts) | ~6 GB |
| Markov head (W₁ + W₂) | 129280×256 + 256×129280 ≈ 132 MB |
| Confidence head | 4352 × 1 ≈ negligible |
| HC head + shared head | ~50 MB |
| **Total draft model** | **~8 GB** |
