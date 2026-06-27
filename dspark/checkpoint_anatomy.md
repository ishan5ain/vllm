# DSpark Checkpoint Weight Anatomy

> Source: `deepseek-ai/DeepSeek-V4-Flash-DSpark` model.safetensors.index.json

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
