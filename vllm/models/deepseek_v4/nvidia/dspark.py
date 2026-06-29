# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculative decoding draft model for DeepSeek V4.

DSpark extends the DFlash parallel draft model with:

1. A **semi-autoregressive head** (Markov W₁W₂, rank 256) that introduces
   inter-token dependencies within the parallel draft block (γ=5 tokens).
2. A **confidence head** (Linear[4352→1]) that predicts per-position
   acceptance probabilities.
3. **Bidirectional block attention** (is_causal=False) within the draft block.

Architecture (verified against the checkpoint's ``inference/model.py`` —
``DSparkBlock``/``Transformer.forward_spec``; see
``dspark/checkpoint_anatomy.md`` → "✅ VERIFIED MAPPING"):

- 3 stages (mtp.0, mtp.1, mtp.2), each a standard ``DeepseekV4DecoderLayer``.
  There is **no** enorm/hnorm/e_proj/h_proj — token embeddings feed the
  blocks directly.
- Input stage (mtp.0): ``main_proj`` + ``main_norm`` project the concatenated
  target context (layers 40,41,42) into ``main_x`` (the former, fabricated
  ``fc`` weight).
- Output stage (mtp.2): ``hc_head_*`` + ``shared_head`` (pre-head norm
  ``mtp.2.norm`` + tied LM head shared from target) + Markov/confidence heads.
- Token embedding and the LM head are shared from the target model.

Known gap (fix C, pending — not blind-coded): target context should enter via
DSparkAttention cross-attention (per-stage ``main_kv`` in the draft KV cache),
not by adding to the anchor embedding. See ``dspark/phase_cd_plan.md``.
"""

import typing
from collections.abc import Callable, Iterable

import regex as re
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
)
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_mtp import SharedHead
from vllm.model_executor.models.deepseek_v2 import get_spec_layer_idx_from_weight_name
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.deepseek_v4.common.ops import (
    mtp_shared_head_rmsnorm,
)
from vllm.sequence import IntermediateTensors

from .model import (
    DeepseekV4DecoderLayer,
    make_deepseek_v4_expert_params_mapping,
)

logger = init_logger(__name__)

# MoE expert scales suffix detection — matches the pattern in mtp.py.
# fp4 experts register ``..._weight_scale``; fp8 register ``..._weight_scale_inv``.
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


class DeepSeekV4DSparkLayer(nn.Module):
    """Single DSpark backbone stage.

    Matches the checkpoint's ``mtp.{stage}`` layout (verified against
    ``inference/model.py`` ``DSparkBlock`` — see
    ``dspark/checkpoint_anatomy.md`` → "✅ VERIFIED MAPPING"). Each stage is a
    standard ``DeepseekV4DecoderLayer`` (MLA attention + MoE FFN + mHC
    residuals). DSpark has **no** enorm/hnorm/e_proj/h_proj — token embeddings
    feed the blocks directly and the target context enters via attention.

    Stage-specific extras:
    - Input stage (mtp.0): ``main_proj`` (3·D → D, fp8) + ``main_norm`` — the
      target-context projection (``main_x``). This is what earlier code called
      ``fc``; the real weights are ``mtp.0.main_proj`` / ``mtp.0.main_norm``.
    - Output stage (mtp.2): ``hc_head_*`` + ``shared_head`` (pre-head norm
      ``mtp.2.norm`` + tied LM head shared from the target).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        topk_indices_buffer: torch.Tensor,
        prefix: str,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        is_output_layer: bool = False,
        is_input_layer: bool = False,
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_mult = config.hc_mult

        self.mtp_block = DeepseekV4DecoderLayer(
            vllm_config,
            prefix,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )

        # Input stage (mtp.0) only: target-context projection (main_x).
        # main_proj is fp8-quantized in the checkpoint (carries a ``.scale``),
        # so it is created with quant_config like the other V4 linears.
        if is_input_layer:
            self.target_layer_ids = getattr(
                config, "dspark_target_layer_ids", [40, 41, 42]
            )
            self.main_proj = ReplicatedLinear(
                len(self.target_layer_ids) * config.hidden_size,
                config.hidden_size,
                bias=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.main_proj",
            )
            self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output stage (mtp.2) only: hc_head + shared_head (pre-head norm + LM head).
        if is_output_layer:
            self.hc_eps = config.hc_eps
            self.hc_dim = self.hc_mult * config.hidden_size
            self.hc_head_fn = nn.Parameter(
                torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(self.hc_mult, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_head_scale = nn.Parameter(
                torch.empty(1, dtype=torch.float32),
                requires_grad=False,
            )
            self.shared_head = SharedHead(
                config=config, prefix=prefix, quant_config=quant_config
            )

    def project_main_context(self, target_context: torch.Tensor) -> torch.Tensor:
        """main_x = main_norm(main_proj(target_context)).

        Only valid on the input stage (mtp.0). ``target_context`` is the
        concatenated target hidden states [*, 3·D]; returns [*, D].
        """
        return self.main_norm(self.main_proj(target_context))

    def run_block(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Run the decoder block on 3D [*, hc_mult, D] hidden states.

        Token embeddings feed the block directly (no input projection). The
        mHC post-residual mapping is applied here so the returned hidden state
        is the full per-stage output, ready for the next stage or hc_head.
        """
        hidden_states, residual, post_mix, res_mix = self.mtp_block(
            positions=positions, x=hidden_states, input_ids=None
        )
        return mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)


class DSparkInnerModel(nn.Module):
    """DSpark inner draft model — holds layers, heads, and weight loading.

    This is the actual model that contains all DSpark parameters.
    Wrapped by ``DeepSeekV4DSparkModel`` for vLLM compatibility
    (``load_eagle_model`` expects a ``.model`` attribute).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        # ── DSpark config fields ──────────────────────────────────────
        self.block_size = getattr(config, "dspark_block_size", 5)
        self.noise_token_id = getattr(config, "dspark_noise_token_id", 128799)
        self.target_layer_ids = getattr(config, "dspark_target_layer_ids", [40, 41, 42])
        self.markov_rank = getattr(config, "dspark_markov_rank", 256)

        # ── Layer indices ─────────────────────────────────────────────
        self.mtp_start_layer_idx = config.num_hidden_layers  # 43
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 3)
        if self.num_mtp_layers != 3:
            logger.warning(
                "DSpark expects 3 draft layers; config has "
                "num_nextn_predict_layers=%d. Overriding to 3.",
                self.num_mtp_layers,
            )
            self.num_mtp_layers = 3
        # Write back so get_spec_layer_idx_from_weight_name finds all 3 layers.
        config.num_nextn_predict_layers = self.num_mtp_layers

        topk_tokens = config.index_topk
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            topk_tokens,
            dtype=torch.int32,
        )

        # Three aux streams shared across all DSpark layers.
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]

        # ── Backbone layers (mtp.0, mtp.1) + output layer (mtp.2) ────
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): DeepSeekV4DSparkLayer(
                    vllm_config,
                    self.topk_indices_buffer,
                    f"{prefix}.layers.{idx}",
                    aux_stream_list=aux_stream_list,
                    is_input_layer=(idx == self.mtp_start_layer_idx),
                    is_output_layer=(
                        idx == self.mtp_start_layer_idx + self.num_mtp_layers - 1
                    ),
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )

        # ── Embedding ─────────────────────────────────────────────────
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        # ── DSpark-specific heads ─────────────────────────────────────
        # Markov head W₁ — vocab-parallel embedding (vocab split across TP)
        self.markov_w1 = VocabParallelEmbedding(
            config.vocab_size,
            self.markov_rank,
            prefix=maybe_prefix(prefix, "markov_w1"),
        )
        # Markov head W₂ — column-parallel linear (output vocab split across TP)
        self.markov_w2 = ColumnParallelLinear(
            self.markov_rank,
            config.vocab_size,
            bias=False,
            prefix=maybe_prefix(prefix, "markov_w2"),
        )
        # Confidence head — replicated (output=1, same on all ranks)
        self.confidence_proj = ReplicatedLinear(
            config.hidden_size + self.markov_rank,
            1,
            bias=False,
            prefix=maybe_prefix(prefix, "confidence_proj"),
        )
        # Target-context projection lives on the input stage (mtp.0) as
        # main_proj + main_norm (see DeepSeekV4DSparkLayer). There is no
        # separate top-level ``fc`` — that was a fabricated weight. Accessed
        # via the ``input_layer`` property (no duplicate module registration).

        # ── Logits processor ──────────────────────────────────────────
        self.logits_processor = LogitsProcessor(config.vocab_size)

    @property
    def input_layer(self) -> "DeepSeekV4DSparkLayer":
        return self.layers[str(self.mtp_start_layer_idx)]

    # ═══════════════════════════════════════════════════════════════════
    # Forward pass (compatible with MTPSpeculator step-by-step calling)
    # ═══════════════════════════════════════════════════════════════════

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Minimal block run for warmup/dummy paths.

        DSpark's real proposal path is ``forward_dspark_block``; this method
        only needs to be runnable (e.g. for profiling/dummy runs). It embeds
        the input, expands to hc_mult streams, runs every stage, and returns
        the flattened output. Token embeddings feed the blocks directly — no
        input projection (matches the reference architecture).
        """
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds.unsqueeze(-2).expand(-1, self.config.hc_mult, -1)
        for layer_key in sorted(self.layers.keys(), key=int):
            hidden_states = self.layers[layer_key].run_block(positions, hidden_states)
        return hidden_states.flatten(1)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer_key = str(self.mtp_start_layer_idx + current_step_idx)
        mtp_layer = self.layers[layer_key]

        if not hasattr(mtp_layer, "hc_head_fn"):
            logger.error(
                "compute_logits called on non-output DSpark layer %s. "
                "DSpark requires logits only from the output layer (mtp.2).",
                layer_key,
            )
            return None

        hidden_states = hidden_states.view(
            -1, mtp_layer.hc_mult, mtp_layer.config.hidden_size
        )
        hidden_states = hc_head_fused_kernel_tilelang(
            hidden_states,
            mtp_layer.hc_head_fn,
            mtp_layer.hc_head_scale,
            mtp_layer.hc_head_base,
            mtp_layer.rms_norm_eps,
            mtp_layer.hc_eps,
        )
        hidden_states = mtp_shared_head_rmsnorm(
            hidden_states,
            mtp_layer.shared_head.norm.weight.data,
            mtp_layer.shared_head.norm.variance_epsilon,
        )
        logits = self.logits_processor(mtp_layer.shared_head.head, hidden_states)
        return logits

    # ═══════════════════════════════════════════════════════════════════
    # DSpark-specific methods (Phase 2+ — stubbed for now)
    # ═══════════════════════════════════════════════════════════════════

    def project_context(self, target_hidden_states: torch.Tensor) -> torch.Tensor:
        """Project concatenated target hidden states [*, 3·D] → [*, D].

        Uses the input stage's ``main_proj`` + ``main_norm`` (checkpoint
        ``mtp.0.main_proj`` / ``mtp.0.main_norm``).
        """
        return self.input_layer.project_main_context(target_hidden_states)

    def markov_bias(self, prev_token_ids: torch.Tensor) -> torch.Tensor:
        """Compute Markov transition bias for previous tokens.

        B(x_{k-1}, ·) = W₂(W₁[x_{k-1}])   ∈ ℝ^V

        Phase 2: integrate into sequential draft sampling loop.
        """
        prev_emb = self.markov_w1(prev_token_ids.long())
        return self.markov_w2(prev_emb)

    def confidence_score(
        self,
        hidden_states: torch.Tensor,
        markov_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-position confidence logits.

        cₖ = σ(wᵀ [hₖ; W₁[x_{k-1}]])

        Phase 2: integrate into draft block and verification scheduling.
        """
        features = torch.cat([hidden_states, markov_embeddings], dim=-1)
        return self.confidence_proj(features)

    # ═══════════════════════════════════════════════════════════════════
    # DSpark block generation (Phase 2)
    # ═══════════════════════════════════════════════════════════════════

    def forward_dspark_block(
        self,
        draft_input_ids: torch.Tensor,  # [B*γ]
        draft_positions: torch.Tensor,  # [B*γ]
        anchor_token_ids: torch.Tensor,  # [B]
        target_context: torch.Tensor,  # [B, 3 * hidden_size]
        temperature: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Full DSpark block generation (called within forward context).

        The speculator must set up ``set_forward_context`` with
        ``causal=False`` attention metadata before calling this.

        This method:
          1. Projects target context via main_proj+main_norm (main_x)
          2. Embeds draft tokens and runs all stages on the flat [B*γ] seq
          3. Computes base logits via hc_head
          4. Runs Markov sequential sampling
          5. Computes confidence scores

        Returns dict with draft_tokens [B,γ], draft_logits [B,γ,V],
        confidence [B,γ].

        NOTE (fix C, pending): in the reference architecture the target
        context enters *only* through DSparkAttention cross-attention
        (per-layer ``main_kv`` written into the draft KV cache). That requires
        speculator/KV-cache changes and is not yet wired. As an INTERIM, the
        projected ``main_x`` is added to the anchor (position-0) embedding so
        some context flows; this is an approximation and will be replaced by
        cross-attention. See dspark/checkpoint_anatomy.md (fix C) and
        dspark/phase_cd_plan.md.
        """
        B = anchor_token_ids.size(0)
        gamma = self.block_size

        # 1. Project target context: [B, 3*D] → [B, D] via main_proj+main_norm.
        main_x = self.project_context(target_context)  # [B, hidden_size]

        # 2. Embed draft tokens: [B*γ, D]. Embeddings feed the blocks directly
        #    (no enorm/hnorm/e_proj/h_proj — those do not exist in DSpark).
        draft_embeds = self.embed_tokens(draft_input_ids)  # [B*γ, D]

        # 3. INTERIM context injection (see NOTE / fix C): add main_x to the
        #    anchor position embedding. Proper path is cross-attention KV.
        embeds_2d = draft_embeds.reshape(B, gamma, -1)  # [B, γ, D]
        embeds_2d[:, 0, :] = embeds_2d[:, 0, :] + main_x
        flat_embeds = embeds_2d.reshape(B * gamma, -1)  # [B*γ, D]

        # 4. Run all stages with 3D [*, hc_mult, D] input so the decoder
        #    block's mHC encoding produces genuine multi-stream hidden states
        #    (hc_mult=4), matching what the hc_head expects.
        output_key = str(self.mtp_start_layer_idx + self.num_mtp_layers - 1)
        output_layer = self.layers[output_key]
        hc_mult = output_layer.hc_mult
        hidden_states = flat_embeds.reshape(B * gamma, 1, -1).expand(
            -1, hc_mult, -1
        )  # [B*γ, hc_mult, D]
        for layer_key in sorted(self.layers.keys(), key=int):
            hidden_states = self.layers[layer_key].run_block(
                draft_positions, hidden_states
            )

        # 5. Compute base logits via hc_head on the output layer.
        #    hidden_states is already [B*γ, hc_mult, D] from the backbone.
        hc_output = hc_head_fused_kernel_tilelang(
            hidden_states,
            output_layer.hc_head_fn,
            output_layer.hc_head_scale,
            output_layer.hc_head_base,
            output_layer.rms_norm_eps,
            output_layer.hc_eps,
        )  # [B*γ, D]
        hc_output = mtp_shared_head_rmsnorm(
            hc_output,
            output_layer.shared_head.norm.weight.data,
            output_layer.shared_head.norm.variance_epsilon,
        )
        base_logits = self.logits_processor(
            output_layer.shared_head.head, hc_output
        ).reshape(B, gamma, -1)  # [B, γ, V]

        # Reshape hidden states for Markov/confidence — take first stream
        # from the mhc-encoded backbone output.
        hidden_3d = hidden_states.reshape(B, gamma, hc_mult, self.config.hidden_size)[
            :, :, 0, :
        ]  # [B, γ, D]

        # 6. Markov sequential sampling.
        draft_tokens_list = []
        draft_logits_list = []
        confidence_list = []
        prev = anchor_token_ids.long()  # [B]

        for k in range(gamma):
            prev_emb = self.markov_w1(prev)  # [B, rank]
            bias = self.markov_w2(prev_emb)  # [B, V]
            step_logits = base_logits[:, k, :] + bias  # [B, V]
            draft_logits_list.append(step_logits)

            if temperature == 0.0:
                next_token = step_logits.argmax(dim=-1)
            else:
                probs = torch.softmax(step_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            draft_tokens_list.append(next_token)

            h_k = hidden_3d[:, k, :]  # [B, D]
            c_k = self.confidence_score(h_k, prev_emb)  # [B, 1]
            confidence_list.append(c_k.squeeze(-1))

            prev = next_token

        draft_tokens = torch.stack(draft_tokens_list, dim=1)  # [B, γ]
        draft_logits = torch.stack(draft_logits_list, dim=1)  # [B, γ, V]
        confidence = torch.stack(confidence_list, dim=1)  # [B, γ]

        return {
            "draft_tokens": draft_tokens,
            "draft_logits": draft_logits,
            "confidence": confidence,
        }

    # ═══════════════════════════════════════════════════════════════════
    # Weight loading
    # ═══════════════════════════════════════════════════════════════════

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Weight name remapping for checkpoint compatibility.
        # Token embedding (top-level ``embed.weight``) and the tied LM head
        # (top-level ``head.weight``) are shared from the target model in
        # load_eagle_model, so they are not remapped here. The output stage's
        # pre-head norm ``mtp.2.norm.weight`` maps to ``shared_head.norm``.
        WEIGHT_NAME_REMAPPING: dict[str, str] = {
            ".norm.weight": ".shared_head.norm.weight",
            # DSpark-specific: checkpoint paths → model parameter paths
            ".markov_head.markov_w1.weight": ".markov_w1.weight",
            ".markov_head.markov_w2.weight": ".markov_w2.weight",
            ".confidence_head.proj.weight": ".confidence_proj.weight",
        }

        def _remap_weight_name(name: str) -> str:
            for old_pattern, new_pattern in WEIGHT_NAME_REMAPPING.items():
                if old_pattern in name:
                    # Guard: .norm.weight also appears as a substring of the
                    # decoder norms (attn_norm/ffn_norm/kv_norm/q_norm) and the
                    # input-stage main_norm — only remap the output norm.
                    if old_pattern == ".norm.weight" and (
                        "attn_norm" in name
                        or "ffn_norm" in name
                        or "kv_norm" in name
                        or "q_norm" in name
                        or "main_norm" in name
                    ):
                        continue
                    name = name.replace(old_pattern, new_pattern)
            return name

        def _find_mtp_layer_idx(name: str) -> int:
            subnames = name.split(".")
            for subname in subnames:
                try:
                    return int(subname)
                except ValueError:
                    continue
            return 0

        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # TP for attention
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        # Pre-compute expert mapping ONCE.
        first_layer = next(iter(self.layers.values()))
        if first_layer.mtp_block.ffn.use_mega_moe:
            expert_mapping = make_deepseek_v4_expert_params_mapping(
                self.config.n_routed_experts
            )
        else:
            expert_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.n_routed_experts,
            )

        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )

        for name, loaded_weight in weights:
            mtp_layer_idx = _find_mtp_layer_idx(name)
            name = name.replace(
                f"mtp.{mtp_layer_idx}.",
                f"layers.{self.config.num_hidden_layers + mtp_layer_idx}.",
            )

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue

            name = _remap_weight_name(name)
            name = self._rewrite_spec_layer_name(spec_layer, name)

            # Top-level DSpark heads (promoted off the output stage) load
            # directly. Handle them before the stacked-params loop because
            # "markov_w1" collides with the "w1" stacked weight-name substring.
            if name in (
                "markov_w1.weight",
                "markov_w2.weight",
                "confidence_proj.weight",
            ):
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
                continue

            if spec_layer != self.mtp_start_layer_idx and "layers." not in name:
                continue
            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else ".weight_scale_inv"
                )
                name = name.removesuffix(".scale") + suffix
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        param = params_dict[name_mapped]
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            loaded_params.add(name_mapped)
                            break
                    continue
                elif "attn_sink" in name:
                    if name not in params_dict:
                        continue
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    if ".shared_experts.w2" in name:
                        name = name.replace(
                            ".shared_experts.w2", ".shared_experts.down_proj"
                        )
                    if name.endswith(".ffn.gate.bias"):
                        name = name.replace(
                            ".ffn.gate.bias",
                            ".ffn.gate.e_score_correction_bias",
                        )
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        loaded_layers: set[int] = set()
        for param_name in loaded_params:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, param_name)
            if spec_layer is not None:
                loaded_layers.add(spec_layer)
        for layer_idx in range(
            self.mtp_start_layer_idx,
            self.mtp_start_layer_idx + self.num_mtp_layers,
        ):
            if layer_idx not in loaded_layers:
                raise ValueError(
                    f"DSpark draft layer {layer_idx} weights "
                    f"missing from checkpoint (loaded={sorted(loaded_layers)}, "
                    f"n_predict={self.config.num_nextn_predict_layers}). "
                    f"The checkpoint may have been quantized without "
                    f"including the DSpark layers. "
                    f"Use a checkpoint that includes DSpark layer weights, "
                    f"or disable speculative decoding."
                )
        # Completeness check: every model parameter MUST receive a
        # checkpoint weight. A parameter left out here keeps its random
        # initialization and silently corrupts the draft — this is the
        # root cause of the observed 0% acceptance (the context projection
        # and any fabricated input projections never get real weights).
        # Token embedding and the tied LM head are intentionally shared
        # from the target model (wired in load_eagle_model after this
        # call), so they are exempt.
        externally_shared = ("embed_tokens", "shared_head.head")
        unloaded = sorted(
            pname
            for pname in params_dict
            if pname not in loaded_params
            and not any(s in pname for s in externally_shared)
        )
        if unloaded:
            raise ValueError(
                "DSpark draft model has parameters with no checkpoint "
                "source; they would remain randomly initialized and "
                "produce ~0% draft acceptance. "
                f"{len(unloaded)} unloaded parameter(s): {unloaded}. "
                "This means the model definition does not match the "
                "checkpoint layout. See dspark/checkpoint_anatomy.md for "
                "the authoritative weight mapping (e.g. the context "
                "projection is mtp.0.main_proj + mtp.0.main_norm, not a "
                "separate `fc`; DSpark has no enorm/hnorm/e_proj/h_proj)."
            )

        self.finalize_mega_moe_weights()
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def finalize_mega_moe_weights(self) -> None:
        for layer in self.layers.values():
            layer.mtp_block.ffn.finalize_mega_moe_weights()

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """Rewrite weight name to match model parameter layout.

        Adds ``.mtp_block`` for decoder-block weights; promotes shared
        and DSpark-specific weights to top-level ``model.*``.
        """
        spec_layer_weight_names = [
            "embed_tokens",
            "shared_head",
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
            # DSpark-specific top-level weights
            "markov_w1",
            "markov_w2",
            "confidence_proj",
            # Per-stage weights outside mtp_block (input stage only)
            "main_norm",
            "main_proj",
        ]
        shared_weight_names = [
            "embed_tokens",
            "markov_w1",
            "markov_w2",
            "confidence_proj",
        ]
        spec_layer_weight = False
        shared_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            # Decoder-block weights go under layers.{idx}.mtp_block.*
            name = name.replace(
                f"layers.{spec_layer}.",
                f"layers.{spec_layer}.mtp_block.",
            )
        elif shared_weight:
            # Top-level shared weights (embed, Markov, confidence) live
            # directly on the inner model, not under layers.
            name = name.replace(f"layers.{spec_layer}.", "")
        # else: per-stage spec weights (shared_head, hc_head_*, main_proj,
        # main_norm) stay under layers.{idx}.* unchanged.
        return name


class DeepSeekV4DSparkModel(nn.Module):
    """DSpark speculative decoding draft model — vLLM wrapper.

    Thin wrapper around ``DSparkInnerModel`` for vLLM compatibility.
    ``load_eagle_model()`` expects a ``.model`` attribute pointing to
    the inner model (for embedding sharing and topk_indices_buffer).

    Delegates forward, compute_logits, and weight loading to the
    inner model.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        inner_prefix = maybe_prefix(prefix, "model")
        self.model = DSparkInnerModel(vllm_config=vllm_config, prefix=inner_prefix)
        self.config = self.model.config

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def forward_dspark_block(self, **kwargs: typing.Any) -> dict[str, torch.Tensor]:
        return self.model.forward_dspark_block(**kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return self.model.load_weights(weights)

    def finalize_mega_moe_weights(self) -> None:
        self.model.finalize_mega_moe_weights()

    # ── Accessors needed by load_eagle_model ─────────────────────────

    @property
    def block_size(self) -> int:
        return self.model.block_size

    @property
    def noise_token_id(self) -> int:
        return self.model.noise_token_id

    @property
    def layers(self) -> nn.ModuleDict:
        return self.model.layers

    @property
    def topk_indices_buffer(self) -> torch.Tensor:
        return self.model.topk_indices_buffer

    @topk_indices_buffer.setter
    def topk_indices_buffer(self, value: torch.Tensor) -> None:
        self.model.topk_indices_buffer = value

    @property
    def embed_tokens(self) -> nn.Module:
        return self.model.embed_tokens

    @embed_tokens.setter
    def embed_tokens(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    @property
    def mtp_start_layer_idx(self) -> int:
        return self.model.mtp_start_layer_idx

    @property
    def num_mtp_layers(self) -> int:
        return self.model.num_mtp_layers
