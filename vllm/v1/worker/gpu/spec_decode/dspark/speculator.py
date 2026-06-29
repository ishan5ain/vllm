# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculative decoding speculator.

DSpark produces all γ draft tokens in one forward pass using a
semi-autoregressive head (Markov W₁W₂) with bidirectional attention
within the draft block.

Unlike MTP/Eagle (step-by-step), DSpark calls the draft model once per
target step. The setup follows the DFlash pattern for KV cache and
attention metadata, but DSpark injects the target context via hidden
state addition rather than KV cache precomputation.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

logger = init_logger(__name__)

# Env-gated diagnostics (DSPARK_DEBUG=1) for the 0%-acceptance investigation.
_DSPARK_DEBUG = os.environ.get("DSPARK_DEBUG", "0") == "1"
_DSPARK_DEBUG_CALLS = int(os.environ.get("DSPARK_DEBUG_CALLS", "3"))
# File sink: vLLM is often launched on a pty not captured by `docker logs`.
_DSPARK_DEBUG_FILE = os.environ.get("DSPARK_DEBUG_FILE", "/tmp/dspark_debug.log")
# Runtime toggle file (env vars don't reliably reach the worker subprocess):
# enable with `docker exec <container> touch /tmp/dspark_debug_on` — no restart.
_DSPARK_DEBUG_TOGGLE = os.environ.get("DSPARK_DEBUG_TOGGLE", "/tmp/dspark_debug_on")
_dspark_dbg_count = 0


def _dspark_dbg_enabled() -> bool:
    if _DSPARK_DEBUG:
        return True
    try:
        return os.path.exists(_DSPARK_DEBUG_TOGGLE)
    except OSError:
        return False


def _dspark_dbg_should_log() -> bool:
    global _dspark_dbg_count
    if not _dspark_dbg_enabled() or _dspark_dbg_count >= _DSPARK_DEBUG_CALLS:
        return False
    _dspark_dbg_count += 1
    return True


def _dspark_dbg_emit(msg: str) -> None:
    logger.info("%s", msg)
    try:
        with open(_DSPARK_DEBUG_FILE, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


class DSparkSpeculator(DraftModelSpeculator):
    """Speculator for DSpark block draft generation.

    Requires the target model to expose ``get_dspark_context_hidden_states()``
    which returns concatenated hidden states from target layers 40, 41, 42
    as a tensor of shape [T, 3 * hidden_size].

    Phase 2 limitations:
    - Eager mode only (no CUDA graphs).
    - The DSpark context is passed via ``aux_hidden_states[0]``.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, device)

        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )

        self.supports_mm_inputs = False

        # Each request produces exactly γ (= num_speculative_steps) tokens
        # in one forward pass.
        self.num_query_per_req = self.num_speculative_steps

        # No CUDA graphs for Phase 2.
        self.query_cudagraph_manager = None
        self.draft_kv_cache_group_id: int = -1
        self.draft_block_size: int = -1

    # ═══════════════════════════════════════════════════════════════════
    # CUDA graph stubs (Phase 2: eager only)
    # ═══════════════════════════════════════════════════════════════════

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        logger.info_once(
            "DSpark speculator: CUDA graphs not yet supported. Running in eager mode."
        )

    def capture(self, *args: Any, **kwargs: Any) -> None:
        pass

    # ═══════════════════════════════════════════════════════════════════
    # Model loading
    # ═══════════════════════════════════════════════════════════════════

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        """Load DSpark draft model via the standard eagle model loader.

        Uses ``load_eagle_model()`` which wires up embedding sharing
        and topk_indices_buffer between target and draft models.
        """
        from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model

        self._target_model = target_model
        return load_eagle_model(target_model, self.vllm_config)

    @property
    def model_returns_tuple(self) -> bool:
        return False

    @property
    def advance_draft_positions(self) -> bool:
        return True

    # ═══════════════════════════════════════════════════════════════════
    # Attention setup (KV cache, slot mappings)
    # ═══════════════════════════════════════════════════════════════════

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        super().set_attn(model_state, kv_cache_config, block_tables)

        # DSpark draft layers share a single KV cache group.
        draft_groups = [gid for gid, g in enumerate(self.attn_groups) if g]
        if draft_groups:
            self.draft_kv_cache_group_id = draft_groups[0]
            self.draft_block_size = self.block_tables.block_sizes[
                self.draft_kv_cache_group_id
            ]
        else:
            logger.warning(
                "DSpark speculator: no draft attention groups found. "
                "The draft model may not have its own attention layers."
            )

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
    ) -> dict[str, Any] | None:
        """Build attention metadata with causal=False for bidirectional
        attention within the DSpark draft block."""
        return super()._build_draft_attn_metadata(
            num_reqs,
            num_reqs_padded,
            num_tokens_padded,
            num_query_per_req=self.num_speculative_steps,
            causal=False,  # DSpark requires bidirectional within the block
        )

    # ═══════════════════════════════════════════════════════════════════
    # Draft model forward (wrapped in attention context)
    # ═══════════════════════════════════════════════════════════════════

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None = None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            last_hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=None,
            )
        return last_hidden_states

    # ═══════════════════════════════════════════════════════════════════
    # Proposal (main entry point)
    # ═══════════════════════════════════════════════════════════════════

    def _get_anchor_data(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract anchor token IDs and positions for each request.

        Follows the same logic as the DFlash kernel: uses last_sampled
        for decode requests and next_prefill_tokens for chunked prefills.

        Returns:
            anchor_tokens: [num_reqs] int64 on device
            anchor_positions: [num_reqs] int64 on device
            anchor_indices: [num_reqs] int64 on device (indices into
                the target's token buffer for context lookup)
        """
        num_reqs = input_batch.num_reqs
        device = self.device

        # Build anchor tokens per request.
        anchor_tokens = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        anchor_positions = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        anchor_indices = torch.zeros(num_reqs, dtype=torch.int64, device=device)

        query_start_loc = input_batch.query_start_loc
        positions = input_batch.positions

        for req_idx in range(num_reqs):
            qe = int(query_start_loc[req_idx + 1].item())
            rejected = int(num_rejected[req_idx].item())
            valid_end = qe - rejected
            anchor_idx = valid_end - 1
            anchor_indices[req_idx] = anchor_idx
            anchor_positions[req_idx] = positions[anchor_idx]

            sampled = int(num_sampled[req_idx].item())
            if sampled > 0:
                anchor_tokens[req_idx] = last_sampled[req_idx]
            else:
                anchor_tokens[req_idx] = next_prefill_tokens[req_idx]

        return anchor_tokens, anchor_positions, anchor_indices

    def _prepare_dspark_inputs(
        self,
        input_batch: InputBatch,
        anchor_tokens: torch.Tensor,  # [num_reqs]
        anchor_positions: torch.Tensor,  # [num_reqs]
    ) -> None:
        """Populate ``input_buffers`` with DSpark draft input.

        draft_input_ids  = [anchor, mask, mask, mask, mask] per request
        draft_positions   = [anchor_pos+1, ..., anchor_pos+γ] per request
        draft_query_start = [0, γ, 2γ, ..., R*γ]
        draft_seq_lens    = anchor_pos + γ (absolute sequence length)
        """
        num_reqs = input_batch.num_reqs
        gamma = self.num_speculative_steps

        # DSpark noise token ID from model config.
        noise_token_id = getattr(self.model, "noise_token_id", 128799)

        ib = self.input_buffers
        ib.input_ids.zero_()
        ib.positions.zero_()
        ib.query_start_loc.zero_()
        ib.seq_lens.zero_()

        for req_idx in range(num_reqs):
            q_base = req_idx * gamma
            # input_ids: [anchor, mask, mask, mask, mask]
            ib.input_ids[q_base] = anchor_tokens[req_idx]
            for k in range(1, gamma):
                ib.input_ids[q_base + k] = noise_token_id
            # positions: anchor_pos+1, ..., anchor_pos+γ
            for k in range(gamma):
                ib.positions[q_base + k] = anchor_positions[req_idx] + 1 + k
            # seq_lens: absolute sequence length for attention
            ib.seq_lens[req_idx] = anchor_positions[req_idx] + gamma + 1

        # query_start_loc: [0, γ, 2γ, ..., R*γ]
        for req_idx in range(num_reqs + 1):
            ib.query_start_loc[req_idx] = req_idx * gamma

        # Pad to max_num_reqs for CUDA graph safety.
        last_end = num_reqs * gamma
        for i in range(num_reqs + 1, self.max_num_reqs + 1):
            ib.query_start_loc[i] = last_end
        for i in range(num_reqs, self.max_num_reqs):
            ib.seq_lens[i] = 0

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        gamma = self.num_speculative_steps
        num_query_tokens = num_reqs * gamma
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(max_seq_len + gamma, self.max_model_len)

        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )

        # 1. Extract anchor tokens and positions.
        anchor_tokens, anchor_positions, anchor_indices = self._get_anchor_data(
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
        )

        # 2. Get DSpark context (target layers 40, 41, 42 concatenated).
        target_context_all = None
        if aux_hidden_states is not None and len(aux_hidden_states) > 0:
            target_context_all = aux_hidden_states[0]  # [T, 3*D]
        if target_context_all is None:
            target_model = getattr(self, "_target_model", None)
            if target_model is not None and hasattr(
                target_model, "get_dspark_context_hidden_states"
            ):
                ctx_buf = target_model.get_dspark_context_hidden_states()
                if ctx_buf is not None:
                    target_context_all = ctx_buf
        if target_context_all is None:
            # No DSpark context available — return empty draft tokens.
            # This should not happen in normal operation; the target model
            # always runs forward() before the speculator is called.
            logger.warning_once(
                "DSpark speculator: no target context available. "
                "Returning empty draft tokens."
            )
            return torch.zeros(
                self.max_num_reqs, gamma, dtype=torch.int64, device=self.device
            )

        # Slice context at anchor positions.
        anchor_context = target_context_all[anchor_indices]  # [B, 3*D]

        # 3. Prepare draft inputs.
        self._prepare_dspark_inputs(input_batch, anchor_tokens, anchor_positions)

        # 4. Build draft attention metadata (causal=False for bidirectional).
        draft_attn_metadata = self._build_draft_attn_metadata(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs,
            num_tokens_padded=num_query_tokens,
        )

        # 5. Build slot mappings for draft layers.
        draft_slot_mappings = build_slot_mappings_by_layer(
            self.block_tables.slot_mappings[:, :num_query_tokens],
            self.kv_cache_config,
        )

        # 6. Call the DSpark model through the attention context.
        #    _run_model sets up forward context (causal=False) and calls
        #    model.forward(input_ids, positions). But DSpark's backbone
        #    needs to run differently — all layers on the full block.
        #    So we call forward_dspark_block directly, wrapped in the
        #    same forward context.
        batch_descriptor = BatchDescriptor(num_tokens=num_query_tokens)
        with set_forward_context(
            draft_attn_metadata,
            self.vllm_config,
            num_tokens=num_query_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=draft_slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            result = self.model.forward_dspark_block(
                draft_input_ids=self.input_buffers.input_ids[:num_query_tokens],
                draft_positions=self.input_buffers.positions[:num_query_tokens],
                anchor_token_ids=anchor_tokens,
                target_context=anchor_context,
                temperature=0.0,
            )

        draft_tokens = result["draft_tokens"]  # [B, γ]

        if _dspark_dbg_should_log():
            with torch.no_grad():
                ctx = anchor_context.float()
                _dspark_dbg_emit(
                    f"DSPARK_DEBUG propose: num_reqs={num_reqs} gamma={gamma} "
                    f"anchor_tokens={anchor_tokens[:4].tolist()} "
                    f"anchor_positions={anchor_positions[:4].tolist()} "
                    f"anchor_indices={anchor_indices[:4].tolist()} "
                    f"ctx_shape={tuple(anchor_context.shape)} "
                    f"ctx_norm={float(ctx.norm()):.3f} "
                    f"ctx_nan={bool(torch.isnan(ctx).any())} "
                    f"draft_tokens[0]={draft_tokens[0].tolist()}"
                )

        # Pad to [max_num_reqs, γ].
        padded = torch.zeros(
            self.max_num_reqs, gamma, dtype=torch.int64, device=self.device
        )
        padded[:num_reqs, :] = draft_tokens[:, :gamma]
        self.draft_tokens[:num_reqs, :] = padded[:num_reqs, :]

        return padded
