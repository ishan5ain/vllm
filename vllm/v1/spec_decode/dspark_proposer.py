# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark proposer — thin wrapper around DSparkSpeculator.

Extends ``SpecDecodeBaseProposer`` (like ``EagleProposer``) so the
model_runner can call ``self.drafter.propose()`` with the standard
interface. Internally delegates to ``DSparkSpeculator``.
"""

import torch

from vllm.config import VllmConfig
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


class DSparkProposer(SpecDecodeBaseProposer):
    """Proposer adapter for DSpark speculative decoding."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config,
            device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
