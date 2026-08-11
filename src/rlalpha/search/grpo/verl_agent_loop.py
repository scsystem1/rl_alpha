from __future__ import annotations

from typing import Any

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from vllm.sampling_params import StructuredOutputsParams

from ..prompts import DSL_GRAMMAR


class RLAlphaStructuredSingleTurnAgentLoop(SingleTurnAgentLoop):
    """QuantEvolver/Verl single-turn rollout with the RLAlpha DSL grammar.

    This is a rollout adapter only.  Generation, log probabilities and policy
    optimization remain owned by Verl and vLLM.
    """

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        constrained = dict(sampling_params)
        # Verl calls vLLM's low-level Python server rather than its OpenAI
        # request parser.  At this boundary vLLM does not coerce dictionaries
        # and SamplingParams.verify() requires the native dataclass.
        constrained["structured_outputs"] = StructuredOutputsParams(grammar=DSL_GRAMMAR)
        return await super().run(constrained, **kwargs)
