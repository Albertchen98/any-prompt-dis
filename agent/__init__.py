"""VLM-grounded crop-then-segment agent on top of FlowDIS.

A local Qwen3.5-VL model reasons about which object the user means (from a complex
disambiguating prompt or a clicked point), grounds it to a bounding box; we crop that
region, run FlowDIS on the clean crop, then paste the mask back into a full-image mask.
"""

from agent.grounding import GroundedObject, GroundingParseError

__all__ = ["GroundedObject", "GroundingParseError"]
