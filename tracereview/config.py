"""Frozen limits for the one-shot TraceReview evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from agentloop.errors import AgentLoopError


@dataclass(frozen=True, slots=True)
class TraceReviewConfig:
    """Only limits that apply to a lossless, non-interactive trace request."""

    token_estimator: str = "utf8_bytes_div3_v1"
    context_window: int = 1_000_000
    max_output_tokens: int = 32_768
    provider_safety_margin: int = 32_768
    network_retries: int = 2

    def __post_init__(self) -> None:
        if self.token_estimator != "utf8_bytes_div3_v1":
            raise AgentLoopError("TraceReview token_estimator is unsupported")
        for name in (
            "context_window",
            "max_output_tokens",
            "provider_safety_margin",
        ):
            if getattr(self, name) <= 0:
                raise AgentLoopError(
                    f"TraceReview configuration value must be positive: {name}"
                )
        if self.network_retries < 0:
            raise AgentLoopError(
                "TraceReview network_retries must be non-negative"
            )
        if self.physical_hard_limit <= 0:
            raise AgentLoopError(
                "TraceReview output and safety margins exceed context window"
            )

    @property
    def physical_hard_limit(self) -> int:
        return (
            self.context_window
            - self.max_output_tokens
            - self.provider_safety_margin
        )

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["physical_hard_limit"] = self.physical_hard_limit
        return value


DEFAULT_CONFIG = TraceReviewConfig()
