"""Frozen protocol limits shared by Raw and Graph evaluation arms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .errors import AgentLoopError


@dataclass(frozen=True, slots=True)
class AgentLoopConfig:
    """Limits chosen from the 200 visible Repo samples and formal runners."""

    token_estimator: str = "utf8_bytes_div3_v1"
    context_window: int = 1_000_000
    max_output_tokens: int = 32_768
    provider_safety_margin: int = 32_768
    working_prompt_trigger: int = 131_072
    compression_target: int = 98_304
    index_budget: int = 3_072
    tool_result_budget: int = 32_768
    max_atomic_unit_tokens: int = 65_536
    max_search_hits: int = 12
    max_exact_path_hits: int = 12
    max_search_characters: int = 512
    max_read_ids: int = 6
    preferred_recent_reads: int = 1
    network_retries: int = 2
    format_repair_attempts: int = 2

    def __post_init__(self) -> None:
        positive = {
            name: value
            for name, value in asdict(self).items()
            if name not in {
                "token_estimator",
                "network_retries",
                "format_repair_attempts",
            }
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise AgentLoopError(f"configuration value must be positive: {invalid[0]}")
        if self.token_estimator != "utf8_bytes_div3_v1":
            raise AgentLoopError("unsupported token_estimator")
        if self.network_retries < 0:
            raise AgentLoopError("network_retries must be non-negative")
        if self.format_repair_attempts < 0:
            raise AgentLoopError("format_repair_attempts must be non-negative")
        if self.compression_target >= self.working_prompt_trigger:
            raise AgentLoopError(
                "compression_target must be below working_prompt_trigger"
            )
        if self.working_prompt_trigger >= self.physical_hard_limit:
            raise AgentLoopError(
                "working_prompt_trigger must be below physical_hard_limit"
            )
        if self.max_exact_path_hits < self.max_search_hits:
            raise AgentLoopError(
                "max_exact_path_hits cannot be below max_search_hits"
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


DEFAULT_CONFIG = AgentLoopConfig()
