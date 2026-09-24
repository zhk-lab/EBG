"""Errors raised by the shared repository-audit agent loop."""

from __future__ import annotations

from typing import Any

from evaluation_core.contracts import (
    PredictionError,
    PredictionFormatError,
    PredictionGroundingError,
)


class AgentLoopError(ValueError):
    """Base error for deterministic AgentLoop failures."""


class ProtocolError(AgentLoopError):
    """A model response does not follow the action protocol."""


class ActionParameterError(AgentLoopError):
    """A known action contains invalid tool parameters."""

    def __init__(self, message: str, *, action: str) -> None:
        super().__init__(message)
        self.action = action


class BackendError(AgentLoopError):
    """An evidence backend or its input is invalid."""


class ContextUnfitError(AgentLoopError):
    """Required fixed content and one complete unit do not fit the context."""


class RetryableModelError(RuntimeError):
    """A transient provider failure that may be retried byte-for-byte."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = dict(raw_response) if raw_response is not None else None
        self.usage = dict(usage) if usage is not None else {}


class ProviderContextError(RuntimeError):
    """The provider rejected a locally valid request as too long."""
