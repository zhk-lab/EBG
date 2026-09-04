"""Generic validation boundary for terminal AgentLoop predictions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol

from .errors import AgentLoopError
from .evidence import EvidenceSpan


class FinishContract(Protocol):
    """Validate one terminal prediction against an injected formal contract."""

    @property
    def resume_identity(self) -> dict[str, Any]: ...

    def validate(
        self,
        prediction: dict[str, Any],
        *,
        input_id: str,
        observed_spans: Sequence[EvidenceSpan],
    ) -> dict[str, Any]: ...


def validated_finish_contract_identity(
    contract: FinishContract,
) -> dict[str, Any]:
    """Return a detached JSON identity suitable for resumable state."""

    identity = contract.resume_identity
    if not isinstance(identity, dict) or not identity:
        raise AgentLoopError(
            "finish contract must expose a non-empty resume identity"
        )
    try:
        serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True)
        result = json.loads(serialized)
    except (TypeError, ValueError) as error:
        raise AgentLoopError(
            "finish contract identity must be JSON-serializable"
        ) from error
    if not isinstance(result, dict):
        raise AgentLoopError("finish contract identity must be an object")
    return result
