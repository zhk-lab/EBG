"""Common interface and deterministic packing for evidence backends."""

from __future__ import annotations

import zlib
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol

from .evidence import EvidenceUnit, ToolResult, render_tool_result
from .errors import ContextUnfitError


TokenCounter = Callable[[str], int]


class EvidenceBackend(Protocol):
    input_id: str
    benchmark: str
    kind: str
    initial_index: str
    initial_index_tokens: int

    @property
    def resume_identity(self) -> dict[str, Any]: ...

    @property
    def initial_ids(self) -> frozenset[str]: ...

    @property
    def priority_groups(self) -> dict[str, tuple[str, ...]]: ...

    def has_id(self, unit_id: str) -> bool: ...

    def search(
        self,
        text: str,
        *,
        limit: int,
        exact_path_limit: int | None = None,
    ) -> ToolResult: ...

    def read(
        self,
        ids: Sequence[str],
        *,
        token_budget: int,
        max_atomic_unit_tokens: int,
        count_tokens: TokenCounter,
    ) -> ToolResult: ...


def content_identity(parts: Iterable[str]) -> dict[str, Any]:
    """Return a compact change detector for trusted resumable inputs."""

    checksum = 0
    total_bytes = 0
    part_count = 0
    for part in parts:
        encoded = part.encode("utf-8")
        length = len(encoded).to_bytes(8, "big")
        checksum = zlib.crc32(length, checksum)
        checksum = zlib.crc32(encoded, checksum)
        total_bytes += len(encoded)
        part_count += 1
    return {
        "format": "crc32-length-v1",
        "parts": part_count,
        "bytes": total_bytes,
        "checksum": f"{checksum & 0xFFFFFFFF:08x}",
    }


def pack_read_result(
    requested_ids: Sequence[str],
    units_by_id: dict[str, EvidenceUnit],
    *,
    token_budget: int,
    max_atomic_unit_tokens: int,
    count_tokens: TokenCounter,
) -> ToolResult:
    """Keep the longest complete request prefix that fits the soft budget."""

    chosen: list[EvidenceUnit] = []
    for unit_id in requested_ids:
        proposed = [*chosen, units_by_id[unit_id]]
        candidate = ToolResult(
            action="read",
            status="ok",
            requested_ids=tuple(requested_ids),
            units=tuple(proposed),
            deferred_ids=tuple(requested_ids[len(proposed) :]),
        )
        if count_tokens(render_tool_result(candidate)) <= token_budget:
            chosen = proposed
            continue
        if not chosen:
            atomic_tokens = count_tokens(render_tool_result(candidate))
            if atomic_tokens > max_atomic_unit_tokens:
                raise ContextUnfitError(
                    f"complete unit {unit_id} requires {atomic_tokens} tokens, "
                    f"above the frozen atomic limit {max_atomic_unit_tokens}"
                )
            chosen = proposed
        break
    deferred = tuple(requested_ids[len(chosen) :])
    result = ToolResult(
        action="read",
        status="ok",
        requested_ids=tuple(requested_ids),
        units=tuple(chosen),
        deferred_ids=deferred,
    )
    oversize = bool(
        chosen
        and count_tokens(render_tool_result(result)) > token_budget
    )
    return ToolResult(
        action=result.action,
        status=result.status,
        requested_ids=result.requested_ids,
        units=result.units,
        deferred_ids=result.deferred_ids,
        oversize_unit=oversize,
    )
