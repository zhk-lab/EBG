"""Canonical message reconstruction and deterministic receipt compression."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from .config import AgentLoopConfig
from .errors import ContextUnfitError
from .evidence import ToolResult, render_tool_result


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    turn: int
    raw_response: str
    action: dict[str, Any]
    tool_result: ToolResult


@dataclass(frozen=True, slots=True)
class CompressionManifest:
    tokens_before: int
    tokens_after: int
    working_compression_triggered: bool
    search_receipt_turns: tuple[int, ...]
    read_receipt_units: tuple[tuple[int, str], ...]
    duplicate_receipt_units: tuple[tuple[int, str], ...]
    protected_oversize: bool
    forced: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "working_compression_triggered": self.working_compression_triggered,
            "search_receipt_turns": list(self.search_receipt_turns),
            "read_receipt_units": [list(item) for item in self.read_receipt_units],
            "duplicate_receipt_units": [
                list(item) for item in self.duplicate_receipt_units
            ],
            "protected_oversize": self.protected_oversize,
            "forced": self.forced,
        }


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    messages: tuple[dict[str, str], ...]
    token_count: int
    manifest: CompressionManifest


class FastTokenCounter:
    """Conservatively estimate tokens from UTF-8 bytes without dependencies."""

    ESTIMATOR = "utf8_bytes_div3_v1"

    def __init__(self, estimator: str = ESTIMATOR) -> None:
        if estimator != self.ESTIMATOR:
            raise ContextUnfitError(f"unknown token estimator: {estimator}")

    def count_text(self, text: str) -> int:
        if not isinstance(text, str):
            raise ContextUnfitError("token estimator input must be text")
        return (len(text.encode("utf-8")) + 2) // 3

    def count_messages(self, messages: Sequence[dict[str, str]]) -> int:
        serialized = json.dumps(
            {"messages": list(messages)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return self.count_text(serialized)


def prepare_messages(
    *,
    system: str,
    initial_user: str,
    history: Sequence[HistoryEntry],
    max_rounds: int,
    config: AgentLoopConfig,
    counter: FastTokenCounter,
    force_compression: bool = False,
    priority_groups: dict[str, tuple[str, ...]] | None = None,
) -> PreparedRequest:
    """Rebuild one request from canonical history; never mutate stored results."""

    all_full = {
        entry.turn: {unit.unit_id for unit in entry.tool_result.units}
        for entry in history
    }
    selected_search_ids = _selected_search_ids(history)
    full_search = {entry.turn: True for entry in history}
    messages = _messages(
        system,
        initial_user,
        history,
        max_rounds,
        all_full,
        full_search,
        selected_search_ids,
        priority_groups or {},
        compression_state="none",
    )
    before = counter.count_messages(messages)

    triggered = force_compression or before >= config.working_prompt_trigger
    full_units = {turn: set(ids) for turn, ids in all_full.items()}
    duplicate_receipts: set[tuple[int, str]] = set()
    if triggered:
        duplicate_receipts = _deduplicate_read_units(history, full_units)
    search_receipts: set[int] = set()
    read_receipts: set[tuple[int, str]] = set()
    lossless = _messages(
        system,
        initial_user,
        history,
        max_rounds,
        full_units,
        full_search,
        selected_search_ids,
        priority_groups or {},
        compression_state=("deduplicated" if duplicate_receipts else "none"),
    )
    current = counter.count_messages(lossless)
    if triggered:
        latest_turn = history[-1].turn if history else -1
        compressible_searches = _compressible_search_turns(
            history, selected_search_ids
        )
        for entry in history:
            if (
                entry.turn == latest_turn
                or entry.tool_result.action != "search"
                or not full_search.get(entry.turn)
                or entry.turn not in compressible_searches
            ):
                continue
            full_search[entry.turn] = False
            search_receipts.add(entry.turn)
            current = _count_current(
                system,
                initial_user,
                history,
                max_rounds,
                full_units,
                full_search,
                counter,
                priority_groups or {},
            )
            if current <= config.compression_target:
                break

    final_messages = _messages(
        system,
        initial_user,
        history,
        max_rounds,
        full_units,
        full_search,
        selected_search_ids,
        priority_groups or {},
        compression_state=(
            "compressed"
            if triggered
            else ("deduplicated" if duplicate_receipts else "none")
        ),
    )
    after = counter.count_messages(final_messages)
    if after > config.physical_hard_limit:
        raise ContextUnfitError(
            f"required request uses {after} tokens, above physical hard limit "
            f"{config.physical_hard_limit}"
        )
    manifest = CompressionManifest(
        tokens_before=before,
        tokens_after=after,
        working_compression_triggered=triggered,
        search_receipt_turns=tuple(sorted(search_receipts)),
        read_receipt_units=tuple(sorted(read_receipts)),
        duplicate_receipt_units=tuple(sorted(duplicate_receipts)),
        protected_oversize=(triggered and after > config.compression_target),
        forced=force_compression,
    )
    return PreparedRequest(tuple(final_messages), after, manifest)


def _deduplicate_read_units(
    history: Sequence[HistoryEntry], full_units: dict[int, set[str]]
) -> set[tuple[int, str]]:
    latest_by_id: dict[str, int] = {}
    latest_by_source: dict[tuple[Any, ...], int] = {}
    duplicates: set[tuple[int, str]] = set()
    for entry in reversed(history):
        for unit in reversed(entry.tool_result.units):
            latest_source_turn = latest_by_source.get(unit.source_key)
            duplicate = unit.unit_id in latest_by_id or (
                latest_source_turn is not None
                and latest_source_turn != entry.turn
            )
            if duplicate:
                full_units[entry.turn].discard(unit.unit_id)
                duplicates.add((entry.turn, unit.unit_id))
            else:
                latest_by_id[unit.unit_id] = entry.turn
                latest_by_source[unit.source_key] = entry.turn
    return duplicates


def _count_current(
    system: str,
    initial_user: str,
    history: Sequence[HistoryEntry],
    max_rounds: int,
    full_units: dict[int, set[str]],
    full_search: dict[int, bool],
    counter: FastTokenCounter,
    priority_groups: dict[str, tuple[str, ...]],
) -> int:
    return counter.count_messages(
        _messages(
            system,
            initial_user,
            history,
            max_rounds,
            full_units,
            full_search,
            _selected_search_ids(history),
            priority_groups,
            compression_state="compressed",
        )
    )


def _messages(
    system: str,
    initial_user: str,
    history: Sequence[HistoryEntry],
    max_rounds: int,
    full_units: dict[int, set[str]],
    full_search: dict[int, bool],
    selected_search_ids: dict[int, set[str]],
    priority_groups: dict[str, tuple[str, ...]],
    *,
    compression_state: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": initial_user},
    ]
    shown_documents: set[tuple[str, str, str]] = set()
    for entry in history:
        messages.append({"role": "assistant", "content": entry.raw_response})
        messages.append(
            {
                "role": "user",
                "content": render_tool_result(
                    entry.tool_result,
                    full_unit_ids=full_units.get(entry.turn, set()),
                    full_search=full_search.get(entry.turn, True),
                    selected_search_ids=selected_search_ids.get(entry.turn, set()),
                    shown_document_keys=shown_documents,
                ),
            }
        )
    run_state = _run_state(
        history,
        max_rounds,
        compression_state,
        priority_groups,
    )
    messages[-1] = {
        "role": "user",
        "content": messages[-1]["content"] + "\n\n" + run_state,
    }
    return messages


def _run_state(
    history: Sequence[HistoryEntry],
    max_rounds: int,
    compression_state: str,
    priority_groups: dict[str, tuple[str, ...]],
) -> str:
    del priority_groups
    remaining = max_rounds - len(history)
    read_ids: list[str] = []
    deferred: list[str] = []
    for entry in history:
        result = entry.tool_result
        if result.action == "read" and result.status == "ok":
            for unit_id in result.returned_ids:
                if unit_id not in read_ids:
                    read_ids.append(unit_id)
            deferred = [item for item in deferred if item not in result.returned_ids]
            for unit_id in result.deferred_ids:
                if unit_id not in deferred:
                    deferred.append(unit_id)
    lines = [
        "[[RUN STATE]]",
        f"Must finish now: {'yes' if remaining == 1 else 'no'}",
        (
            "Allowed action this turn: finish only. Output one bare JSON object."
            if remaining == 1
            else "Allowed action this turn: choose exactly one of search, read, or "
            "finish. Output one bare JSON object, then stop."
        ),
        "Pending deferred IDs: " + (", ".join(deferred) if deferred else "none"),
        f"Compression state: {compression_state}",
        f"Successfully read repository units: {len(read_ids)}",
        "Unread index entries are optional, not a checklist. Before finish, compare "
        "every already-read source block with the full task document; do not stop "
        "after the first supported result.",
    ]
    if remaining <= 2:
        lines.append(f"Remaining rounds including finish: {remaining}")
    if remaining == 2:
        lines.append(
            "This is the last evidence-action round. Finish now if evidence is "
            "sufficient; otherwise request only one specifically missing new unit."
        )
    lines.extend(("[[END RUN STATE]]", f"Round {len(history) + 1}"))
    return "\n".join(lines)


def _selected_search_ids(
    history: Sequence[HistoryEntry],
) -> dict[int, set[str]]:
    later_reads: set[str] = set()
    selected: dict[int, set[str]] = {}
    for entry in reversed(history):
        if entry.tool_result.action == "search":
            hit_ids = {hit.unit_id for hit in entry.tool_result.hits}
            selected[entry.turn] = hit_ids & later_reads
        elif entry.tool_result.action == "read" and entry.tool_result.status == "ok":
            later_reads.update(entry.tool_result.returned_ids)
    return selected


def _compressible_search_turns(
    history: Sequence[HistoryEntry],
    selected_ids: dict[int, set[str]],
) -> set[int]:
    """Return old searches whose navigation value is preserved elsewhere."""

    latest_turn_by_result: dict[tuple[Any, ...], int] = {}
    result_keys: dict[int, tuple[Any, ...]] = {}
    for entry in history:
        if entry.tool_result.action != "search":
            continue
        result = entry.tool_result
        key = (
            result.search_text,
            result.total_matches,
            tuple(
                (
                    hit.unit_id,
                    hit.name,
                    hit.path,
                    hit.symbol,
                    hit.matched_field,
                )
                for hit in result.hits
            ),
            result.group_hints,
        )
        result_keys[entry.turn] = key
        latest_turn_by_result[key] = entry.turn

    compressible: set[int] = set()
    for entry in history:
        if entry.tool_result.action != "search":
            continue
        result = entry.tool_result
        hit_ids = {hit.unit_id for hit in result.hits}
        fully_consumed = (
            not result.group_hints
            and hit_ids <= selected_ids.get(entry.turn, set())
        )
        repeated_later = latest_turn_by_result[result_keys[entry.turn]] > entry.turn
        if fully_consumed or repeated_later:
            compressible.add(entry.turn)
    return compressible
