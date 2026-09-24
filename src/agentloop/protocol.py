"""Three-action JSON protocol with strict field validation and prose extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .errors import ActionParameterError, ProtocolError


@dataclass(frozen=True, slots=True)
class SearchAction:
    text: str

    @property
    def action(self) -> str:
        return "search"

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "text": self.text}


@dataclass(frozen=True, slots=True)
class ReadAction:
    ids: tuple[str, ...]

    @property
    def action(self) -> str:
        return "read"

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "ids": list(self.ids)}


@dataclass(frozen=True, slots=True)
class FinishAction:
    prediction: dict[str, Any]

    @property
    def action(self) -> str:
        return "finish"

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "prediction": self.prediction}


AgentAction = SearchAction | ReadAction | FinishAction


def parse_action(
    content: str,
    *,
    max_read_ids: int,
    max_search_characters: int,
) -> AgentAction:
    """Parse an action, accepting prose around a unique embedded JSON action.

    Responses starting with JSON retain the existing first-action policy.
    """

    if not isinstance(content, str) or not content.strip():
        raise ProtocolError("model response is empty")
    stripped = content.strip()
    decoder = json.JSONDecoder(
        object_pairs_hook=_strict_object_with_line_range_repair,
        parse_constant=_reject_constant,
    )
    try:
        if stripped.startswith(("{", "[")):
            value, end = decoder.raw_decode(stripped)
            if stripped[end:].lstrip().startswith(("}", "]", ",")):
                raise ValueError("model response contains content after the action object")
            if isinstance(value, list):
                if stripped[end:].strip() or len(value) < 2 or not all(
                    _looks_like_action(item) for item in value
                ):
                    raise ValueError("model response must be one bare JSON object")
                value = value[0]
        else:
            value = _extract_embedded_action(stripped, decoder)
        value = _repair_duplicate_line_ranges(value)
    except ProtocolError:
        raise
    except (json.JSONDecodeError, ValueError) as error:
        raise ProtocolError(f"model response is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ProtocolError("model response must be a JSON object")
    action = value.get("action")
    if action not in {"search", "read", "finish"}:
        raise ProtocolError(f"unknown action: {action!r}")

    if action == "search":
        if set(value) != {"action", "text"}:
            raise ActionParameterError(
                "search requires only action and text", action="search"
            )
        text = value.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ActionParameterError("search text must be non-empty", action="search")
        normalized = " ".join(text.split())
        if len(normalized) > max_search_characters:
            raise ActionParameterError(
                f"search text exceeds {max_search_characters} characters",
                action="search",
            )
        return SearchAction(normalized)

    if action == "read":
        if set(value) != {"action", "ids"}:
            raise ActionParameterError(
                "read requires only action and ids", action="read"
            )
        ids = value.get("ids")
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= max_read_ids
            or not all(isinstance(item, str) and item for item in ids)
        ):
            raise ActionParameterError(
                f"read ids must contain 1 to {max_read_ids} non-empty strings",
                action="read",
            )
        if len(ids) != len(set(ids)):
            raise ActionParameterError("read ids must be unique", action="read")
        return ReadAction(tuple(ids))

    if set(value) != {"action", "prediction"}:
        raise ActionParameterError(
            "finish requires only action and prediction", action="finish"
        )
    prediction = value.get("prediction")
    if not isinstance(prediction, dict):
        raise ActionParameterError(
            "finish prediction must be an object", action="finish"
        )
    return FinishAction(prediction)


def _looks_like_action(value: Any) -> bool:
    return isinstance(value, dict) and value.get("action") in {
        "search",
        "read",
        "finish",
    }


def _extract_embedded_action(text: str, decoder: json.JSONDecoder) -> dict[str, Any]:
    """Skip surrounding prose without mistaking nested objects for extra actions."""
    candidates = []
    index = 0
    while (index := text.find("{", index)) != -1:
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index += 1
            continue
        if _looks_like_action(value):
            candidates.append(value)
        index = end
    if not candidates:
        raise ProtocolError("model response must be one bare JSON object")
    if len(candidates) != 1:
        raise ProtocolError("analysis contains multiple action objects; expected one")
    return candidates[0]


def canonical_action(action: AgentAction) -> str:
    return json.dumps(
        action.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


class _DuplicateObject:
    def __init__(self, pairs: list[tuple[str, Any]]) -> None:
        self.pairs = pairs


def _strict_object_with_line_range_repair(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any] | _DuplicateObject:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            return _DuplicateObject(pairs)
        result[key] = value
    return result


def _repair_duplicate_line_ranges(value: Any, *, in_line_ranges: bool = False) -> Any:
    """Repair only ``{"start": a, "start": b}`` inside ``line_ranges``."""

    if isinstance(value, _DuplicateObject):
        pairs = value.pairs
        if (
            in_line_ranges
            and len(pairs) == 2
            and pairs[0][0] == pairs[1][0] == "start"
            and all(type(item) is int and item >= 1 for _, item in pairs)
        ):
            return {"start": pairs[0][1], "end": pairs[1][1]}
        duplicate = next(
            key
            for index, (key, _) in enumerate(pairs)
            if key in {previous for previous, _ in pairs[:index]}
        )
        raise ValueError(f"duplicate JSON key: {duplicate}")
    if isinstance(value, list):
        return [
            _repair_duplicate_line_ranges(item, in_line_ranges=in_line_ranges)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _repair_duplicate_line_ranges(
                item,
                in_line_ranges=key == "line_ranges",
            )
            for key, item in value.items()
        }
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")
