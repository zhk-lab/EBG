"""Atomic, resumable persistence for AgentLoop runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import AgentLoopError


class RunStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def load_state(self) -> dict[str, Any] | None:
        path = self.root / "state.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AgentLoopError(f"cannot load AgentLoop state: {path}") from error
        if not isinstance(value, dict):
            raise AgentLoopError("AgentLoop state must contain an object")
        return value

    def save_state(self, value: dict[str, Any]) -> None:
        self._write_json(self.root / "state.json", value)

    def load_response(
        self, turn: int, *, format_retry: int = 0
    ) -> dict[str, Any] | None:
        stem = self._turn_stem(turn, format_retry)
        text_path = self.root / "responses" / f"{stem}.txt"
        json_path = self.root / "responses" / f"{stem}.json"
        if not json_path.is_file():
            if text_path.is_file():
                raise AgentLoopError(
                    f"incomplete persisted response for turn {turn}"
                )
            return None
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AgentLoopError(f"cannot load persisted response for turn {turn}") from error
        if not isinstance(metadata, dict):
            raise AgentLoopError(f"persisted response metadata is invalid for turn {turn}")
        content = metadata.get("content")
        if not isinstance(content, str):
            raise AgentLoopError(f"persisted response content is invalid for turn {turn}")
        expected_mirror = content.encode("utf-8")
        try:
            mirror_matches = (
                text_path.is_file()
                and text_path.read_bytes() == expected_mirror
            )
        except OSError as error:
            raise AgentLoopError(
                f"cannot inspect persisted response mirror for turn {turn}"
            ) from error
        if not mirror_matches:
            self._write_text(text_path, content)
        return metadata

    def load_request(
        self,
        turn: int,
        provider_retry: int,
        *,
        format_retry: int = 0,
    ) -> dict[str, Any] | None:
        stem = self._turn_stem(turn, format_retry)
        path = (
            self.root
            / "requests"
            / f"{stem}_retry_{provider_retry}.json"
        )
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AgentLoopError(f"cannot load persisted request: {path}") from error
        if not isinstance(value, dict):
            raise AgentLoopError(f"persisted request is invalid: {path}")
        return value

    def save_request(
        self,
        turn: int,
        *,
        messages: list[dict[str, str]],
        token_count: int,
        compression: dict[str, Any],
        provider_retry: int,
        format_retry: int = 0,
    ) -> None:
        stem = self._turn_stem(turn, format_retry)
        self._write_json(
            self.root / "requests" / f"{stem}_retry_{provider_retry}.json",
            {
                "turn": turn,
                **self._format_metadata(format_retry),
                "provider_retry": provider_retry,
                "token_count": token_count,
                "compression": compression,
                "messages": messages,
            },
        )

    def save_request_if_unchanged(
        self,
        turn: int,
        *,
        messages: list[dict[str, str]],
        token_count: int,
        compression: dict[str, Any],
        provider_retry: int,
        format_retry: int = 0,
    ) -> None:
        """Create a resumable request or verify the existing request is identical."""

        expected = {
            "turn": turn,
            **self._format_metadata(format_retry),
            "provider_retry": provider_retry,
            "token_count": token_count,
            "compression": compression,
            "messages": messages,
        }
        existing = self.load_request(
            turn,
            provider_retry,
            format_retry=format_retry,
        )
        if existing is not None:
            if existing != expected:
                raise AgentLoopError(
                    f"persisted request differs while resuming turn {turn}"
                )
            return
        self.save_request(
            turn,
            messages=messages,
            token_count=token_count,
            compression=compression,
            provider_retry=provider_retry,
            format_retry=format_retry,
        )

    def save_response(
        self,
        turn: int,
        *,
        content: str,
        raw_response: dict[str, Any],
        usage: dict[str, Any],
        provider_retry: int,
        format_retry: int = 0,
    ) -> None:
        stem = self._turn_stem(turn, format_retry)
        self._write_json(
            self.root / "responses" / f"{stem}.json",
            {
                "turn": turn,
                **self._format_metadata(format_retry),
                "content": content,
                "provider_retry": provider_retry,
                "usage": usage,
                "raw_response": raw_response,
            },
        )
        self._write_text(
            self.root / "responses" / f"{stem}.txt",
            content,
        )

    def save_provider_failure(
        self,
        turn: int,
        *,
        provider_retry: int,
        error: str,
        raw_response: dict[str, Any] | None,
        usage: dict[str, Any],
        format_retry: int = 0,
    ) -> None:
        """Persist one failed provider attempt without treating it as an action."""

        stem = self._turn_stem(turn, format_retry)
        self._write_json(
            self.root
            / "provider_failures"
            / f"{stem}_retry_{provider_retry}.json",
            {
                "turn": turn,
                **self._format_metadata(format_retry),
                "provider_retry": provider_retry,
                "error": error,
                "usage": usage,
                "raw_response": raw_response,
            },
        )

    def save_prediction_response(self, value: dict[str, Any]) -> None:
        self._write_json(self.root / "prediction_response.json", value)

    def save_prediction(self, value: dict[str, Any]) -> None:
        self._write_json(self.root / "prediction.json", value)

    def save_trace_view(self, content: str) -> None:
        self._write_text(self.root / "trace_behavior_view.txt", content)

    @staticmethod
    def _turn_stem(turn: int, format_retry: int) -> str:
        if turn <= 0 or format_retry < 0:
            raise AgentLoopError("turn must be positive and format_retry non-negative")
        if format_retry == 0:
            return f"turn_{turn:03d}"
        return f"turn_{turn:03d}_format_{format_retry:02d}"

    @staticmethod
    def _format_metadata(format_retry: int) -> dict[str, int]:
        return {"format_retry": format_retry} if format_retry else {}

    def _write_json(self, path: Path, value: Any) -> None:
        content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        self._write_text(path, content)

    def _write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8", newline="")
        temporary.replace(path)
