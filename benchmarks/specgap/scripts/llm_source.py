"""Shared LLM API and remote JSONL source helpers for SpecGAP generators."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import requests
from json_repair import repair_json


DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com/v1",
    "glm": "https://open.bigmodel.cn/api/paas/v4",
    "openai_compatible": "",
}

DEFAULT_MODELS = {
    "deepseek": "deepseek-v4-pro",
    "glm": "glm-5.2",
    "openai_compatible": "",
}


class BatchAbortError(RuntimeError):
    """Stop all work in a batch while keeping unfinished candidates retryable."""


class FatalAPIError(BatchAbortError):
    """An account-level API error that retrying candidate work cannot fix."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {detail}")


class ResponseLengthError(RuntimeError):
    """The provider exhausted the configured output budget."""


class InputTokenLimitError(RuntimeError):
    """The provider reported an input longer than the evaluation limit."""


def load_env_file(path: Path) -> None:
    """Load unset environment variables from a simple dotenv file."""
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def iter_remote_jsonl_rows(
    url: str,
    headers: dict[str, str],
    offset: int,
    limit: int,
    timeout: int,
    retries: int,
    chunk_size: int = 1024 * 1024,
) -> Iterator[dict[str, Any]]:
    """Stream a selected slice of a remote JSONL file using byte ranges."""
    byte_start = 0
    line_index = 0
    selected_count = 0
    pending = b""

    while selected_count < limit:
        request_headers = {
            **headers,
            "Accept-Encoding": "identity",
            "Range": f"bytes={byte_start}-{byte_start + chunk_size - 1}",
            "User-Agent": "AgentVerifyBench/0.1",
        }

        response: requests.Response | None = None
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = requests.get(
                    url,
                    headers=request_headers,
                    timeout=(30, timeout),
                )
                if response.status_code not in {200, 206}:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:1000]}"
                    )
                break
            except Exception as exc:  # noqa: BLE001 - retry transient source errors.
                last_error = exc
                if response is not None:
                    response.close()
                    response = None
                if attempt < retries:
                    time.sleep(min(2**attempt, 20))

        if response is None:
            raise RuntimeError(
                f"Remote range download failed after {retries} retries: {last_error}"
            )

        chunk = response.content
        status_code = response.status_code
        content_range = response.headers.get("Content-Range", "")
        response.close()

        if status_code == 200 and byte_start > 0:
            raise RuntimeError("Remote server stopped honoring byte-range requests.")
        if not chunk:
            break

        data = pending + chunk
        lines = data.split(b"\n")
        pending = lines.pop()

        for raw_line in lines:
            if not raw_line.strip():
                continue
            if line_index >= offset and selected_count < limit:
                yield json.loads(raw_line.decode("utf-8"))
                selected_count += 1
            line_index += 1

        byte_start += len(chunk)
        if status_code == 200 or len(chunk) < chunk_size:
            break
        if content_range:
            match = re.search(r"/(\d+)$", content_range)
            if match and byte_start >= int(match.group(1)):
                break

    if selected_count < limit and pending.strip() and line_index >= offset:
        yield json.loads(pending.decode("utf-8"))


def get_api_key(provider: str, explicit_key: str | None) -> str:
    """Resolve an API key from the explicit option or provider environment."""
    if explicit_key:
        return explicit_key

    env_names = {
        "deepseek": ["DEEPSEEK_API_KEY", "LLM_API_KEY"],
        "glm": ["GLM_API_KEY", "ZHIPUAI_API_KEY", "LLM_API_KEY"],
        "openai_compatible": ["LLM_API_KEY", "OPENAI_API_KEY"],
    }[provider]
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    raise SystemExit(f"Missing API key. Set one of: {', '.join(env_names)}")


def get_base_url(provider: str, explicit_base_url: str | None) -> str:
    """Resolve and normalize the provider's OpenAI-compatible base URL."""
    base_url = explicit_base_url or DEFAULT_BASE_URLS[provider]
    if not base_url:
        raise SystemExit("--base-url is required for --provider openai_compatible")
    return base_url.rstrip("/")


def get_model(provider: str, explicit_model: str | None) -> str:
    """Resolve the configured model name."""
    model = explicit_model or os.environ.get("LLM_MODEL") or DEFAULT_MODELS[provider]
    if not model:
        raise SystemExit("--model is required for --provider openai_compatible")
    return model


def call_chat_completion(
    messages: list[dict[str, str]],
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    args: argparse.Namespace,
    attempts: int | None = None,
    telemetry_path: Path | None = None,
) -> str:
    """Call an OpenAI-compatible endpoint and optionally persist billing telemetry.

    The telemetry file never contains the API key or request headers.  When the
    provider returns an OpenAI-compatible ``usage`` object it is preserved
    verbatim so later cost reports can use official token counts rather than
    estimating them from character length.
    """
    batch_abort_event = getattr(args, "batch_abort_event", None)
    if batch_abort_event is not None and batch_abort_event.is_set():
        raise BatchAbortError(
            "batch is stopping after an API account/authentication error"
        )

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    thinking = getattr(args, "thinking", None)
    reasoning_effort = getattr(args, "reasoning_effort", None)
    enable_thinking = getattr(args, "enable_thinking", None)
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if thinking is not None:
        payload["thinking"] = {"type": thinking}
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking
    max_attempts = attempts if attempts is not None else args.retries
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        started_at = datetime.now(timezone.utc)
        started_monotonic = time.monotonic()
        response: requests.Response | None = None
        telemetry: dict[str, Any] = {
            "schema_version": "specgap-api-telemetry-1.0",
            "provider": provider,
            "model": model,
            "attempt": attempt,
            "started_at": started_at.isoformat(),
            "request": {
                "message_count": len(messages),
                "input_characters": sum(
                    len(str(message.get("content") or ""))
                    for message in messages
                ),
                "max_tokens": args.max_tokens,
                "max_input_tokens": getattr(args, "max_input_tokens", None),
                "temperature": payload.get("temperature"),
                "thinking": thinking,
                "reasoning_effort": reasoning_effort,
                "enable_thinking": enable_thinking,
            },
        }
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=args.timeout,
            )
            if response.status_code in {401, 402, 403}:
                raise FatalAPIError(response.status_code, response.text[:1000])
            if response.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:1000]}"
                )
            data = response.json()
            telemetry["http_status"] = response.status_code
            telemetry["response_id"] = data.get("id")
            telemetry["usage"] = (
                data.get("usage") if isinstance(data.get("usage"), dict) else {}
            )
            choice = data["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            finish_reason = choice.get("finish_reason")
            telemetry["finish_reason"] = finish_reason
            telemetry["response_characters"] = len(content)
            max_input_tokens = getattr(args, "max_input_tokens", None)
            if max_input_tokens is not None:
                prompt_tokens = telemetry["usage"].get("prompt_tokens")
                if not isinstance(prompt_tokens, int):
                    raise InputTokenLimitError(
                        "API response does not include usage.prompt_tokens"
                    )
                if prompt_tokens > max_input_tokens:
                    telemetry["raw_response"] = data
                    raise InputTokenLimitError(
                        f"input token limit exceeded ({prompt_tokens} > "
                        f"{max_input_tokens})"
                    )
            if finish_reason == "length":
                telemetry["raw_response"] = data
                raise ResponseLengthError(
                    f"API response reached max_tokens={args.max_tokens}"
                )
            if not content.strip():
                reasoning_preview = str(message.get("reasoning_content", ""))[:500]
                raise RuntimeError(
                    "API returned empty assistant content; "
                    f"finish_reason={finish_reason}; "
                    f"reasoning_preview={reasoning_preview}"
                )
            telemetry["status"] = "complete"
            return content
        except Exception as exc:  # noqa: BLE001 - keep CLI retry errors simple.
            last_error = exc
            telemetry["status"] = "failed"
            telemetry["error_type"] = type(exc).__name__
            telemetry["error"] = str(exc)[:2000]
            if response is not None:
                telemetry["http_status"] = response.status_code
            if isinstance(
                exc,
                (FatalAPIError, InputTokenLimitError, ResponseLengthError),
            ):
                if batch_abort_event is not None:
                    if isinstance(exc, FatalAPIError):
                        batch_abort_event.set()
                raise
            if attempt < max_attempts:
                time.sleep(min(2**attempt, 20))
        finally:
            finished_at = datetime.now(timezone.utc)
            telemetry["finished_at"] = finished_at.isoformat()
            telemetry["duration_seconds"] = round(
                time.monotonic() - started_monotonic, 6
            )
            if telemetry_path is not None:
                telemetry_path.parent.mkdir(parents=True, exist_ok=True)
                telemetry_path.write_text(
                    json.dumps(telemetry, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

    raise RuntimeError(f"API call failed after {max_attempts} attempts: {last_error}")


def count_chat_input_tokens(
    messages: list[dict[str, str]],
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    args: argparse.Namespace,
    telemetry_path: Path,
) -> int:
    """Count a prompt with the selected provider's tokenizer before evaluation."""
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": 1,
        "response_format": {"type": "json_object"},
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.thinking is not None:
        payload["thinking"] = {"type": args.thinking}
    if args.reasoning_effort is not None:
        payload["reasoning_effort"] = args.reasoning_effort
    if getattr(args, "enable_thinking", None) is not None:
        payload["enable_thinking"] = args.enable_thinking

    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        response: requests.Response | None = None
        telemetry: dict[str, Any] = {
            "schema_version": "specgap-token-count-1.0",
            "provider": provider,
            "model": model,
            "attempt": attempt,
            "input_characters": sum(len(message["content"]) for message in messages),
        }
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=args.timeout,
            )
            if response.status_code in {401, 402, 403}:
                raise FatalAPIError(response.status_code, response.text[:1000])
            if response.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:1000]}"
                )
            data = response.json()
            usage = data.get("usage")
            prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            if not isinstance(prompt_tokens, int):
                raise RuntimeError("token-count response lacks usage.prompt_tokens")
            telemetry.update(
                {
                    "status": "complete",
                    "http_status": response.status_code,
                    "usage": usage,
                    "finish_reason": data["choices"][0].get("finish_reason"),
                }
            )
            return prompt_tokens
        except Exception as exc:  # noqa: BLE001 - mirror completion retry policy.
            last_error = exc
            telemetry.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                }
            )
            if response is not None:
                telemetry["http_status"] = response.status_code
            if isinstance(exc, FatalAPIError):
                raise
            if attempt < args.retries:
                time.sleep(min(2**attempt, 20))
        finally:
            telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            telemetry_path.write_text(
                json.dumps(telemetry, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    raise RuntimeError(f"token count failed after {args.retries} attempts: {last_error}")


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse an LLM response as a JSON object, repairing common syntax damage."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        candidate = match.group(0) if match else text
        repaired = repair_json(candidate)
        data = json.loads(repaired)
        print("[repair] repaired malformed model JSON", file=sys.stderr)

    if isinstance(data, list):
        objects = [item for item in data if isinstance(item, dict)]
        if len(objects) == 1:
            data = objects[0]
            print("[repair] unwrapped model JSON object", file=sys.stderr)

    if not isinstance(data, dict):
        raise ValueError("Model output is not a JSON object.")
    return data
