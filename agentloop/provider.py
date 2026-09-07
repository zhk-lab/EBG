"""Minimal OpenAI-compatible client used by evaluation runners."""

from __future__ import annotations

import http.client
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .errors import AgentLoopError, ProviderContextError, RetryableModelError


@dataclass(frozen=True, slots=True)
class ModelCompletion:
    content: str
    raw_response: dict[str, Any]
    usage: dict[str, Any]


class ModelClient(Protocol):
    @property
    def profile(self) -> dict[str, Any]: ...

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int,
    ) -> ModelCompletion: ...


class OpenAICompatibleJsonClient:
    """Send one non-streaming completion without hidden retries."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 300.0,
        disable_thinking: bool = True,
        thinking_parameter: str = "thinking",
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        request_options: dict[str, Any] | None = None,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise AgentLoopError("base_url must be HTTP(S)")
        if not api_key or not model or timeout <= 0:
            raise AgentLoopError("api_key, model, and positive timeout are required")
        if reasoning_effort not in {None, "none", "light", "medium", "high"}:
            raise AgentLoopError("unsupported reasoning_effort")
        if thinking_parameter not in {"thinking", "enable_thinking"}:
            raise AgentLoopError("unsupported thinking parameter")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.disable_thinking = disable_thinking
        self.thinking_parameter = thinking_parameter
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.request_options = None if request_options is None else dict(request_options)
        self.opener = opener

    @property
    def profile(self) -> dict[str, Any]:
        """Stable, non-secret identity used to guard resumable runs."""

        profile = {
            "provider": "openai_compatible_json",
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "temperature": (
                self.temperature
                if self.temperature is not None
                else "provider_default"
            ),
            "response_protocol": "json_object_v1",
            "thinking": "disabled" if self.disable_thinking else "omitted",
            "reasoning_effort": self.reasoning_effort or "omitted",
        }
        if self.request_options is not None:
            profile.update(
                temperature=self.request_options.get("temperature", "provider_default"),
                thinking=self.request_options.get("thinking", "omitted"),
                reasoning_effort=self.request_options.get("reasoning_effort", "omitted"),
                request_options=self.request_options,
            )
        return profile

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int,
    ) -> ModelCompletion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.disable_thinking:
            if self.thinking_parameter == "enable_thinking":
                payload["enable_thinking"] = False
            else:
                payload["thinking"] = {"type": "disabled"}
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = (
                "low" if self.reasoning_effort == "light" else self.reasoning_effort
            )
        if self.request_options is not None:
            for key in ("temperature", "thinking", "enable_thinking", "reasoning_effort"):
                payload.pop(key, None)
            payload.update(self.request_options)
        request = urllib.request.Request(
            _chat_url(self.base_url),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self.opener(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            if error.code == 413 or "context" in detail.casefold():
                raise ProviderContextError("provider rejected request length") from error
            if error.code in {408, 409, 425, 429} or error.code >= 500:
                raise RetryableModelError(f"provider HTTP {error.code}") from error
            raise AgentLoopError(f"provider HTTP {error.code}") from error
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
        ) as error:
            raise RetryableModelError(type(error).__name__) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RetryableModelError("provider returned invalid JSON") from error

        if not isinstance(raw, dict):
            raise RetryableModelError("provider response is not an object")
        usage_value = raw.get("usage")
        usage = usage_value if isinstance(usage_value, dict) else {}
        returned_model = str(raw.get("model") or "")
        if returned_model and _canonical_model_id(returned_model) != _canonical_model_id(
            self.model
        ):
            raise AgentLoopError(
                f"provider returned model {returned_model!r}, expected {self.model!r}"
            )
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RetryableModelError(
                "provider response contains no choice",
                raw_response=raw,
                usage=usage,
            )
        choice = choices[0]
        if not isinstance(choice, dict):
            raise RetryableModelError(
                "provider choice is invalid",
                raw_response=raw,
                usage=usage,
            )
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        finish_reason = choice.get("finish_reason")
        if finish_reason not in {"stop", "length"} or not isinstance(content, str):
            raise AgentLoopError(
                "provider completion is unusable: "
                f"{finish_reason!r}"
            )
        return ModelCompletion(
            content=content,
            raw_response=raw,
            usage=usage,
        )


def _chat_url(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return base_url + "/chat/completions"
    return base_url + "/v1/chat/completions"


def _canonical_model_id(model: str) -> str:
    """Compare provider aliases while ignoring punctuation-only rewrites."""

    return re.sub(r"[^a-z0-9]+", "", model.casefold())


def validated_model_profile(client: ModelClient) -> dict[str, Any]:
    profile = client.profile
    if not isinstance(profile, dict) or not profile:
        raise AgentLoopError("model client must expose a non-empty public profile")
    try:
        serialized = json.dumps(profile, ensure_ascii=False, sort_keys=True)
        result = json.loads(serialized)
    except (TypeError, ValueError) as error:
        raise AgentLoopError("model client profile must be JSON-serializable") from error
    if not isinstance(result, dict):
        raise AgentLoopError("model client profile must be an object")
    return result
