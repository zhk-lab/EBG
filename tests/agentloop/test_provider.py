from __future__ import annotations

import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentloop.errors import AgentLoopError, ProviderContextError, RetryableModelError
from agentloop.provider import OpenAICompatibleJsonClient


class FakeResponse:
    def __init__(self, value: dict) -> None:
        self.body = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def completion() -> dict:
    return {
        "model": "gpt-5-6-luna",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": '{"action":"finish","prediction":{}}'},
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }


class ProviderTests(unittest.TestCase):
    def test_specgap_request_disables_thinking_and_uses_plain_json(self) -> None:
        captured: dict = {}

        def opener(request, *, timeout):
            captured.update(json.loads(request.data.decode("utf-8")))
            return FakeResponse(completion())

        client = OpenAICompatibleJsonClient(
            base_url="https://example.com/v1",
            api_key="secret",
            model="gpt-5-6-luna",
            disable_thinking=True,
            reasoning_effort=None,
            opener=opener,
        )
        result = client.complete([], max_output_tokens=32_768)

        self.assertEqual(captured["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", captured)
        self.assertEqual(captured["response_format"], {"type": "json_object"})
        self.assertNotIn("tools", captured)
        self.assertNotIn("tool_choice", captured)
        self.assertNotIn("parallel_tool_calls", captured)
        self.assertEqual(captured["max_tokens"], 32_768)
        self.assertEqual(result.usage["prompt_tokens"], 3)
        self.assertEqual(
            client.profile,
            {
                "provider": "openai_compatible_json",
                "base_url": "https://example.com/v1",
                "model": "gpt-5-6-luna",
                "timeout": 300.0,
                "temperature": "provider_default",
                "response_protocol": "json_object_v1",
                "thinking": "disabled",
                "reasoning_effort": "omitted",
            },
        )

    def test_silentswap_request_uses_medium_without_thinking_field(self) -> None:
        captured: dict = {}

        def opener(request, *, timeout):
            captured.update(json.loads(request.data.decode("utf-8")))
            return FakeResponse(completion())

        client = OpenAICompatibleJsonClient(
            base_url="https://example.com/v1",
            api_key="secret",
            model="gpt-5-6-luna",
            disable_thinking=False,
            reasoning_effort="medium",
            opener=opener,
        )
        client.complete([], max_output_tokens=100)

        self.assertNotIn("thinking", captured)
        self.assertEqual(captured["reasoning_effort"], "medium")
        self.assertEqual(client.profile["thinking"], "omitted")
        self.assertEqual(client.profile["reasoning_effort"], "medium")

    def test_light_reasoning_maps_to_provider_low(self) -> None:
        captured: dict = {}

        def opener(request, *, timeout):
            captured.update(json.loads(request.data.decode("utf-8")))
            return FakeResponse(completion())

        client = OpenAICompatibleJsonClient(
            base_url="https://example.com/v1",
            api_key="secret",
            model="gpt-5-6-luna",
            disable_thinking=False,
            reasoning_effort="light",
            opener=opener,
        )
        client.complete([], max_output_tokens=100)

        self.assertEqual(captured["reasoning_effort"], "low")
        self.assertEqual(client.profile["reasoning_effort"], "light")

    def test_qwen_request_disables_thinking_with_enable_thinking(self) -> None:
        captured: dict = {}

        def opener(request, *, timeout):
            captured.update(json.loads(request.data.decode("utf-8")))
            response = completion()
            response["model"] = "qwen3.7-max-2026-06-08"
            return FakeResponse(response)

        client = OpenAICompatibleJsonClient(
            base_url="https://example.com/v1",
            api_key="secret",
            model="qwen3.7-max-2026-06-08",
            disable_thinking=True,
            thinking_parameter="enable_thinking",
            opener=opener,
        )
        client.complete([], max_output_tokens=100)

        self.assertIs(captured["enable_thinking"], False)
        self.assertNotIn("thinking", captured)
        self.assertEqual(client.profile["thinking"], "disabled")

    def test_provider_accepts_punctuation_only_model_alias(self) -> None:
        client = OpenAICompatibleJsonClient(
            base_url="https://example.com/v1",
            api_key="secret",
            model="gpt-5.6-luna",
            opener=lambda request, timeout: FakeResponse(completion()),
        )

        result = client.complete([], max_output_tokens=10)
        self.assertEqual(result.content, '{"action":"finish","prediction":{}}')

    def test_provider_rejects_a_different_returned_model(self) -> None:
        client = OpenAICompatibleJsonClient(
            base_url="https://example.com/v1",
            api_key="secret",
            model="gpt-5.6-luna",
            opener=lambda request, timeout: FakeResponse(
                {**completion(), "model": "gpt-5-6-terra"}
            ),
        )

        with self.assertRaises(AgentLoopError):
            client.complete([], max_output_tokens=10)

    def test_explicit_temperature_is_sent_and_reported(self) -> None:
        captured: dict = {}

        def opener(request, *, timeout):
            captured.update(json.loads(request.data.decode("utf-8")))
            return FakeResponse(completion())

        client = OpenAICompatibleJsonClient(
            base_url="https://example.com/v1",
            api_key="secret",
            model="gpt-5-6-luna",
            temperature=0,
            opener=opener,
        )
        client.complete([], max_output_tokens=10)

        self.assertEqual(captured["temperature"], 0)
        self.assertEqual(client.profile["temperature"], 0)

    def test_unusable_completion_is_not_treated_as_network_error(self) -> None:
        client = OpenAICompatibleJsonClient(
            base_url="https://example.com/v1",
            api_key="secret",
            model="gpt-5-6-luna",
            opener=lambda request, timeout: FakeResponse(
                {
                    "model": "gpt-5-6-luna",
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {"content": None},
                        }
                    ],
                }
            ),
        )

        with self.assertRaises(AgentLoopError) as raised:
            client.complete([], max_output_tokens=10)
        self.assertNotIsInstance(raised.exception, RetryableModelError)

    def test_length_limited_text_is_returned_for_format_correction(self) -> None:
        client = OpenAICompatibleJsonClient(
            base_url="https://example.com/v1",
            api_key="secret",
            model="gpt-5-6-luna",
            opener=lambda request, timeout: FakeResponse(
                {
                    "model": "gpt-5-6-luna",
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": '{"action":"finish"'},
                        }
                    ],
                }
            ),
        )

        result = client.complete([], max_output_tokens=10)
        self.assertEqual(result.content, '{"action":"finish"')

    def test_connection_reset_is_retryable(self) -> None:
        def opener(request, *, timeout):
            raise ConnectionResetError("reset")

        with self.assertRaises(RetryableModelError):
            self._client(opener).complete([], max_output_tokens=10)

    def test_provider_context_error_is_distinct(self) -> None:
        def opener(request, *, timeout):
            raise urllib.error.HTTPError(
                request.full_url,
                400,
                "bad request",
                {},
                io.BytesIO(b"maximum context length exceeded"),
            )

        with self.assertRaises(ProviderContextError):
            self._client(opener).complete([], max_output_tokens=10)

    def test_invalid_reasoning_effort_is_rejected(self) -> None:
        with self.assertRaises(AgentLoopError):
            OpenAICompatibleJsonClient(
                base_url="https://example.com/v1",
                api_key="secret",
                model="gpt-5-6-luna",
                reasoning_effort="extreme",
            )

    @staticmethod
    def _client(opener) -> OpenAICompatibleJsonClient:
        return OpenAICompatibleJsonClient(
            base_url="https://example.com/v1",
            api_key="secret",
            model="gpt-5-6-luna",
            opener=opener,
        )


if __name__ == "__main__":
    unittest.main()
