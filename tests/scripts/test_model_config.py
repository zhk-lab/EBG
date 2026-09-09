from __future__ import annotations

import argparse
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentloop.provider import OpenAICompatibleJsonClient
from scripts.model_config import resolve_model_settings
from scripts.main.predict import BatchConfig, run_batch, BatchExperimentError
from scripts.main.judge import JudgeBatchConfig, _default_client_factory, _expected_model_profile, _load_prediction_manifest
from tests.support import ProjectTemporaryDirectory


class ModelConfigTests(unittest.TestCase):
    def test_full_batch_freezes_same_options_for_all_six_arms(self) -> None:
        with ProjectTemporaryDirectory() as root:
            for benchmark, sample in (("specgap", "sg_001"), ("silentswap", "ss_001"), ("feedbacktrace", "ft_001")):
                (root / "evaluation" / benchmark / "artifacts/visible_bundles" / sample).mkdir(parents=True)
            options = {"reasoning_effort": "low", "temperature": 0.0}
            config = BatchConfig(experiment_name="full", experiment_root=root / "experiments", split_file=None,
                phase="full", benchmarks=("specgap", "silentswap", "feedbacktrace"), arms=("raw", "graph"),
                model="configured-model", base_url="http://127.0.0.1:28080/v1", request_options=options,
                artifact_root=root / "evaluation")
            captured = []

            def run_one(args):
                captured.append(args)
                return {"status": "complete", "turns": 1}

            run_batch(config, run_one=run_one, report=lambda _: None)
            self.assertEqual(len(captured), 6)
            self.assertTrue(all(args.request_options == options for args in captured))
            manifest = json.loads((config.output_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(all(value == options for value in manifest["prediction_requests"].values()))
            judge = JudgeBatchConfig(experiment_name="full", experiment_root=config.experiment_root, phase="full",
                benchmarks=config.benchmarks, arms=config.arms, artifact_root=config.artifact_root,
                request_options={"thinking": {"type": "disabled"}})
            self.assertEqual(_load_prediction_manifest(judge)["selected_ids"], manifest["selected_ids"])
            with self.assertRaises(BatchExperimentError):
                run_batch(replace(config, request_options={"reasoning_effort": "none"}), run_one=run_one)
            self.assertEqual(len(captured), 6)
            with patch.dict(os.environ, {}, clear=True):
                client = _default_client_factory(judge)()
            self.assertEqual(client.profile, _expected_model_profile(judge))
            self.assertEqual(client.profile["request_options"], judge.request_options)

    def test_profiles_send_exact_configured_fields(self) -> None:
        with ProjectTemporaryDirectory() as root:
            env = root / ".env"
            env.write_text(
                'BEG_MODEL_PROFILE=GPT_LUNA\n'
                'JUDGE_MODEL_PROFILE=KIMI\n'
                'GPT_BASE_URL=https://example.test/v1\n'
                'GPT_API_KEY="test-key"\n'
                'GPT_LUNA_MODEL_NAME=test-luna\n'
                'GPT_REASONING_EFFORT=none\n'
                'KIMI_BASE_URL=https://example.test/v1\n'
                'KIMI_MODEL_NAME=test-kimi\n'
                'KIMI_THINKING_MODE=required\n'
                'KIMI_REASONING_EFFORT=low\n'
                'QWEN_BASE_URL=https://example.test/v1\n'
                'QWEN_MODEL_NAME=test-qwen\n'
                'QWEN_ENABLE_THINKING=false\n'
                'DEEPSEEK_PRO_BASE_URL=https://example.test/v1\n'
                'DEEPSEEK_PRO_MODEL_NAME=test-deepseek\n'
                'DEEPSEEK_PRO_THINKING_TYPE=disabled\n', encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                args = argparse.Namespace(env_file=env)
                settings = resolve_model_settings(args)
                self.assertEqual(settings.api_key_env, "GPT_API_KEY")
                self.assertEqual(settings.model, "test-luna")
                self.assertEqual(settings.request_options, {"reasoning_effort": "none"})
                self.assertEqual(resolve_model_settings(args, judge=True).request_options, {"reasoning_effort": "low"})
                for profile, expected in (
                    ("QWEN", {"enable_thinking": False}),
                    ("DEEPSEEK_PRO", {"thinking": {"type": "disabled"}}),
                ):
                    args.model_profile = profile
                    current = resolve_model_settings(args)
                    self.assertEqual(current.request_options, expected)
                    captured = {}

                    def opener(request, *, timeout):
                        captured.update(json.loads(request.data))
                        return io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}).encode())

                    client = OpenAICompatibleJsonClient(base_url=current.base_url, model=current.model,
                        api_key="test", request_options=current.request_options, opener=opener)
                    client.complete([], max_output_tokens=10)
                    actual = {key: captured[key] for key in ("thinking", "enable_thinking", "reasoning_effort", "temperature", "top_p") if key in captured}
                    self.assertEqual(actual, expected)
                    self.assertEqual(client.profile["request_options"], actual)

    def test_cli_and_environment_overrides_and_omit(self) -> None:
        with patch.dict(os.environ, {
            "BEG_MODEL_PROFILE": "CUSTOM", "CUSTOM_MODEL_NAME": "arbitrary-model",
            "CUSTOM_BASE_URL": "https://example.test", "CUSTOM_REASONING_EFFORT": "high",
            "CUSTOM_TEMPERATURE": "0.7", "BEG_TEMPERATURE": "0.4",
        }, clear=True):
            args = argparse.Namespace(env_file=Path("nonexistent.env"), reasoning_effort="omit", temperature="0", top_p="0.9")
            settings = resolve_model_settings(args)
            self.assertEqual(settings.request_options, {"temperature": 0.0, "top_p": 0.9})
            args.temperature = "omit"
            self.assertNotIn("temperature", resolve_model_settings(args).request_options)

    def test_missing_thinking_setting_is_not_inferred_from_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = resolve_model_settings(argparse.Namespace(env_file=Path("nonexistent.env"), model="deepseek-any", base_url="https://example.test"))
            self.assertEqual(settings.request_options, {})

    def test_required_thinking_does_not_accept_none(self) -> None:
        with patch.dict(os.environ, {"BEG_THINKING_MODE": "required", "BEG_REASONING_EFFORT": "none"}, clear=True):
            with self.assertRaisesRegex(ValueError, "conflicts"):
                resolve_model_settings(argparse.Namespace(env_file=Path("nonexistent.env"), model="test", base_url="https://example.test"))


if __name__ == "__main__":
    unittest.main()
