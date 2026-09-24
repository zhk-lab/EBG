from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentloop.errors import RetryableModelError
from agentloop.provider import ModelCompletion
from scripts.main.judge import (
    BatchJudgeError,
    JudgeBatchConfig,
    _default_client_factory,
    _expected_model_profile,
    _load_judge_module,
    _write_json,
    run_batch_judges,
)
from scripts.main.layout import migrate_experiment, update_summary


class RoutingClient:
    def __init__(self, config: JudgeBatchConfig, calls: list[list[dict[str, str]]]) -> None:
        self.config = config
        self.calls = calls

    @property
    def profile(self) -> dict:
        return _expected_model_profile(self.config)

    def complete(
        self, messages: list[dict[str, str]], *, max_output_tokens: int
    ) -> ModelCompletion:
        self.calls.append(messages)
        if "gold_conditions" in messages[1]["content"]:
            response = _perfect_specgap_response()
        elif "evidence_id_relation" in messages[1]["content"]:
            response = {
                "verification_point_relation": "equivalent",
                "evidence_case_evaluation": {},
            }
        else:
            response = _perfect_silentswap_response()
        content = json.dumps(response)
        return ModelCompletion(
            content=content,
            raw_response={"model": self.config.judge_model, "content": content},
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        )


class ScriptedClient(RoutingClient):
    def __init__(
        self,
        config: JudgeBatchConfig,
        calls: list[list[dict[str, str]]],
        outcomes: list[object],
    ) -> None:
        super().__init__(config, calls)
        self.outcomes = outcomes

    def complete(
        self, messages: list[dict[str, str]], *, max_output_tokens: int
    ) -> ModelCompletion:
        self.calls.append(messages)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        content = outcome if isinstance(outcome, str) else json.dumps(outcome)
        return ModelCompletion(
            content=content,
            raw_response={"model": self.config.judge_model, "content": content},
            usage={"prompt_tokens": 3, "completion_tokens": 1},
        )


class BatchJudgeTests(unittest.TestCase):
    def test_valid_correction_is_not_replaced_by_failed_retry(self):
        from scripts.main.judge import _needs_deferred_retry, _attempt_usage
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory), benchmarks=("silentswap",))
            root = config.output_root / "judges" / config.judge_model / "ss_test"
            _write(root / "retry_1/status.json", {"status": "failed"})
            result = {"status": "complete", "arm": "graph", "input_id": "ss_test"}
            self.assertFalse(_needs_deferred_retry(config, result))
            _write(root / "alignment_correction/attempts/alignment_1_001.json", {
                "provider_called": True, "usage": {"input_tokens": 10, "output_tokens": 2}})
            self.assertEqual(_attempt_usage(root)["input_tokens"], 10)

    def test_partial_silentswap_answers_are_scored_and_resumed(self) -> None:
        for count in (0, 4):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                config = self._fixture(Path(directory), benchmarks=("silentswap",))
                prediction = _silentswap_prediction()
                prediction["swaps"] = prediction["swaps"][:count]
                _write(config.output_root / "runs/ss_test/prediction.json", prediction)
                response = _perfect_silentswap_response()
                for dimension in ("location_checks", "code_change_checks"):
                    for check in response[dimension][count:]:
                        check["matched_candidate_swap_number"] = None
                        for key, value in check.items():
                            if isinstance(value, bool):
                                check[key] = False
                calls = []
                outcomes = [response]
                for _ in range(2):
                    result = run_batch_judges(
                        config,
                        client_factory=lambda: ScriptedClient(config, calls, outcomes),
                        report=lambda _: None,
                    )
                    self.assertEqual(result["totals"]["completed_samples"], 1)
                    scores = result["groups"]["silentswap/graph"]["score_means"]
                    self.assertEqual(scores["code_change_correct"], 0.5 if count == 4 else 0)
                self.assertEqual(len(calls), 1)

    def test_migrated_judge_reuses_results_and_preserves_prediction_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory), benchmarks=("specgap",))
            calls = []
            prediction_summary = {"totals": {"samples": 1}}
            update_summary(config.output_root, prediction=prediction_summary)
            first = run_batch_judges(config, client_factory=lambda: RoutingClient(config, calls), report=lambda _: None)
            old_prediction = config.output_root / "runs/graph/sg_test"
            old_prediction.parent.mkdir(parents=True)
            (config.output_root / "runs/sg_test").rename(old_prediction)
            old_judge = config.judge_root / "specgap/graph/sg_test"
            old_judge.parent.mkdir(parents=True)
            (config.judge_root / "sg_test").rename(old_judge)
            saved_input = _read(old_judge / "judge_input.json")
            saved_input["prediction_path"] = str(old_prediction / "prediction.json")
            _write(old_judge / "judge_input.json", saved_input)
            _write(config.output_root / "summary.json", prediction_summary)
            _write(config.judge_root / "summary.json", first)
            migrate_experiment(config.output_root)

            def forbidden_factory():
                raise AssertionError("migration must not trigger new model calls")

            resumed = run_batch_judges(config, client_factory=forbidden_factory, report=lambda _: None)
            self.assertEqual(resumed["totals"]["completed_samples"], 1)
            self.assertEqual(len(calls), 1)
            saved = _read(config.output_root / "summary.json")
            self.assertEqual(saved["prediction"], prediction_summary)
            self.assertEqual(saved["judges"][config.judge_model], resumed)
            self.assertFalse((config.judge_root / "summary.json").exists())

    def test_full_judging_uses_prediction_manifest_and_freezes_request_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory), benchmarks=("feedbacktrace",))
            config = replace(config, phase="full", request_options={"reasoning_effort": "low", "temperature": 0.0})
            self.assertEqual(
                config.judge_root, config.output_root / "judges" / config.judge_model
            )
            manifest_path = config.output_root / "manifest.json"
            manifest = _read(manifest_path)
            manifest.update(phase="full", split_file=None, split_id="full")
            _write(manifest_path, manifest)
            _write(config.output_root / "runs/ft_test/prediction.json", _feedbacktrace_prediction())
            calls = []
            for _ in range(2):
                result = run_batch_judges(config, client_factory=lambda: RoutingClient(config, calls), report=lambda _: None)
                self.assertEqual(result["totals"]["completed_samples"], 1)
            self.assertEqual(len(calls), 1)
            with self.assertRaises(BatchJudgeError):
                changed = replace(config, request_options={"reasoning_effort": "high"})
                run_batch_judges(changed, client_factory=lambda: RoutingClient(changed, calls), report=lambda _: None)
            self.assertEqual(len(calls), 1)

    def test_feedbacktrace_joint_judge_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root, benchmarks=("feedbacktrace",))
            calls: list[list[dict[str, str]]] = []

            summary = run_batch_judges(
                config,
                client_factory=lambda: RoutingClient(config, calls),
                report=lambda _: None,
            )

            group = summary["groups"]["feedbacktrace/graph"]
            self.assertEqual(group["completed_samples"], 1)
            self.assertEqual(
                group["score_means"],
                {
                    "verification_point_alignment": 1.0,
                    "evidence_location_score": 1.0,
                    "evidence_hit_rate": 1.0,
                },
            )
            self.assertEqual(len(calls), 1)

    def test_two_benchmarks_persist_usage_and_resume_valid_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root, benchmarks=("specgap", "silentswap"))
            calls: list[list[dict[str, str]]] = []

            first = run_batch_judges(
                config,
                client_factory=lambda: RoutingClient(config, calls),
                report=lambda _: None,
            )

            self.assertEqual(first["totals"]["selected_samples"], 2)
            self.assertEqual(first["totals"]["completed_samples"], 2)
            self.assertEqual(first["totals"]["usage"], {
                "calls": 2,
                "input_tokens": 20,
                "output_tokens": 4,
                "total_tokens": 24,
            })
            self.assertEqual(
                first["groups"]["specgap/graph"]["score_means"]["f1"], 1.0
            )
            self.assertEqual(
                first["groups"]["silentswap/graph"]["score_means"][
                    "code_change_correct"
                ],
                1.0,
            )
            self.assertEqual(len(calls), 2)
            manifest = _read(config.judge_root / "manifest.json")
            self.assertEqual(
                manifest["retry_policy"],
                {
                    "network_retries": 2,
                    "max_http_attempts_per_request": 3,
                    "json_format_repairs": 1,
                    "content_validation_retries": 0,
                    "complete_sample_reruns": 1,
                },
            )

            def forbidden_factory():
                raise AssertionError("valid saved results must be reused")

            second = run_batch_judges(
                config,
                client_factory=forbidden_factory,
                report=lambda _: None,
            )

            self.assertEqual(second["totals"]["usage"]["calls"], 2)
            self.assertTrue(all(item["resumed"] for item in second["samples"]))
            attempt = (
                config.judge_root
                / "sg_test"
                / "attempts"
                / "initial_001.json"
            )
            self.assertEqual(_read(attempt)["status"], "valid")

    def test_one_json_format_repair_can_recover_without_new_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root, benchmarks=("specgap",))
            calls: list[list[dict[str, str]]] = []
            outcomes: list[object] = ["[]", _perfect_specgap_response()]

            summary = run_batch_judges(
                config,
                client_factory=lambda: ScriptedClient(config, calls, outcomes),
                report=lambda _: None,
            )

            group = summary["groups"]["specgap/graph"]
            self.assertEqual(group["completed_samples"], 1)
            self.assertEqual(group["failed_samples"], 0)
            self.assertEqual(group["score_means"]["f1"], 1.0)
            self.assertEqual(group["usage"], {
                "calls": 2,
                "input_tokens": 6,
                "output_tokens": 2,
                "total_tokens": 8,
            })
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1][-2], {"role": "assistant", "content": "[]"})
            self.assertIn("JSON FORMAT REPAIR", calls[1][-1]["content"])
            sample_root = config.judge_root / "sg_test/attempts"
            self.assertEqual(_read(sample_root / "initial_001.json")["status"], "json_invalid")
            self.assertEqual(
                _read(sample_root / "format_repair_1_001.json")["status"], "valid"
            )

    def test_second_malformed_json_fails_without_another_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root, benchmarks=("specgap",))
            config = replace(config, deferred_retries=0)
            calls: list[list[dict[str, str]]] = []
            outcomes: list[object] = ["[]", "not-json"]

            summary = run_batch_judges(
                config,
                client_factory=lambda: ScriptedClient(config, calls, outcomes),
                report=lambda _: None,
            )

            sample = summary["samples"][0]
            self.assertEqual(sample["status"], "failed")
            self.assertIn("JudgeResponseFormatError", sample["failure"])
            self.assertEqual(sample["usage"]["calls"], 2)
            self.assertEqual(len(calls), 2)

    def test_content_validation_error_is_not_retried_when_repairs_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root, benchmarks=("specgap",))
            config = replace(config, deferred_retries=0, format_repairs=0)
            calls: list[list[dict[str, str]]] = []
            outcomes: list[object] = [{}]

            summary = run_batch_judges(
                config,
                client_factory=lambda: ScriptedClient(config, calls, outcomes),
                report=lambda _: None,
            )

            sample = summary["samples"][0]
            self.assertEqual(sample["status"], "failed")
            self.assertIn("JudgeContentValidationError", sample["failure"])
            self.assertEqual(sample["usage"]["calls"], 1)
            self.assertEqual(len(calls), 1)
            attempt = config.judge_root / "sg_test/attempts/initial_001.json"
            self.assertEqual(_read(attempt)["status"], "content_invalid")


    def test_saved_content_failure_is_rejected_without_another_judge_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root, benchmarks=("specgap",))
            config = replace(config, deferred_retries=0, format_repairs=0)
            attempt = (
                config.judge_root
                / "sg_test/attempts/initial_001.json"
            )
            response = _perfect_specgap_response()
            response["unmatched_findings"] = [
                {"finding_id": "F001", "reason": "contradicts the positive match"}
            ]
            _write(
                attempt,
                {
                    "schema_version": 2,
                    "stage": "initial",
                    "network_attempt": 1,
                    "provider_called": True,
                    "status": "content_invalid",
                    "content": json.dumps(response),
                    "raw_response": {"content": json.dumps(response)},
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 2,
                        "total_tokens": 9,
                    },
                    "failure_class": "content_validation",
                },
            )

            def forbidden_factory():
                raise AssertionError("saved Judge content must be reused")

            summary = run_batch_judges(
                config,
                client_factory=forbidden_factory,
                report=lambda _: None,
            )

            self.assertEqual(summary["samples"][0]["status"], "failed")
            self.assertIn("invalid unmatched finding", summary["samples"][0]["failure"])
            self.assertEqual(
                summary["groups"]["specgap/graph"]["score_means"]["f1"], 0.0
            )
            self.assertEqual(_read(attempt)["status"], "content_invalid")
            self.assertEqual(json.loads(_read(attempt)["content"]), response)


    def test_network_failure_retries_twice_then_resumes_without_rerunning_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root, benchmarks=("specgap",))
            config = replace(config, deferred_retries=0)
            calls: list[list[dict[str, str]]] = []
            outcomes: list[object] = [
                RetryableModelError("temporary 1", usage={"prompt_tokens": 1}),
                RetryableModelError("temporary 2", usage={"prompt_tokens": 1}),
                RetryableModelError("temporary 3", usage={"prompt_tokens": 1}),
            ]

            first = run_batch_judges(
                config,
                client_factory=lambda: ScriptedClient(config, calls, outcomes),
                report=lambda _: None,
            )

            self.assertEqual(first["samples"][0]["status"], "failed")
            self.assertIn(
                "JudgeNetworkRetriesExhausted", first["samples"][0]["failure"]
            )
            self.assertEqual(first["samples"][0]["usage"]["calls"], 3)
            self.assertEqual(len(calls), 3)

            def forbidden_factory():
                raise AssertionError("an exhausted Judge sample must not call HTTP again")

            second = run_batch_judges(
                replace(config, deferred_retries=1),
                client_factory=forbidden_factory,
                report=lambda _: None,
            )
            self.assertEqual(second["samples"][0]["status"], "failed")
            self.assertEqual(len(calls), 3)

    def test_failures_retry_after_the_first_pass_and_resume_without_calls(self) -> None:
        failures = {
            "content": [{}],
            "format": ["[]", "not-json"],
            "network": [RetryableModelError("temporary") for _ in range(3)],
        }
        for kind, failed_outcomes in failures.items():
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                config = replace(
                    self._fixture(Path(directory), benchmarks=("specgap", "silentswap")),
                    workers=1,
                    format_repairs=1 if kind == "format" else 0,
                )
                outcomes = [*failed_outcomes, _perfect_silentswap_response(), _perfect_specgap_response()]
                calls, events = [], []
                summary = run_batch_judges(
                    config, client_factory=lambda: ScriptedClient(config, calls, outcomes),
                    report=events.append,
                )
                self.assertFalse(outcomes)
                self.assertEqual(summary["totals"]["completed_samples"], 2)
                self.assertEqual(len(calls), len(failed_outcomes) + 2)
                self.assertIn("gold_conditions", calls[-1][1]["content"])
                self.assertNotIn("gold_conditions", calls[-2][1]["content"])
                retry_index = next(i for i, e in enumerate(events) if e["event"] == "judge_retry_batch_started")
                self.assertEqual(sum(e["event"] == "judge_sample_finished" for e in events[:retry_index]), 2)
                self.assertEqual(summary["totals"]["automatic_retries"], {
                    "initial_failures": 1, "retried": 1, "recovered": 1, "failed_after_retry": 0,
                })
                self.assertEqual(summary["totals"]["usage"]["calls"], len(calls))
                def forbidden():
                    raise AssertionError("completed reruns must be reused")
                resumed = run_batch_judges(config, client_factory=forbidden, report=lambda _: None)
                self.assertEqual(resumed["totals"], summary["totals"])
                self.assertTrue(all(r["resumed"] for r in resumed["samples"]))

    def test_deferred_failure_is_only_retried_once_even_after_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(self._fixture(Path(directory), benchmarks=("specgap",)), format_repairs=0)
            calls, outcomes = [], [{}, {}]
            first = run_batch_judges(
                config, client_factory=lambda: ScriptedClient(config, calls, outcomes),
                report=lambda _: None,
            )
            self.assertEqual(len(calls), 2)
            self.assertEqual(first["totals"]["automatic_retries"]["failed_after_retry"], 1)
            def forbidden():
                raise AssertionError("only one deferred rerun is allowed")
            resumed = run_batch_judges(config, client_factory=forbidden, report=lambda _: None)
            self.assertEqual(resumed["totals"], first["totals"])

    def test_interrupted_deferred_scoring_reuses_saved_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(self._fixture(Path(directory), benchmarks=("specgap",)), format_repairs=0)
            calls, outcomes = [], [{}, _perfect_specgap_response()]
            retry_result = config.judge_root / "sg_test/retry_1/result.json"
            def interrupt_after_response(path, value):
                if path == retry_result:
                    raise KeyboardInterrupt()
                _write_json(path, value)
            with patch("scripts.main.judge._write_json", side_effect=interrupt_after_response):
                with self.assertRaises(KeyboardInterrupt):
                    run_batch_judges(
                        config, client_factory=lambda: ScriptedClient(config, calls, outcomes),
                        report=lambda _: None,
                    )
            self.assertEqual(len(calls), 2)
            def forbidden():
                raise AssertionError("saved deferred response must be reused after interruption")
            resumed = run_batch_judges(config, client_factory=forbidden, report=lambda _: None)
            self.assertEqual(resumed["totals"]["completed_samples"], 1)
            self.assertEqual(resumed["totals"]["usage"]["calls"], 2)
            self.assertEqual(resumed["totals"]["automatic_retries"]["recovered"], 1)

    def test_missing_prediction_does_not_trigger_deferred_model_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory), benchmarks=("specgap",))
            path = config.output_root / "runs/sg_test/prediction.json"
            path.unlink()
            def forbidden():
                raise AssertionError("a missing prediction cannot be judged")
            summary = run_batch_judges(config, client_factory=forbidden, report=lambda _: None)
            self.assertEqual(summary["totals"]["failed_samples"], 1)
            self.assertEqual(summary["totals"]["automatic_retries"]["retried"], 0)

    def test_network_failure_succeeds_on_third_http_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root, benchmarks=("specgap",))
            calls: list[list[dict[str, str]]] = []
            outcomes: list[object] = [
                RetryableModelError("temporary 1"),
                RetryableModelError("temporary 2"),
                _perfect_specgap_response(),
            ]

            summary = run_batch_judges(
                config,
                client_factory=lambda: ScriptedClient(config, calls, outcomes),
                report=lambda _: None,
            )

            self.assertEqual(summary["samples"][0]["status"], "complete")
            self.assertEqual(summary["samples"][0]["usage"]["calls"], 3)
            self.assertEqual(len(calls), 3)

    def test_judge_manifest_rejects_configuration_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root, benchmarks=("specgap",))
            calls: list[list[dict[str, str]]] = []
            run_batch_judges(
                config,
                client_factory=lambda: RoutingClient(config, calls),
                report=lambda _: None,
            )

            with self.assertRaisesRegex(BatchJudgeError, "frozen JSON differs"):
                run_batch_judges(
                    replace(config, network_retries=1),
                    client_factory=lambda: RoutingClient(config, calls),
                    report=lambda _: None,
                )

    def test_local_gateway_needs_no_secret_and_freezes_no_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = JudgeBatchConfig(
                experiment_name="x",
                experiment_root=Path(directory),
                phase="development",
                benchmarks=("specgap",),
                arms=("graph",),
            )
            with patch.dict(os.environ, {}, clear=True):
                client = _default_client_factory(config)()

            self.assertEqual(client.profile["model"], "glm-5-2")
            self.assertEqual(client.profile["thinking"], "omitted")
            self.assertEqual(client.profile["reasoning_effort"], "omitted")
            self.assertEqual(client.profile["temperature"], "provider_default")

    def test_explicit_non_default_judge_model_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = JudgeBatchConfig(
                experiment_name="x",
                experiment_root=Path(directory),
                phase="development",
                benchmarks=("specgap",),
                arms=("graph",),
                judge_model="qwen-3.8",
            )

            config.validate()

    def test_qwen_client_uses_qwen_thinking_switch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = JudgeBatchConfig(
                experiment_name="x",
                experiment_root=Path(directory),
                phase="development",
                benchmarks=("specgap",),
                arms=("graph",),
                judge_model="qwen3.7-max-2026-06-08",
                request_options={"enable_thinking": False},
                base_url="https://example.com/v1",
                api_key_env="QWEN_API_KEY",
            )
            with patch.dict(os.environ, {"QWEN_API_KEY": "secret"}, clear=True):
                client = _default_client_factory(config)()

            self.assertEqual(client.profile["request_options"], {"enable_thinking": False})

    def _fixture(
        self, root: Path, *, benchmarks: tuple[str, ...]
    ) -> JudgeBatchConfig:
        split_path = root / "split.json"
        split = {
            "split_id": "test_split",
            "selection": {"development_count": 1, "formal_count": 1},
            "benchmarks": {
                "specgap": {
                    "development": ["sg_test"],
                    "formal": ["sg_formal"],
                },
                "silentswap": {
                    "development": ["ss_test"],
                    "formal": ["ss_formal"],
                },
                "feedbacktrace": {
                    "development": ["ft_test"],
                    "formal": ["ft_formal"],
                },
            },
        }
        _write(split_path, split)
        experiment_root = root / "experiments"
        output_root = experiment_root / "pilot"
        manifest = {
            "schema_version": 1,
            "experiment_name": "pilot",
            "split_file": str(split_path.resolve()),
            "split_id": "test_split",
            "phase": "development",
            "selected_ids": {
                "specgap": ["sg_test"],
                "silentswap": ["ss_test"],
                "feedbacktrace": ["ft_test"],
            },
            "benchmarks": list(benchmarks),
            "arms": ["graph"],
            "model": "gpt-5-6-luna",
            "base_url": "http://127.0.0.1:28080/v1",
            "thinking": "disabled",
            "prepare_only": False,
        }
        _write(output_root / "manifest.json", manifest)
        artifacts = root / "evaluation"

        if "specgap" in benchmarks:
            formal_root = root / "formal"
            _write_specgap_reference(formal_root)
            module = _load_judge_module("specgap")
            module.DEFAULT_FORMAL_DATA_ROOT = formal_root
            repository = artifacts / "specgap/artifacts/visible_bundles/sg_test/repository"
            source = repository / "pkg/api.py"
            source.parent.mkdir(parents=True)
            source.write_text("\n" * 8 + "def public():\n    return 1\n", encoding="utf-8")
            repository_patch = patch.object(
                module, "_default_repository_root", return_value=repository
            )
            repository_patch.start()
            self.addCleanup(repository_patch.stop)
            loader_patch = patch(
                "scripts.main.judge._load_judge_module",
                side_effect=lambda benchmark: (
                    module if benchmark == "specgap" else _load_judge_module(benchmark)
                ),
            )
            loader_patch.start()
            self.addCleanup(loader_patch.stop)
            _write(
                output_root
                / "runs/sg_test/prediction.json",
                _specgap_prediction(),
            )
            _write(
                artifacts / "specgap/artifacts/hidden_gold/sg_test.json",
                _specgap_gold(),
            )
            document = (
                artifacts
                / "specgap/artifacts/visible_bundles/sg_test/documents/3_document_after.md"
            )
            document.parent.mkdir(parents=True, exist_ok=True)
            document.write_text("Visible document.\n", encoding="utf-8")

        if "silentswap" in benchmarks:
            _write(
                output_root
                / "runs/ss_test/prediction.json",
                _silentswap_prediction(),
            )
            _write(
                artifacts / "silentswap/artifacts/hidden_gold/ss_test.json",
                _silentswap_gold(),
            )

        if "feedbacktrace" in benchmarks:
            _write(
                output_root
                / "runs/ft_test/prediction.json",
                _feedbacktrace_prediction(),
            )
            _write(
                artifacts / "feedbacktrace/artifacts/hidden_gold/ft_test.json",
                _feedbacktrace_gold(),
            )
            _write(
                artifacts
                / "feedbacktrace/artifacts/visible_bundles/ft_test/trace/model_input.json",
                _feedbacktrace_model_input(),
            )

        return JudgeBatchConfig(
            experiment_name="pilot",
            experiment_root=experiment_root,
            phase="development",
            benchmarks=benchmarks,
            arms=("graph",),
            workers=1,
            artifact_root=artifacts,
        )


def _feedbacktrace_model_input() -> dict:
    return {
        "input_id": "ft_test",
        "events": [
            {
                "evidence_id": "e_1",
                "event_type": "assistant_response",
                "content": "I selected plan A.",
                "turn_number": 1,
            }
        ],
    }


def _feedbacktrace_gold() -> dict:
    return {
        "input_id": "ft_test",
        "benchmark": "feedbacktrace",
        "verdict": "KEY",
        "verification_point": "Confirm whether plan A should be retained.",
        "evidence_ids": ["e_1"],
    }


def _feedbacktrace_prediction() -> dict:
    return {
        "input_id": "ft_test",
        "benchmark": "feedbacktrace",
        "verdict": "KEY",
        "verification_point": "Confirm whether plan A should be retained.",
        "supporting_evidence_ids": ["e_1"],
        "criticality": "must_disclose",
    }


def _write_specgap_reference(root: Path) -> None:
    """Provide the formal reference files consumed by the real SpecGap scorer."""
    part = {
        **_specgap_gold()["conditions"][0],
        "type": "behavior",
        "importance": "must_ask",
        "expected_verification_question": "Should public calls return one?",
        "why_important": "Callers observe the result.",
        "downstream_impact": "The API value changes.",
    }
    mapping = {
        "mapping_status": "direct",
        "coverage": "full",
        "mapping_explanation": "The return statement fixes the value.",
        "evidence": [{
            "evidence_id": "kc_001_ev001",
            "evidence_type": "implementation",
            "relation": "implements",
            "strength": "direct",
            "locations": [{
                "file_path": "pkg/api.py",
                "symbol": {"kind": "function", "qualified_name": "public"},
                "line_ranges": [{"start": 10, "end": 10}],
            }],
            "explanation": "The public return value.",
        }],
    }
    sample = root / "1_sample"
    _write(sample / "2_deleted_parts.json", {"deleted_parts": [part]})
    _write(sample / "4_code_mapping.json", {"code_mappings": {
        "comparison_items": [{
            "condition_id": "kc_001",
            "condition_by_condition": mapping,
            "holistic_alignment": {"evidence": []},
        }],
        "gold_labels": [{
            "condition_id": "kc_001",
            "gold_source": "condition_by_condition",
            "mapping_explanation": "The public return value.",
        }],
    }})


def _specgap_gold() -> dict:
    return {
        "input_id": "sg_test",
        "source_sample_id": "sample",
        "conditions": [
            {
                "condition_id": "kc_001",
                "normalized_condition": "Public calls return one.",
                "source_text": "Public calls return one.",
                "implementation_locations": [
                    {
                        "file": "pkg/api.py",
                        "symbol": "public",
                        "line_ranges": [{"start": 10, "end": 10}],
                    }
                ],
            }
        ],
    }


def _specgap_prediction() -> dict:
    return {
        "input_id": "sg_test",
        "benchmark": "specgap",
        "findings": [
            {
                "finding_id": "F001",
                "claim": "Public calls return one.",
                "verification_question": "Should public calls return one?",
                "why_important": "Callers observe the result.",
                "downstream_impact": "The API value changes.",
                "code_evidence": [
                    {
                        "path": "pkg/api.py",
                        "symbol": "public",
                        "start_line": 10,
                        "end_line": 10,
                        "explanation": "The return fixes the behavior.",
                    }
                ],
            }
        ],
    }


def _perfect_specgap_response() -> dict:
    return {
        "matches": [
            {
                "finding_id": "F001",
                "condition_id": "kc_001",
                "match_score": 1.0,
                "question_score": 1.0,
                "reason": "same behavior",
            }
        ],
        "unmatched_findings": [],
    }


def _silentswap_gold() -> dict:
    gold = {
        "input_id": "ss_test",
        "swaps": [
            {
                "swap_type": "parsing_matching",
                "original_semantics": f"before {number}",
                "swapped_semantics": f"after {number}",
                "localization": _location(number),
            }
            for number in range(1, 6)
        ],
    }
    gold["source_line_counts"] = {
        f"pkg/file_{number}.py": 100 for number in range(1, 6)
    }
    gold["official_judge_reference"] = {
        "swaps": [
            {
                "swap_number": number,
                "gold": {key: value for key, value in swap.items() if key != "localization"},
                "localization": swap["localization"],
                "document_target": f"document {number}",
                "executed_behavior_difference": {
                    "on_original": {"stdout": f"before {number}"},
                    "after_all_five_swaps": {"stdout": f"after {number}"},
                },
            }
            for number, swap in enumerate(gold["swaps"], start=1)
        ],
        "changed_files": list(gold["source_line_counts"]),
    }
    return gold


def _silentswap_prediction() -> dict:
    return {
        "input_id": "ss_test",
        "benchmark": "silentswap",
        "swaps": [
            {
                "target": _location(number),
                "swap_type": "parsing_matching",
                "code_change": f"operation {number} changes",
                "trigger_condition": f"input {number}",
                "behavioral_effect": {
                    "before": f"before {number}",
                    "after": f"after {number}",
                },
            }
            for number in range(1, 6)
        ],
    }


def _location(number: int) -> dict:
    return {
        "file": f"pkg/file_{number}.py",
        "symbol": {"kind": "function", "qualified_name": [f"function_{number}"]},
        "line_ranges": [{"start": number, "end": number}],
    }


def _perfect_silentswap_response() -> dict:
    return {
        "location_checks": [
            {
                "reference_swap_number": number,
                "matched_candidate_swap_number": number,
                "file_correct": True,
                "symbol_kind_correct": True,
                "qualified_name_correct": True,
                "line_range_acceptable": True,
                "note": "correct",
            }
            for number in range(1, 6)
        ],
        "code_change_checks": [
            {
                "reference_swap_number": number,
                "matched_candidate_swap_number": number,
                "concrete_operation_correct": True,
                "all_changed_operations_covered": True,
                "original_logic_correct": True,
                "swapped_logic_correct": True,
                "direction_correct": True,
                "no_material_contradiction": True,
                "note": "correct",
            }
            for number in range(1, 6)
        ],
        "score_notes": {
            "location_correct": "All five locations match.",
            "code_change_correct": "All five code changes match.",
        },
        "missing_or_incorrect": [],
        "contradicted_claims": [],
        "rationale": "all five match",
    }


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
