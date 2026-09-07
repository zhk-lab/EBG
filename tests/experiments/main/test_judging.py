from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
    _normalize_specgap_response,
    run_batch_judges,
)


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
    def test_full_judging_uses_prediction_manifest_and_freezes_request_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory), benchmarks=("feedbacktrace",))
            config = replace(config, phase="full", request_options={"reasoning_effort": "low", "temperature": 0.0})
            manifest_path = config.output_root / "manifest.json"
            manifest = _read(manifest_path)
            manifest.update(phase="full", split_file=None, split_id="full")
            _write(manifest_path, manifest)
            _write(config.output_root / "runs/full/feedbacktrace/graph/ft_test/prediction.json", _feedbacktrace_prediction())
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
                    "complete_sample_reruns": 0,
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
                / "specgap"
                / "graph"
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
            sample_root = config.judge_root / "specgap/graph/sg_test/attempts"
            self.assertEqual(_read(sample_root / "initial_001.json")["status"], "json_invalid")
            self.assertEqual(
                _read(sample_root / "format_repair_1_001.json")["status"], "valid"
            )

    def test_second_malformed_json_fails_without_another_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root, benchmarks=("specgap",))
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

    def test_parseable_content_validation_error_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root, benchmarks=("specgap",))
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
            attempt = config.judge_root / "specgap/graph/sg_test/attempts/initial_001.json"
            self.assertEqual(_read(attempt)["status"], "content_invalid")

    def test_specgap_normalization_makes_four_observed_shapes_valid(self) -> None:
        judge = _load_judge_module("specgap")
        cases = [
            (
                [
                    _judge_match("F001", "kc_001", 1.0),
                    _judge_match("F002", "kc_003", 1.0),
                    _judge_match("F003", "kc_002", 0.0),
                ],
                [_unmatched("F003")],
                {"F001", "F002", "F003"},
                {"kc_001", "kc_002", "kc_003"},
            ),
            (
                [
                    _judge_match("F001", "kc_003", 1.0),
                    _judge_match("F002", "kc_001", 1.0),
                    _judge_match("F003", "kc_005", 1.0),
                ],
                [_unmatched("F002")],
                {"F001", "F002", "F003"},
                {"kc_001", "kc_003", "kc_005"},
            ),
            (
                [
                    _judge_match("F001", "kc_001", 1.0),
                    _judge_match("F002", "kc_004", 0.5),
                    _judge_match("F003", "kc_006", 1.0),
                ],
                [_unmatched("F002")],
                {"F001", "F002", "F003"},
                {"kc_001", "kc_004", "kc_006"},
            ),
            (
                [
                    _judge_match("F001", "kc_003", 1.0),
                    _judge_match("F002", "kc_001", 1.0),
                    _judge_match("F003", "kc_004", 0.5),
                    _judge_match("F004", "kc_004", 0.0),
                ],
                [_unmatched("F004")],
                {"F001", "F002", "F003", "F004"},
                {"kc_001", "kc_003", "kc_004"},
            ),
        ]

        for matches, unmatched, finding_ids, condition_ids in cases:
            with self.subTest(matches=matches, unmatched=unmatched):
                original_positive = [
                    dict(item) for item in matches if item["match_score"] > 0
                ]
                normalized = _normalize_specgap_response(
                    {"matches": matches, "unmatched_findings": unmatched}
                )
                validated = judge.validate_llm_response(
                    normalized,
                    finding_ids=finding_ids,
                    condition_ids=condition_ids,
                )

                self.assertEqual(validated, normalized)
                for retained in normalized["matches"]:
                    self.assertIn(retained, original_positive)

    def test_saved_structural_failure_is_normalized_without_another_judge_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root, benchmarks=("specgap",))
            attempt = (
                config.judge_root
                / "specgap/graph/sg_test/attempts/initial_001.json"
            )
            response = _perfect_specgap_response()
            response["unmatched_findings"] = [_unmatched("F001")]
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

            self.assertEqual(summary["samples"][0]["status"], "complete")
            self.assertTrue(summary["samples"][0]["resumed"])
            self.assertEqual(
                summary["groups"]["specgap/graph"]["score_means"]["f1"], 1.0
            )
            self.assertEqual(_read(attempt)["status"], "valid")

    def test_specgap_normalization_keeps_highest_match_with_stable_ties(self) -> None:
        response = {
            "matches": [
                _judge_match("F001", "kc_001", 0.5, reason="lower"),
                _judge_match("F001", "kc_002", 1.0, reason="highest"),
                _judge_match("F002", "kc_002", 0.5, reason="gold duplicate"),
                _judge_match("F003", "kc_003", 0.5, reason="first tie"),
                _judge_match("F003", "kc_003", 0.5, reason="second tie"),
            ],
            "unmatched_findings": [
                _unmatched("F001"),
                _unmatched("F002"),
                _unmatched("F003"),
            ],
        }

        normalized = _normalize_specgap_response(response)

        self.assertEqual(
            normalized["matches"],
            [response["matches"][1], response["matches"][3]],
        )
        self.assertEqual(normalized["unmatched_findings"], [_unmatched("F002")])
        judge = _load_judge_module("specgap")
        judge.validate_llm_response(
            normalized,
            finding_ids={"F001", "F002", "F003"},
            condition_ids={"kc_001", "kc_002", "kc_003"},
        )

    def test_specgap_normalization_drops_condition_id_from_unmatched_findings(self) -> None:
        response = {
            "matches": [_judge_match("F001", "kc_002", 1.0)],
            "unmatched_findings": [_unmatched("kc_001")],
        }

        normalized = _normalize_specgap_response(response, finding_ids={"F001"})

        self.assertEqual(normalized["matches"], response["matches"])
        self.assertEqual(normalized["unmatched_findings"], [])

    def test_specgap_normalization_does_not_repair_semantic_schema_errors(self) -> None:
        judge = _load_judge_module("specgap")
        invalid_score = {
            "matches": [_judge_match("F001", "kc_001", 0.25)],
            "unmatched_findings": [],
        }
        normalized_score = _normalize_specgap_response(invalid_score)
        self.assertEqual(normalized_score, invalid_score)
        with self.assertRaisesRegex(ValueError, "invalid match_score"):
            judge.validate_llm_response(
                normalized_score,
                finding_ids={"F001"},
                condition_ids={"kc_001"},
            )

        missing_assignment = {
            "matches": [_judge_match("F001", "kc_001", 0.0)],
            "unmatched_findings": [],
        }
        normalized_missing = _normalize_specgap_response(missing_assignment)
        self.assertEqual(normalized_missing["matches"], [])
        self.assertEqual(normalized_missing["unmatched_findings"], [])
        with self.assertRaisesRegex(ValueError, "account for every finding"):
            judge.validate_llm_response(
                normalized_missing,
                finding_ids={"F001"},
                condition_ids={"kc_001"},
            )

    def test_network_failure_retries_twice_then_resumes_without_rerunning_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root, benchmarks=("specgap",))
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
                config,
                client_factory=forbidden_factory,
                report=lambda _: None,
            )
            self.assertEqual(second["samples"][0]["status"], "failed")
            self.assertEqual(len(calls), 3)

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

    @staticmethod
    def _fixture(
        root: Path, *, benchmarks: tuple[str, ...]
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
            _write(
                output_root
                / "runs/development/specgap/graph/sg_test/prediction.json",
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
                / "runs/development/silentswap/graph/ss_test/prediction.json",
                _silentswap_prediction(),
            )
            _write(
                artifacts / "silentswap/artifacts/hidden_gold/ss_test.json",
                _silentswap_gold(),
            )

        if "feedbacktrace" in benchmarks:
            _write(
                output_root
                / "runs/development/feedbacktrace/graph/ft_test/prediction.json",
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


def _specgap_gold() -> dict:
    return {
        "input_id": "sg_test",
        "conditions": [
            {
                "condition_id": "kc_001",
                "normalized_condition": "Public calls return one.",
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


def _judge_match(
    finding_id: str,
    condition_id: str,
    match_score: float,
    *,
    reason: str = "judged relation",
) -> dict:
    return {
        "finding_id": finding_id,
        "condition_id": condition_id,
        "match_score": match_score,
        "question_score": 1.0,
        "reason": reason,
    }


def _unmatched(finding_id: str) -> dict:
    return {"finding_id": finding_id, "reason": "no positive match"}


def _silentswap_gold() -> dict:
    return {
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
        "score_notes": {},
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
