from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentloop.benchmark_configs import repo_benchmark_config
from agentloop import prepare_initial_request
from agentloop.errors import AgentLoopError
from scripts.main.predict import (
    BatchConfig,
    BatchExperimentError,
    _client,
    _aggregate_results,
    _freeze_manifest,
    _run,
    collect_response_usage,
    load_split,
    run_batch,
)
from tests.support import ProjectTemporaryDirectory, make_repo_bundle, make_trace_bundle


class BatchPredictionTests(unittest.TestCase):
    def test_manual_recovery_preserves_automatic_retry_failure_count(self) -> None:
        totals = _aggregate_results([{
            "benchmark": "silentswap", "status": "complete", "turns": 1,
            "usage": {"calls": 1, "input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            "auto_retried": True, "automatic_retry_status": "failed",
            "initial_failure": "finish_format_retries_exhausted",
        }])
        self.assertEqual(totals["status_counts"], {"complete": 1})
        self.assertEqual(totals["automatic_retries"]["recovered"], 0)
        self.assertEqual(totals["automatic_retries"]["failed_after_retry"], 1)

    def test_legacy_retry_policy_upgrade_preserves_configuration_checks(self) -> None:
        with ProjectTemporaryDirectory() as root:
            path = root / "manifest.json"
            legacy = {"model": "same-model", "silentswap_retry_policy": {
                "format_repairs_per_attempt": 1,
                "deferred_format_failure_reruns": 1,
                "usage_scope": "successful_attempt_only",
            }}
            expected = {**legacy, "silentswap_retry_policy": {
                "format_repairs_per_attempt": 1,
                "deferred_prediction_failure_reruns": 1,
                "usage_scope": "successful_attempt_only",
            }}
            path.write_text(json.dumps(legacy), encoding="utf-8")
            with self.assertRaises(BatchExperimentError):
                _freeze_manifest(path, {**expected, "model": "different-model"})
            self.assertEqual(json.loads(path.read_text()), legacy)
            _freeze_manifest(path, expected)
            self.assertEqual(json.loads(path.read_text()), expected)
            _freeze_manifest(path, expected)

    def test_silentswap_defers_one_rerun_and_counts_only_successful_attempts(self) -> None:
        with ProjectTemporaryDirectory() as root:
            config = self._retry_config(root)
            calls = []
            events = []

            def run_one(args):
                cache = args.output / "outcome.json"
                if cache.is_file():
                    return json.loads(cache.read_text())
                attempt = int(args.output.name[-1])
                calls.append((args.arm, args.input_id, attempt))
                self._write_response(
                    args.output / "responses/turn_001.json",
                    {"input_tokens": 10 * attempt, "output_tokens": attempt},
                )
                failure = None
                if args.input_id == "ss_retry" and attempt == 1:
                    failure = "finish_format_retries_exhausted"
                elif args.input_id == "ss_twice":
                    failure = "action_format_retries_exhausted"
                elif args.input_id == "ss_grounding":
                    failure = "invalid_finish_grounding_or_content"
                result = {"status": "failed" if failure else "complete", "turns": 1, "failure": failure}
                cache.write_text(json.dumps(result), encoding="utf-8")
                if not failure:
                    (args.output / "prediction.json").write_text(json.dumps({
                        "benchmark": "silentswap", "input_id": args.input_id, "attempt": attempt,
                    }), encoding="utf-8")
                return result

            first = run_batch(config, run_one=run_one, report=events.append)
            self.assertTrue(all(attempt == 1 for _, _, attempt in calls[:8]))
            self.assertTrue(all(attempt == 2 for _, _, attempt in calls[8:]))
            self.assertEqual(len(calls), 14)
            self.assertEqual(first["totals"]["automatic_retries"], {
                "initial_failures": 6,
                "initial_format_failures": 4, "retried": 6,
                "recovered": 2, "failed_after_retry": 4,
            })
            self.assertEqual(first["totals"]["status_counts"], {"complete": 4, "failed": 4})
            self.assertEqual(first["totals"]["usage"]["total_tokens"], 66)
            self.assertEqual(events[-1]["event"], "batch_finished")
            self.assertIn("最终仍失败 4 条", events[-1]["message"])
            for arm in config.arms:
                sample = config.output_root / "runs" / arm / "ss_retry"
                self.assertEqual(json.loads((sample / "prediction.json").read_text())["attempt"], 2)
                record = json.loads((sample / "batch_result.json").read_text())
                self.assertEqual(record["usage"]["total_tokens"], 22)
                self.assertEqual(record["actual_usage"]["total_tokens"], 33)
                self.assertTrue((sample / "attempt_1/responses/turn_001.json").is_file())

            second = run_batch(config, run_one=run_one, report=lambda _: None)
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 14)

    def test_silentswap_network_failure_is_retried_at_the_end(self) -> None:
        with ProjectTemporaryDirectory() as root:
            config = self._retry_config(root, ids=("ss_network", "ss_ok"), arms=("graph",))
            calls = []

            def run_one(args):
                attempt = int(args.output.name[-1])
                calls.append((args.input_id, attempt))
                args.output.mkdir(parents=True, exist_ok=True)
                if args.input_id == "ss_network" and attempt == 1:
                    raise TimeoutError("network retries exhausted")
                (args.output / "prediction.json").write_text('{"swaps": []}', encoding="utf-8")
                return {"status": "complete", "turns": 1}

            summary = run_batch(config, run_one=run_one, report=lambda _: None)
            self.assertEqual(calls[-1], ("ss_network", 2))
            self.assertEqual(summary["totals"]["status_counts"], {"complete": 2})
            self.assertEqual(summary["totals"]["automatic_retries"]["retried"], 1)

    def test_specgap_defers_all_prediction_failures_only_once(self) -> None:
        with ProjectTemporaryDirectory() as root:
            config = self._retry_config(
                root, benchmark="specgap",
                ids=("sg_ok", "sg_format", "sg_grounding", "sg_network", "sg_twice"),
            )
            calls = []

            def run_one(args):
                cache = args.output / "outcome.json"
                if cache.is_file():
                    return json.loads(cache.read_text())
                attempt = int(args.output.name[-1])
                calls.append((args.arm, args.input_id, attempt))
                self._write_response(
                    args.output / "responses/turn_001.json",
                    {"input_tokens": 10 * attempt, "output_tokens": attempt},
                )
                failure = None
                if args.input_id == "sg_twice":
                    failure = "invalid_finish_grounding_or_content"
                elif attempt == 1:
                    if args.input_id == "sg_network":
                        raise TimeoutError("provider retries exhausted")
                    failure = {
                        "sg_format": "finish_format_retries_exhausted",
                        "sg_grounding": "invalid_finish_grounding_or_content",
                    }.get(args.input_id)
                result = {"status": "failed" if failure else "complete", "turns": 1, "failure": failure}
                cache.write_text(json.dumps(result), encoding="utf-8")
                if not failure:
                    (args.output / "prediction.json").write_text(
                        json.dumps({"input_id": args.input_id, "attempt": attempt}),
                        encoding="utf-8",
                    )
                return result

            first = run_batch(config, run_one=run_one, report=lambda _: None)
            self.assertEqual(len(calls), 18)
            self.assertTrue(all(attempt == 1 for _, _, attempt in calls[:10]))
            self.assertTrue(all(attempt == 2 for _, _, attempt in calls[10:]))
            self.assertEqual(first["totals"]["automatic_retries"], {
                "initial_failures": 8, "initial_format_failures": 2,
                "retried": 8, "recovered": 6, "failed_after_retry": 2,
            })
            self.assertEqual(first["totals"]["status_counts"], {"complete": 8, "failed": 2})
            self.assertEqual(first["totals"]["usage"]["total_tokens"], 154)
            for arm in config.arms:
                sample = config.output_root / "runs" / arm / "sg_grounding"
                self.assertEqual(json.loads((sample / "prediction.json").read_text())["attempt"], 2)
                record = json.loads((sample / "batch_result.json").read_text())
                self.assertEqual(record["actual_usage"]["total_tokens"], 33)
                self.assertEqual(record["usage"]["total_tokens"], 22)
            second = run_batch(config, run_one=run_one, report=lambda _: None)
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 18)

    def test_interrupted_deferred_retry_resumes_its_second_attempt(self) -> None:
        for benchmark in ("specgap", "silentswap"):
            with self.subTest(benchmark=benchmark):
                self._check_interrupted_retry(benchmark)

    def _check_interrupted_retry(self, benchmark) -> None:
        with ProjectTemporaryDirectory() as root:
            input_id = "sg_retry" if benchmark == "specgap" else "ss_retry"
            config = self._retry_config(root, ids=(input_id,), arms=("graph",), benchmark=benchmark)
            calls = []
            interrupted = False

            def run_one(args):
                nonlocal interrupted
                attempt = int(args.output.name[-1])
                response = args.output / "responses/turn_001.json"
                if not response.exists():
                    calls.append(attempt)
                    self._write_response(response, {"input_tokens": 10, "output_tokens": 1})
                if attempt == 2 and not interrupted:
                    interrupted = True
                    raise KeyboardInterrupt
                if attempt == 1:
                    return {"status": "failed", "turns": 1, "failure": "finish_format_retries_exhausted"}
                (args.output / "prediction.json").write_text(json.dumps({
                    "input_id": args.input_id, "benchmark": args.benchmark,
                }), encoding="utf-8")
                return {"status": "complete", "turns": 1, "failure": None}

            with self.assertRaises(KeyboardInterrupt):
                run_batch(config, run_one=run_one, report=lambda _: None)
            summary = run_batch(config, run_one=run_one, report=lambda _: None)
            self.assertEqual(calls, [1, 2])
            self.assertEqual(summary["totals"]["usage"]["total_tokens"], 11)
            self.assertEqual(summary["totals"]["automatic_retries"]["recovered"], 1)

    @classmethod
    def _retry_config(cls, root, ids=("ss_ok", "ss_retry", "ss_twice", "ss_grounding"), arms=("raw", "graph"), benchmark="silentswap"):
        values = cls._config_values(root, root / "unused.json")
        values.update(phase="full", split_file=None, benchmarks=(benchmark,), arms=arms,
                      artifact_root=root / "evaluation")
        config = BatchConfig(**values)
        for input_id in ids:
            (config.artifact_root / benchmark / "artifacts/visible_bundles" / input_id).mkdir(parents=True)
        return config

    def test_split_validation_rejects_overlap_and_duplicates(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            split_path = temporary / "split.json"
            self._write_split(split_path)
            loaded = load_split(split_path)
            self.assertEqual(loaded["split_id"], "test_split")

            value = json.loads(split_path.read_text(encoding="utf-8"))
            value["benchmarks"]["specgap"]["formal"] = ["sg_dev"]
            split_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(BatchExperimentError):
                load_split(split_path)

            self._write_split(split_path)
            value = json.loads(split_path.read_text(encoding="utf-8"))
            value["benchmarks"]["silentswap"]["development"] = ["ss_dev", "ss_dev"]
            split_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(BatchExperimentError):
                load_split(split_path)

    def test_manifest_prevents_configuration_drift(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            split_path = temporary / "split.json"
            self._write_split(split_path)
            config = self._config(temporary, split_path)
            run_batch(config, run_one=lambda args: self._complete(args), report=lambda _: None)
            run_batch(config, run_one=lambda args: self._complete(args), report=lambda _: None)
            manifest = json.loads(
                (config.output_root / "manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(manifest["schema_version"], 4)
            self.assertEqual(manifest["specgap_retry_policy"], {
                "format_repairs_per_attempt": 2,
                "deferred_prediction_failure_reruns": 1,
                "usage_scope": "successful_attempt_only",
            })
            self.assertEqual(
                manifest["prompt_variants"],
                {"raw": "baseline", "graph": "BEG"},
            )
            self.assertEqual(
                manifest["prediction_requests"]["specgap"],
                {"reasoning_effort": "none"},
            )

            changed = BatchConfig(
                **{
                    **self._config_values(temporary, split_path),
                    "model": "different-model",
                }
            )
            with self.assertRaises(BatchExperimentError):
                run_batch(
                    changed,
                    run_one=lambda args: self._complete(args),
                    report=lambda _: None,
                )

    def test_response_usage_accepts_both_common_field_names(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            response_root = temporary / "run" / "responses"
            failure_root = temporary / "run" / "provider_failures"
            response_root.mkdir(parents=True)
            self._write_response(
                response_root / "turn_001.json",
                {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            )
            self._write_response(
                response_root / "turn_002.json",
                {"input_tokens": 20, "output_tokens": 6},
            )
            self._write_response(
                response_root / "turn_002_format_01.json",
                {"prompt_tokens": 5, "completion_tokens": 2},
            )
            self._write_response(
                failure_root / "turn_002_retry_0.json",
                {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            )

            usage = collect_response_usage(temporary / "run")

        self.assertEqual(
            usage,
            {"calls": 4, "input_tokens": 42, "output_tokens": 15, "total_tokens": 57},
        )

    def test_reentry_uses_the_same_sample_directory_without_double_counting(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            split_path = temporary / "split.json"
            self._write_split(split_path)
            config = self._config(temporary, split_path)
            calls: list[Path] = []

            def resumable(args):
                calls.append(args.output)
                response = args.output / "responses" / "turn_001.json"
                if not response.exists():
                    self._write_response(
                        response,
                        {"prompt_tokens": 7, "completion_tokens": 3},
                    )
                return {"status": "complete", "turns": 1, "failure": None}

            first = run_batch(config, run_one=resumable, report=lambda _: None)
            second = run_batch(config, run_one=resumable, report=lambda _: None)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(first, second)
        self.assertEqual(second["totals"]["usage"]["calls"], 1)
        self.assertEqual(second["totals"]["usage"]["total_tokens"], 10)

    @staticmethod
    def _write_split(path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "split_id": "test_split",
                    "selection": {"development_count": 1, "formal_count": 1},
                    "benchmarks": {
                        "specgap": {
                            "development": ["sg_dev"],
                            "formal": ["sg_formal"],
                        },
                        "silentswap": {
                            "development": ["ss_dev"],
                            "formal": ["ss_formal"],
                        },
                        "feedbacktrace": {
                            "development": ["ft_dev"],
                            "formal": ["ft_formal"],
                        },
                    },
                }
            ),
            encoding="utf-8",
            newline="",
        )

    @staticmethod
    def _write_response(path: Path, usage: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"usage": usage}),
            encoding="utf-8",
            newline="",
        )

    @staticmethod
    def _complete(args) -> dict:
        return {"status": "complete", "turns": 1, "failure": None}

    @staticmethod
    def _config_values(root: Path, split_path: Path) -> dict:
        return {
            "experiment_name": "pilot",
            "experiment_root": root / "experiments",
            "split_file": split_path,
            "phase": "development",
            "benchmarks": ("specgap",),
            "arms": ("graph",),
            "model": "gpt-test",
            "base_url": "https://example.test/v1",
            "workers": 1,
            "schema_root": PROJECT_ROOT / "schemas",
            "request_options": {"reasoning_effort": "none"},
        }

    @classmethod
    def _config(cls, root: Path, split_path: Path) -> BatchConfig:
        return BatchConfig(**cls._config_values(root, split_path))


class PrepareOnlyTests(unittest.TestCase):
    def test_local_gateway_does_not_require_an_api_key(self) -> None:
        args = SimpleNamespace(
            base_url="http://127.0.0.1:28080/v1",
            model="gpt-5-6-luna",
            api_key_env="BEG_TEST_MISSING_KEY",
            timeout=30.0,
            request_options={"reasoning_effort": "none"},
        )

        with patch.dict(os.environ, {}, clear=True):
            repo_client = _client(args)
            trace_client = _client(args)

        self.assertEqual(repo_client.api_key, "unused-placeholder")
        self.assertEqual(repo_client.model, "gpt-5-6-luna")
        self.assertEqual(repo_client.profile["request_options"], {"reasoning_effort": "none"})
        self.assertEqual(
            repo_client.profile["response_protocol"],
            "json_object_v1",
        )
        self.assertEqual(
            trace_client.profile["response_protocol"],
            "json_object_v1",
        )

    def test_remote_gateway_still_requires_an_api_key(self) -> None:
        args = SimpleNamespace(
            base_url="https://example.com/v1",
            model="gpt-5-6-luna",
            api_key_env="BEG_TEST_MISSING_KEY",
            timeout=30.0,
        )

        with patch.dict(os.environ, {}, clear=True), self.assertRaises(
            AgentLoopError
        ):
            _client(args)

    def test_repo_runner_preserves_configured_prediction_request(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            artifact_root = temporary / "artifacts"
            make_repo_bundle(
                artifact_root / "visible_bundles",
                input_id="ss_schema",
                benchmark="silentswap",
                repository_files={
                    "pkg/api.py": "def public(value):\n    return value\n"
                },
                document="The public API returns a value.\n",
            )
            args = self._args(
                benchmark="silentswap",
                input_id="ss_schema",
                arm="raw",
                artifact_root=artifact_root,
                run_root=temporary / "run",
            )
            args.prepare_only = False
            args.request_options = {"reasoning_effort": "low"}
            captured: dict = {}

            def stop_after_client_binding(_args, **kwargs):
                captured.update(_args.request_options)
                raise AgentLoopError("stop after client binding")

            with patch(
                "scripts.main.predict._client",
                side_effect=stop_after_client_binding,
            ), patch(
                "scripts.main.predict.prepare_initial_request", wraps=prepare_initial_request
            ) as prepare, self.assertRaisesRegex(AgentLoopError, "client binding"):
                _run(args)

        self.assertEqual(captured, {"reasoning_effort": "low"})
        self.assertEqual(prepare.call_args.kwargs["config"].format_repair_attempts, 1)

    def test_repo_prepare_only_is_api_free_resumable_and_immutable(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            artifact_root = temporary / "artifacts"
            bundle = make_repo_bundle(
                artifact_root / "visible_bundles",
                input_id="sg_prepare",
                benchmark="specgap",
                repository_files={
                    "pkg/api.py": "def public(value):\n    return value + 1\n"
                },
                document="UNIQUE_PREPARE_DOCUMENT\n",
            )
            run_root = temporary / "run"
            args = self._args(
                benchmark="specgap",
                input_id="sg_prepare",
                arm="raw",
                artifact_root=artifact_root,
                run_root=run_root,
            )

            with self._no_api_client():
                first = _run(args)
                request_path = (
                    run_root / "requests" / "turn_001_retry_0.json"
                )
                original_request = request_path.read_bytes()
                second = _run(args)

            request = json.loads(original_request.decode("utf-8"))
            rendered = "\n".join(
                message["content"] for message in request["messages"]
            )
            self.assertEqual(first, second)
            self.assertEqual(first["status"], "prepared")
            self.assertEqual(first["prompt_variant"], "baseline")
            self.assertGreater(first["estimated_input_tokens"], 0)
            self.assertEqual(request["token_count"], first["estimated_input_tokens"])
            self.assertEqual(
                [message["role"] for message in request["messages"]],
                ["system", "user"],
            )
            self.assertIn("UNIQUE_PREPARE_DOCUMENT", rendered)
            self.assertIn("INPUT AND SOURCE FORMAT", rendered)
            self.assertIn("Each R Read ID", rendered)
            self.assertNotIn("linear Local Graph JSON", rendered)
            self.assertEqual(request_path.read_bytes(), original_request)
            self._assert_no_model_outputs(run_root)

            document_path = bundle / "documents" / "3_document_after.md"
            document_path.write_text(
                "CHANGED_PREPARE_DOCUMENT\n",
                encoding="utf-8",
                newline="",
            )
            with self._no_api_client(), self.assertRaises(AgentLoopError):
                _run(args)

    def test_feedbacktrace_prepare_only_is_api_free(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            artifact_root = temporary / "artifacts"
            graph_root = (
                artifact_root
                / "behavior_graphs"
                / "ft_prepare_long"
            )
            graph_root.mkdir(parents=True)
            (graph_root / "behavior_graph.json").write_text(
                json.dumps(
                    {
                        "input_id": "ft_prepare_long",
                        "benchmark": "feedbacktrace",
                        "evidence": [
                            {
                                "evidence_id": "E000001",
                                "source_type": "trace",
                                "locator": {
                                    "turn": 1,
                                    "event_index": 0,
                                    "event_type": "user_prompt",
                                    "tool_name": None,
                                    "original_evidence_id": None,
                                },
                                "content": "Inspect the fallback.",
                            },
                            {
                                "evidence_id": "E000002",
                                "source_type": "trace",
                                "locator": {
                                    "turn": 2,
                                    "event_index": 1,
                                    "event_type": "assistant_response",
                                    "tool_name": None,
                                    "original_evidence_id": "e_2",
                                },
                                "content": "I disabled it.",
                            },
                        ],
                        "behaviors": [
                            {
                                "behavior_id": "T0001",
                                "task_id": "K0001",
                                "sequence_index": 1,
                                "start_turn": 1,
                                "end_turn": 2,
                                "demand_refs": [{"evidence_id": "E000001"}],
                                "action_evidence_ids": [],
                                "response_refs": [{"evidence_id": "E000002"}],
                            }
                        ],
                        "edges": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
                newline="",
            )
            run_root = temporary / "run"
            args = self._args(
                benchmark="feedbacktrace",
                input_id="ft_prepare_long",
                arm="graph",
                artifact_root=artifact_root,
                run_root=run_root,
            )

            with self._no_api_client():
                result = _run(args)

            request = json.loads(
                (
                    run_root / "requests" / "turn_001_retry_0.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(result["status"], "prepared")
            self.assertEqual(result["benchmark"], "feedbacktrace")
            self.assertEqual(request["compression"]["mode"], "lossless_one_shot")
            trace_path = run_root / "trace_behavior_view.txt"
            self.assertTrue(trace_path.is_file())
            trace_graph = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(trace_graph["current_task_id"], "K0001")
            self.assertEqual(trace_graph["task_scopes"][0]["task_id"], "K0001")
            self.assertEqual(trace_graph["task_scopes"][0]["status"], "current")
            self.assertEqual(
                trace_graph["task_scopes"][0]["behaviors"][0]["response"][0]["evidence_id"],
                "e_2",
            )
            self.assertNotIn("original_evidence_id", trace_path.read_text(encoding="utf-8"))
            self.assertNotIn('"E000', trace_path.read_text(encoding="utf-8"))
            self.assertEqual(trace_graph["task_scopes"][0]["relations"], [])
            self.assertNotIn("interactions", trace_graph)
            self._assert_no_model_outputs(run_root)

    def test_feedbacktrace_raw_prepare_uses_complete_canonical_trace(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            artifact_root = temporary / "artifacts"
            visible_root = artifact_root / "visible_bundles"
            events = [
                {
                    "event_type": "user_prompt",
                    "turn_number": 1,
                    "content": "Inspect it.",
                },
                {
                    "event_type": "assistant_response",
                    "turn_number": 2,
                    "content": "I changed it.",
                    "evidence_id": "e_2",
                },
            ]
            make_trace_bundle(
                visible_root,
                events,
                input_id="ft_raw_long",
                cutoff=3,
            )
            run_root = temporary / "run"
            args = self._args(
                benchmark="feedbacktrace",
                input_id="ft_raw_long",
                arm="raw",
                artifact_root=artifact_root,
                run_root=run_root,
            )

            with self._no_api_client():
                result = _run(args)

            request = json.loads(
                (run_root / "requests" / "turn_001_retry_0.json").read_text(
                    encoding="utf-8"
                )
            )
            rendered = request["messages"][1]["content"]
            self.assertEqual(result["status"], "prepared")
            self.assertEqual(result["arm"], "raw")
            self.assertEqual(result["prompt_variant"], "baseline")
            self.assertIn('"selectable_evidence_ids":["e_2"]', rendered)
            self.assertIn("I changed it.", rendered)
            self.assertNotIn("[[INTERACTION]]", rendered)
            self._assert_no_model_outputs(run_root)

    @staticmethod
    def _args(
        *,
        benchmark: str,
        input_id: str,
        arm: str,
        artifact_root: Path,
        run_root: Path,
    ):
        return SimpleNamespace(
            benchmark=benchmark,
            input_id=input_id,
            arm=arm,
            artifact_root=artifact_root,
            output=run_root,
            schema_root=PROJECT_ROOT / "schemas",
            api_key_env="BEG_TEST_MISSING_KEY",
            prepare_only=True,
            base_url=None,
            model=None,
            timeout=600.0,
        )


    @staticmethod
    def _no_api_client():
        return patch(
            "scripts.main.predict._client",
            side_effect=AssertionError("prepare-only must not construct a client"),
        )

    def _assert_no_model_outputs(self, run_root: Path) -> None:
        forbidden = (
            run_root / "responses",
            run_root / "prediction.json",
            run_root / "prediction_response.json",
            run_root / "state.json",
        )
        for path in forbidden:
            if path.is_dir():
                self.assertFalse(any(path.iterdir()), str(path))
            else:
                self.assertFalse(path.exists(), str(path))


if __name__ == "__main__":
    unittest.main()
