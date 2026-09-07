from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentloop.errors import AgentLoopError
from agentloop.benchmark_configs import repo_benchmark_config
from scripts.main.predict import _client, _run
from tests.support import ProjectTemporaryDirectory, make_repo_bundle, make_trace_bundle


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
            ), self.assertRaisesRegex(AgentLoopError, "client binding"):
                _run(args)

        self.assertEqual(captured, {"reasoning_effort": "low"})

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
