from __future__ import annotations

import json
import sys
import unittest
from collections.abc import Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentloop import AgentLoop, RawBackend, RunStore
from agentloop.config import DEFAULT_CONFIG
from agentloop.context import FastTokenCounter, prepare_messages
from agentloop.evidence import EvidenceSpan
from agentloop.provider import ModelCompletion
from agentloop.runner import _finish_format_example
from agentloop.errors import (
    AgentLoopError,
    PredictionError,
    PredictionFormatError,
    ProviderContextError,
    RetryableModelError,
)
from beg.core.model import RepoArtifact, TaskDocument, VisibleBundle
from tests.support import ProjectTemporaryDirectory


class ScriptedClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    @property
    def profile(self) -> dict:
        return {"provider": "scripted", "model": "fake"}

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int,
    ) -> ModelCompletion:
        self.calls.append(messages)
        content = self.responses.pop(0)
        return ModelCompletion(
            content=content,
            raw_response={
                "model": "fake",
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            },
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )


class RetryOnceClient(ScriptedClient):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.failed = False

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int,
    ) -> ModelCompletion:
        if not self.failed:
            self.failed = True
            self.calls.append(messages)
            raise RetryableModelError("temporary")
        return super().complete(messages, max_output_tokens=max_output_tokens)


class AlwaysRetryClient(ScriptedClient):
    def __init__(self) -> None:
        super().__init__([])

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int,
    ) -> ModelCompletion:
        self.calls.append(messages)
        raise RetryableModelError("temporary")


class OtherProfileClient(ScriptedClient):
    @property
    def profile(self) -> dict:
        return {"provider": "scripted", "model": "different"}


class ContextOnceClient(ScriptedClient):
    def __init__(self, responses: list[str], reject_call: int) -> None:
        super().__init__(responses)
        self.reject_call = reject_call

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_output_tokens: int,
    ) -> ModelCompletion:
        if len(self.calls) + 1 == self.reject_call:
            self.calls.append(messages)
            raise ProviderContextError("provider limit mismatch")
        return super().complete(messages, max_output_tokens=max_output_tokens)


class GroundedFinishContract:
    def __init__(self, version: str = "v1") -> None:
        self.version = version

    @property
    def resume_identity(self) -> dict:
        return {"kind": "test-grounded", "version": self.version}

    def validate(
        self,
        prediction: dict,
        *,
        input_id: str,
        observed_spans: Sequence[EvidenceSpan],
    ) -> dict:
        for finding in prediction.get("findings", []):
            for evidence in finding.get("code_evidence", []):
                if not any(
                    span.contains(
                        evidence["path"],
                        evidence["start_line"],
                        evidence["end_line"],
                    )
                    for span in observed_spans
                ):
                    raise PredictionError("prediction location was not read")
        return {
            **prediction,
            "input_id": input_id,
            "benchmark": "specgap",
        }


class FormatAwareFinishContract(GroundedFinishContract):
    def validate(
        self,
        prediction: dict,
        *,
        input_id: str,
        observed_spans: Sequence[EvidenceSpan],
    ) -> dict:
        if not isinstance(prediction.get("findings"), list):
            raise PredictionFormatError("findings must be an array")
        return super().validate(
            prediction,
            input_id=input_id,
            observed_spans=observed_spans,
        )


def visible_bundle() -> VisibleBundle:
    code = "def public(value):\n    return value + 1  # " + ("context " * 300) + "\n"
    return VisibleBundle(
        root=Path("."),
        input_id="sg_runner",
        benchmark="specgap",
        task_document=TaskDocument("document.md", "The public API returns a value."),
        repo_artifacts=(
            RepoArtifact("pkg/api.py", Path("pkg/api.py"), "source", code),
        ),
        trace_events=(),
        cutoff_turn=None,
        visible_repository_files=1,
        excluded_repository_files=0,
    )


def finish(path: str = "pkg/api.py", start: int = 1, end: int = 2) -> str:
    prediction = {
        "findings": [
            {
                "finding_id": "F001",
                "claim": "The implementation adds one to the input.",
                "verification_question": "Should the result be incremented?",
                "why_important": "It changes the public result.",
                "downstream_impact": "Callers observe a different number.",
                "code_evidence": [
                    {
                        "path": path,
                        "start_line": start,
                        "end_line": end,
                        "symbol": "public",
                        "explanation": "The return expression performs the increment.",
                    }
                ],
            }
        ]
    }
    return json.dumps({"action": "finish", "prediction": prediction})


class RunnerTests(unittest.TestCase):
    def _runner(
        self,
        root: Path,
        client: ScriptedClient,
        *,
        config=DEFAULT_CONFIG,
        max_rounds: int = 12,
        finish_contract: GroundedFinishContract | None = None,
    ) -> AgentLoop:
        counter = FastTokenCounter(config.token_estimator)
        backend = RawBackend(
            visible_bundle(),
            index_budget=config.index_budget,
            count_tokens=counter.count_text,
        )
        return AgentLoop(
            backend=backend,
            initial_user_prompt="INITIAL USER PROMPT",
            max_rounds=max_rounds,
            finish_contract=finish_contract or GroundedFinishContract(),
            client=client,
            store=RunStore(root),
            config=config,
        )

    def test_read_then_grounded_finish_writes_prediction(self) -> None:
        client = ScriptedClient(
            ['{"action":"read","ids":["R0001"]}', finish()]
        )
        with ProjectTemporaryDirectory() as temporary:
            run_root = temporary / "run"
            outcome = self._runner(run_root, client).run()
            saved = json.loads(
                (run_root / "prediction.json").read_text(encoding="utf-8")
            )
            state = json.loads(
                (run_root / "state.json").read_text(encoding="utf-8")
            )

        self.assertEqual(outcome.status, "complete")
        self.assertEqual(outcome.turns, 2)
        self.assertEqual(saved["input_id"], "sg_runner")
        self.assertEqual(saved["findings"][0]["finding_id"], "F001")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(state["terminal_record"]["turn"], 2)
        self.assertEqual(state["terminal_record"]["action"]["action"], "finish")
        self.assertEqual(state["schema_version"], 6)
        self.assertEqual(state["max_rounds"], 12)
        self.assertEqual(
            state["finish_contract"],
            {"kind": "test-grounded", "version": "v1"},
        )
        self.assertTrue(
            client.calls[0][1]["content"].startswith(
                "INITIAL USER PROMPT\n\n[[RUN STATE]]"
            )
        )
        self.assertEqual(
            client.calls[0][1]["content"].count("INITIAL USER PROMPT"), 1
        )
        self.assertIn("usage", state["terminal_record"])

    def test_non_json_is_repaired_without_consuming_an_action_round(self) -> None:
        client = ScriptedClient(
            ["not json", '{"action":"read","ids":["R0001"]}', finish()]
        )
        with ProjectTemporaryDirectory() as temporary:
            run_root = temporary / "run"
            outcome = self._runner(run_root, client).run()
            state = json.loads(
                (run_root / "state.json").read_text(encoding="utf-8")
            )

        self.assertEqual(outcome.status, "complete")
        self.assertEqual(outcome.turns, 2)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(state["records"][0]["format_retry"], 1)
        self.assertEqual(len(state["records"][0]["format_errors"]), 1)
        self.assertEqual(
            state["records"][0]["format_errors"][0]["category"],
            "action_protocol",
        )
        correction = client.calls[1][-1]["content"]
        self.assertIn("[[ACTION FORMAT CORRECTION]]", correction)
        self.assertIn("Do not request, read, or assume any new evidence", correction)

    def test_action_array_executes_only_the_first_without_correction(self) -> None:
        multiple = (
            '[{"action":"read","ids":["R0001"]},'
            '{"action":"search","text":"unused"}]'
        )
        client = ScriptedClient([multiple, finish()])
        with ProjectTemporaryDirectory() as temporary:
            run_root = temporary / "run"
            outcome = self._runner(run_root, client).run()
            state = json.loads(
                (run_root / "state.json").read_text(encoding="utf-8")
            )

        self.assertEqual(outcome.status, "complete")
        self.assertEqual(outcome.turns, 2)
        self.assertEqual(len(state["records"]), 1)
        self.assertEqual(state["records"][0]["action"]["action"], "read")
        self.assertEqual(state["records"][0]["format_errors"], [])

    def test_concatenated_actions_execute_only_the_first_without_correction(self) -> None:
        multiple = (
            '{"action":"read","ids":["R0001"]}'
            '{"action":"search","text":"unused"}'
        )
        client = ScriptedClient([multiple, finish()])
        with ProjectTemporaryDirectory() as temporary:
            run_root = temporary / "run"
            outcome = self._runner(run_root, client).run()
            state = json.loads(
                (run_root / "state.json").read_text(encoding="utf-8")
            )

        self.assertEqual(outcome.status, "complete")
        self.assertEqual(outcome.turns, 2)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(state["records"][0]["action"]["action"], "read")
        self.assertEqual(state["records"][0]["format_retry"], 0)
        self.assertEqual(state["records"][0]["format_errors"], [])

    def test_finish_schema_error_is_repaired_but_cannot_become_read(self) -> None:
        invalid_finish = '{"action":"finish","prediction":{}}'
        client = ScriptedClient(
            [
                '{"action":"read","ids":["R0001"]}',
                invalid_finish,
                '{"action":"read","ids":["R0001"]}',
                finish(),
            ]
        )
        with ProjectTemporaryDirectory() as temporary:
            run_root = temporary / "run"
            outcome = self._runner(
                run_root,
                client,
                finish_contract=FormatAwareFinishContract(),
            ).run()
            state = json.loads(
                (run_root / "state.json").read_text(encoding="utf-8")
            )

        self.assertEqual(outcome.status, "complete")
        self.assertEqual(outcome.turns, 2)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(state["terminal_record"]["format_retry"], 2)
        self.assertEqual(len(state["terminal_record"]["format_errors"]), 2)
        self.assertTrue(
            all(
                item["category"] == "finish_schema"
                for item in state["terminal_record"]["format_errors"]
            )
        )
        correction = client.calls[2][-1]["content"]
        self.assertTrue(
            correction.startswith(
                "Your finish JSON format is invalid. Please strictly correct it "
                "to the format below:\n"
            )
        )
        self.assertIn('"action": "finish"', correction)
        self.assertIn('"findings": [', correction)
        self.assertIn('"start_line": 1', correction)
        self.assertIn('"end_line": 1', correction)
        self.assertIn("Preserve your original judgment and evidence.", correction)
        self.assertIn(
            "Correct only the format; do not add or change substantive content.",
            correction,
        )
        self.assertIn(
            "Output only the complete JSON object, without explanation.", correction
        )
        self.assertNotIn("sample-wide format-repair budget", correction)
        self.assertNotIn("Pick exactly one corrected template", correction)
        self.assertNotIn("Validation error", correction)

    def test_malformed_finish_json_receives_finish_format_example(self) -> None:
        malformed = '{"action":"finish","prediction":{"findings":[]}}}'
        client = ScriptedClient(
            ['{"action":"read","ids":["R0001"]}', malformed, finish()]
        )
        with ProjectTemporaryDirectory() as temporary:
            run_root = temporary / "run"
            outcome = self._runner(run_root, client).run()
            state = json.loads(
                (run_root / "state.json").read_text(encoding="utf-8")
            )

        self.assertEqual(outcome.status, "complete")
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(
            state["terminal_record"]["format_errors"][0]["category"],
            "finish_schema",
        )
        correction = client.calls[2][-1]["content"]
        self.assertTrue(
            correction.startswith(
                "Your finish JSON format is invalid. Please strictly correct it "
                "to the format below:\n"
            )
        )
        self.assertIn('"action": "finish"', correction)

    def test_analysis_wrapped_actions_execute_without_format_retry(self) -> None:
        client = ScriptedClient([
            'I will search first.\n{"action":"search","text":"main"}',
            'Now read the source.\n```json\n{"action":"read","ids":["R0001"]}\n```',
            'The evidence is sufficient.\n' + finish() + '\nDone.',
        ])
        with ProjectTemporaryDirectory() as temporary:
            run_root = temporary / "run"
            outcome = self._runner(run_root, client).run()
            state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(outcome.status, "complete")
        self.assertEqual(len(client.calls), 3)
        self.assertEqual([record["action"]["action"] for record in state["records"]],
                         ["search", "read"])
        self.assertTrue(all(record["format_retry"] == 0 for record in state["records"]))

    def test_silentswap_finish_format_example_has_five_complete_slots(self) -> None:
        example = json.loads(_finish_format_example("silentswap"))

        self.assertEqual(example["action"], "finish")
        swaps = example["prediction"]["swaps"]
        self.assertEqual(len(swaps), 5)
        for swap in swaps:
            for line_range in swap["target"]["line_ranges"]:
                self.assertEqual(set(line_range), {"start", "end"})

    def test_malformed_finish_envelope_cannot_become_read(self) -> None:
        client = ScriptedClient(
            [
                '{"action":"read","ids":["R0001"]}',
                '{"action":"finish","prediction":[]}',
                '{"action":"read","ids":["R0001"]}',
                finish(),
            ]
        )
        with ProjectTemporaryDirectory() as temporary:
            run_root = temporary / "run"
            outcome = self._runner(run_root, client).run()
            state = json.loads(
                (run_root / "state.json").read_text(encoding="utf-8")
            )

        self.assertEqual(outcome.status, "complete")
        self.assertEqual(outcome.turns, 2)
        self.assertEqual(len(state["records"]), 1)
        self.assertEqual(state["records"][0]["action"]["action"], "read")
        self.assertTrue(
            all(
                item["category"] == "finish_schema"
                for item in state["terminal_record"]["format_errors"]
            )
        )

    def test_format_repairs_stop_after_two_correction_calls(self) -> None:
        client = ScriptedClient(["bad", "still bad", "bad again"])
        with ProjectTemporaryDirectory() as temporary:
            run_root = temporary / "run"
            outcome = self._runner(run_root, client).run()
            state = json.loads(
                (run_root / "state.json").read_text(encoding="utf-8")
            )

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure, "action_format_retries_exhausted")
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(state["records"], [])
        self.assertEqual(len(state["terminal_record"]["format_errors"]), 3)

    def test_format_repair_budget_is_shared_by_the_whole_sample(self) -> None:
        client = ScriptedClient(
            [
                "bad turn one",
                '{"action":"read","ids":["R0001"]}',
                "bad turn two",
                '{"action":"search","text":"pkg"}',
                "bad turn three",
            ]
        )
        with ProjectTemporaryDirectory() as temporary:
            outcome = self._runner(temporary / "run", client).run()

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.turns, 3)
        self.assertEqual(outcome.failure, "action_format_retries_exhausted")
        self.assertEqual(len(client.calls), 5)

    def test_persisted_response_is_reused_without_duplicate_model_call(self) -> None:
        first = '{"action":"read","ids":["R0001"]}'
        client = ScriptedClient([finish()])
        with ProjectTemporaryDirectory() as temporary:
            run_root = temporary / "run"
            store = RunStore(run_root)
            runner = self._runner(run_root, client)
            runner._load_or_initialize()
            prepared = prepare_messages(
                system=runner.system,
                initial_user=runner.initial_user,
                history=[],
                max_rounds=runner.max_rounds,
                config=runner.config,
                counter=runner.counter,
                priority_groups=runner.backend.priority_groups,
            )
            store.save_request(
                1,
                messages=list(prepared.messages),
                token_count=prepared.token_count,
                compression=prepared.manifest.to_dict(),
                provider_retry=0,
            )
            store.save_response(
                1,
                content=first,
                raw_response={"choices": []},
                usage={},
                provider_retry=0,
            )
            (run_root / "responses" / "turn_001.txt").unlink()
            outcome = runner.run()
            repaired = (run_root / "responses" / "turn_001.txt").read_text(
                encoding="utf-8"
            )

        self.assertEqual(outcome.status, "complete")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(repaired, first)

    def test_persisted_format_repair_is_reused_without_duplicate_call(self) -> None:
        client = ScriptedClient([finish()])
        with ProjectTemporaryDirectory() as temporary:
            run_root = temporary / "run"
            store = RunStore(run_root)
            runner = self._runner(run_root, client)
            runner._load_or_initialize()
            base = prepare_messages(
                system=runner.system,
                initial_user=runner.initial_user,
                history=[],
                max_rounds=runner.max_rounds,
                config=runner.config,
                counter=runner.counter,
                priority_groups=runner.backend.priority_groups,
            )
            rejected = "not json"
            store.save_request(
                1,
                messages=list(base.messages),
                token_count=base.token_count,
                compression=base.manifest.to_dict(),
                provider_retry=0,
            )
            store.save_response(
                1,
                content=rejected,
                raw_response={},
                usage={"prompt_tokens": 3},
                provider_retry=0,
            )
            repair = runner._apply_repair_context(
                base,
                format_retry=1,
                repair_context=(
                    rejected,
                    "action_protocol",
                    "model response must be one bare JSON object",
                ),
            )
            store.save_request(
                1,
                messages=list(repair.messages),
                token_count=repair.token_count,
                compression=repair.manifest.to_dict(),
                provider_retry=0,
                format_retry=1,
            )
            store.save_response(
                1,
                content='{"action":"read","ids":["R0001"]}',
                raw_response={},
                usage={"prompt_tokens": 4},
                provider_retry=0,
                format_retry=1,
            )
            outcome = runner.run()
            state = json.loads(
                (run_root / "state.json").read_text(encoding="utf-8")
            )

        self.assertEqual(outcome.status, "complete")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(state["records"][0]["format_retry"], 1)

    def test_unread_location_makes_finish_fail_without_model_repair(self) -> None:
        client = ScriptedClient([finish()])
        with ProjectTemporaryDirectory() as temporary:
            outcome = self._runner(temporary / "run", client).run()

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure, "invalid_finish_grounding_or_content")
        self.assertEqual(len(client.calls), 1)

    def test_last_round_rejects_non_finish(self) -> None:
        client = ScriptedClient(['{"action":"read","ids":["R0001"]}'])
        with ProjectTemporaryDirectory() as temporary:
            outcome = self._runner(
                temporary / "run", client, max_rounds=1
            ).run()

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure, "last_round_requires_finish")

    def test_parameter_error_is_repaired_without_consuming_a_round(self) -> None:
        client = ScriptedClient(
            [
                '{"action":"read","ids":[]}',
                '{"action":"read","ids":["R0001"]}',
                finish(),
            ]
        )
        with ProjectTemporaryDirectory() as temporary:
            run_root = temporary / "run"
            outcome = self._runner(run_root, client).run()
            state = json.loads(
                (run_root / "state.json").read_text(encoding="utf-8")
            )

        self.assertEqual(outcome.turns, 2)
        self.assertEqual(state["records"][0]["tool_result"]["status"], "ok")
        self.assertEqual(state["records"][0]["format_retry"], 1)
        self.assertEqual(len(state["records"][0]["format_errors"]), 1)

    def test_network_retry_reuses_identical_messages(self) -> None:
        client = RetryOnceClient(
            ['{"action":"read","ids":["R0001"]}', finish()]
        )
        with ProjectTemporaryDirectory() as temporary:
            outcome = self._runner(temporary / "run", client).run()

        self.assertEqual(outcome.status, "complete")
        self.assertEqual(client.calls[0], client.calls[1])

    def test_network_failure_stops_after_two_retries(self) -> None:
        client = AlwaysRetryClient()
        with ProjectTemporaryDirectory() as temporary:
            run_root = temporary / "run"
            outcome = self._runner(run_root, client).run()
            failures = sorted((run_root / "provider_failures").glob("*.json"))

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure, "network_retries_exhausted")
        self.assertEqual(len(client.calls), 3)
        self.assertTrue(all(call == client.calls[0] for call in client.calls))
        self.assertEqual(len(failures), 3)

    def test_resume_rejects_a_different_model_profile(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            run_root = temporary / "run"
            self._runner(run_root, ScriptedClient([finish()]))._load_or_initialize()
            changed = self._runner(run_root, OtherProfileClient([finish()]))
            with self.assertRaises(AgentLoopError):
                changed.run()

    def test_resume_rejects_a_different_finish_contract(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            run_root = temporary / "run"
            self._runner(run_root, ScriptedClient([finish()]))._load_or_initialize()
            changed = self._runner(
                run_root,
                ScriptedClient([finish()]),
                finish_contract=GroundedFinishContract("v2"),
            )
            with self.assertRaises(AgentLoopError):
                changed.run()

    def test_provider_context_retry_is_smaller_and_keeps_latest_result(self) -> None:
        client = ContextOnceClient(
            [
                '{"action":"search","text":"pkg"}',
                '{"action":"read","ids":["R0001"]}',
                finish(),
            ],
            reject_call=3,
        )
        with ProjectTemporaryDirectory() as temporary:
            outcome = self._runner(temporary / "run", client).run()

        normal = "\n".join(item["content"] for item in client.calls[2])
        forced = "\n".join(item["content"] for item in client.calls[3])
        counter = FastTokenCounter(DEFAULT_CONFIG.token_estimator)

        self.assertEqual(outcome.status, "complete")
        self.assertEqual(len(client.calls), 4)
        self.assertIn("[[SOURCE UNIT]]", normal)
        self.assertIn("[[SOURCE UNIT]]", forced)
        self.assertIn("[[SEARCH RESULT]]", normal)
        self.assertIn("[[SEARCH RECEIPT]]", forced)
        self.assertLess(
            counter.count_messages(client.calls[3]),
            counter.count_messages(client.calls[2]),
        )

if __name__ == "__main__":
    unittest.main()
