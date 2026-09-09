from __future__ import annotations

import json
import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentloop import RunStore
from agentloop.provider import ModelCompletion
from agentloop.errors import AgentLoopError
from tracereview import TraceReview, render_trace_view
from evaluation_core.contracts import load_prediction_schema
from evaluation_core.messages import build_trace_review_messages, load_task_prompt
from beg.behavior_atomization import build_behaviors
from beg.evidence_intake import build_evidence, load_visible_bundle
from beg.graph_assembly import build_graph
from beg.relation_linking import build_edges
from tests.support import ProjectTemporaryDirectory, make_trace_bundle


def trace_messages(view, _schema):
    return build_trace_review_messages(
        input_id="ft_001_long",
        task_prompt=load_task_prompt("BEG", "feedbacktrace"),
        trace_view=view.text,
    )


class OneShotClient:
    def __init__(self, value: dict) -> None:
        self.content = json.dumps(value)
        self.calls = 0

    @property
    def profile(self) -> dict:
        return {"provider": "scripted", "model": "fake"}

    def complete(self, messages, *, max_output_tokens):
        self.calls += 1
        return ModelCompletion(
            content=self.content,
            raw_response={"choices": [{"message": {"content": self.content}}]},
            usage={},
        )


def trace_graph() -> dict:
    tool = lambda turn, evidence_id: {
        "event_type": "tool_exchange",
        "turn_number": turn,
        "content": (
            'Tool invocation:\n{"input": {"file_path": "C:\\\\work\\\\a.py"}, '
            '"tool_name": "Read"}\n\nTool result: Read\nThe fallback is now disabled.'
        ),
        "evidence_id": evidence_id,
        "tool_name": "Read",
    }
    events = [
        {"event_type": "system", "turn_number": 0, "content": "System context."},
        {"event_type": "user_prompt", "turn_number": 1, "content": "Please inspect it."},
        tool(2, "e_2"),
        {
            "event_type": "assistant_response",
            "turn_number": 3,
            "content": "I changed the fallback.",
            "evidence_id": "e_3",
        },
        {"event_type": "user_prompt", "turn_number": 4, "content": "Why?"},
        {
            "event_type": "assistant_response",
            "turn_number": 5,
            "content": "Because it failed.",
            "evidence_id": "e_5",
        },
        {"event_type": "user_prompt", "turn_number": 6, "content": "Check again."},
        tool(7, "e_7"),
        {
            "event_type": "assistant_response",
            "turn_number": 8,
            "content": "Confirmed.",
            "evidence_id": "e_8",
        },
    ]
    with ProjectTemporaryDirectory() as temporary:
        bundle = load_visible_bundle(
            make_trace_bundle(
                temporary,
                events,
                input_id="ft_trace_long",
                cutoff=9,
            )
        )
        evidence = build_evidence(bundle)
        interactions = build_behaviors(bundle, evidence)
        edges = build_edges(bundle, evidence, interactions)
        return build_graph(bundle, evidence, interactions, edges)


class TraceViewTests(unittest.TestCase):
    def test_trace_order_and_original_evidence_ids_are_preserved(self) -> None:
        view = render_trace_view(trace_graph())
        payload = json.loads(view.text)

        self.assertEqual(
            list(payload),
            ["input_id", "benchmark", "current_task_id", "task_scopes"],
        )
        self.assertEqual(payload["current_task_id"], "K0001")
        self.assertEqual(len(payload["task_scopes"]), 1)
        scope = payload["task_scopes"][0]
        self.assertEqual(scope["task_id"], "K0001")
        self.assertEqual(scope["status"], "current")
        self.assertEqual(
            [item["behavior_id"] for item in scope["behaviors"]],
            ["T0001", "T0002", "T0003"],
        )
        self.assertEqual(
            [item["sequence_index"] for item in scope["behaviors"]],
            [1, 2, 3],
        )
        self.assertEqual(scope["relations"], [])
        self.assertNotIn("events", scope)
        self.assertEqual(view.text.count("I changed the fallback."), 1)
        self.assertNotIn("System context.", view.text)
        self.assertEqual(
            view.evidence_ids, frozenset({"e_2", "e_3", "e_5", "e_7", "e_8"})
        )
        behavior = scope["behaviors"][0]
        self.assertNotIn("events", behavior)
        self.assertEqual(behavior["demand"][0]["content"], "Please inspect it.")
        self.assertEqual(behavior["action"][0]["evidence_id"], "e_2")
        self.assertEqual(behavior["action"][0]["tool_name"], "Read")
        self.assertNotIn("original_evidence_id", view.text)
        self.assertNotIn('"E000', view.text)
        self.assertNotIn("interaction_id", view.text)
        self.assertNotIn('"interactions"', view.text)

    def test_current_scope_contains_the_final_visible_behavior(self) -> None:
        graph = copy.deepcopy(trace_graph())
        graph["behaviors"][0]["task_id"] = "K0001"
        graph["behaviors"][0]["sequence_index"] = 1
        graph["behaviors"][1]["task_id"] = "K0002"
        graph["behaviors"][1]["sequence_index"] = 1
        graph["behaviors"][2]["task_id"] = "K0001"
        graph["behaviors"][2]["sequence_index"] = 2

        payload = json.loads(render_trace_view(graph).text)

        self.assertEqual(payload["current_task_id"], "K0001")
        self.assertEqual(
            [(scope["task_id"], scope["status"]) for scope in payload["task_scopes"]],
            [("K0001", "current"), ("K0002", "history")],
        )

    def test_task_scopes_keep_first_appearance_order(self) -> None:
        graph = copy.deepcopy(trace_graph())
        for index, behavior in enumerate(graph["behaviors"], start=1):
            behavior["task_id"] = f"K{index:04d}"
            behavior["sequence_index"] = 1

        view = render_trace_view(graph)
        payload = json.loads(view.text)

        self.assertEqual(payload["current_task_id"], "K0003")
        self.assertEqual(
            [(scope["task_id"], scope["status"]) for scope in payload["task_scopes"]],
            [("K0001", "history"), ("K0002", "history"), ("K0003", "current")],
        )
        self.assertEqual(
            view.evidence_ids, frozenset({"e_2", "e_3", "e_5", "e_7", "e_8"})
        )

    def test_same_turn_behaviors_keep_demand_order_when_end_turns_interleave(self) -> None:
        graph = copy.deepcopy(trace_graph())
        first, second = graph["behaviors"][:2]
        first["start_turn"] = second["start_turn"] = 1
        first["end_turn"] = max(first["end_turn"], second["end_turn"] + 1)

        payload = json.loads(render_trace_view(graph).text)

        self.assertEqual(
            [item["behavior_id"] for item in payload["task_scopes"][0]["behaviors"]],
            ["T0001", "T0002", "T0003"],
        )

    def test_unknown_task_scope_edge_is_rejected(self) -> None:
        graph = copy.deepcopy(trace_graph())
        graph["edges"] = [{
            "edge_id": "R0001",
            "source_task_id": "K0001",
            "type": "informs",
            "target_task_id": "K9999",
            "evidence_ids": [graph["behaviors"][0]["response_refs"][0]["evidence_id"]],
        }]

        with self.assertRaisesRegex(AgentLoopError, "Task Scope"):
            render_trace_view(graph)

    def test_user_only_evidence_without_behavior_is_not_rendered(self) -> None:
        graph = copy.deepcopy(trace_graph())
        graph["evidence"].insert(
            1,
            {
                "evidence_id": "E999999",
                "source_type": "trace",
                "locator": {
                    "turn": 0,
                    "event_index": 1,
                    "event_type": "user_prompt",
                    "tool_name": None,
                    "original_evidence_id": None,
                },
                "content": "Standalone command with no Agent result.",
            },
        )

        view = render_trace_view(graph)

        self.assertNotIn("Standalone command with no Agent result.", view.text)
        self.assertEqual(
            json.loads(view.text)["task_scopes"][0]["behaviors"][0]["behavior_id"],
            "T0001",
        )

    def test_informs_is_nested_in_its_source_task_scope(self) -> None:
        events = [
            {"event_type": "user_prompt", "turn_number": 1, "content": "Choose the API."},
            {
                "event_type": "assistant_response",
                "turn_number": 2,
                "content": "The API choice was saved to reports/api-choice.json.",
                "evidence_id": "e_choice",
            },
            {
                "event_type": "user_prompt",
                "turn_number": 3,
                "content": (
                    "Using reports/api-choice.json, write the documentation."
                ),
            },
            {
                "event_type": "assistant_response",
                "turn_number": 4,
                "content": "The documentation is complete.",
                "evidence_id": "e_docs",
            },
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=5))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)
            edges = build_edges(bundle, evidence, behaviors)
            payload = json.loads(
                render_trace_view(build_graph(bundle, evidence, behaviors, edges)).text
            )

        self.assertEqual(
            [scope["task_id"] for scope in payload["task_scopes"]],
            ["K0001", "K0002"],
        )
        informs = payload["task_scopes"][0]["relations"]
        self.assertEqual(len(informs), 1)
        self.assertEqual(informs[0]["target_task_id"], "K0002")
        self.assertEqual(informs[0]["support"][0]["evidence_id"], "e_choice")
        self.assertEqual(payload["task_scopes"][1]["relations"], [])

    def test_supersedes_is_nested_in_the_replacing_task_scope(self) -> None:
        events = [
            {"event_type": "user_prompt", "turn_number": 1, "content": "Run it."},
            {
                "event_type": "assistant_response",
                "turn_number": 2,
                "content": "Started job ID eval-42.",
                "evidence_id": "e_started",
            },
            {
                "event_type": "user_prompt",
                "turn_number": 3,
                "content": "Start a new task. Stop job ID eval-42 and replace it.",
            },
            {
                "event_type": "assistant_response",
                "turn_number": 4,
                "content": "Stopped eval-42.",
                "evidence_id": "e_stopped",
            },
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=5))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)
            edges = build_edges(bundle, evidence, behaviors)
            payload = json.loads(
                render_trace_view(build_graph(bundle, evidence, behaviors, edges)).text
            )

        relation = payload["task_scopes"][1]["relations"][0]
        self.assertEqual(relation["type"], "supersedes")
        self.assertEqual(relation["target_task_id"], "K0001")
        self.assertEqual(payload["task_scopes"][0]["relations"], [])

    def test_selectable_evidence_cannot_be_omitted_with_user_only_event(self) -> None:
        graph = copy.deepcopy(trace_graph())
        graph["evidence"].insert(
            1,
            {
                "evidence_id": "E999999",
                "source_type": "trace",
                "locator": {
                    "turn": 0,
                    "event_index": 1,
                    "event_type": "user_prompt",
                    "tool_name": None,
                    "original_evidence_id": "raw-E999999",
                },
                "content": "Unexpected selectable user event.",
            },
        )

        with self.assertRaisesRegex(AgentLoopError, "selectable Evidence was omitted"):
            render_trace_view(graph)

    def test_one_shot_prediction_uses_formal_feedbacktrace_contract(self) -> None:
        client = OneShotClient(
            {
                "verification_point": "Confirm whether the fallback should remain disabled.",
                "supporting_evidence_ids": ["e_3", "e_7"],
                "criticality": "must_disclose",
            }
        )
        with ProjectTemporaryDirectory() as temporary:
            run_root = temporary / "run"
            view = render_trace_view(trace_graph())
            schema = load_prediction_schema(
                PROJECT_ROOT / "schemas", "feedbacktrace"
            )
            prediction = TraceReview(
                view=view,
                messages=trace_messages(view, schema),
                prediction_schema=schema,
                client=client,
                store=RunStore(run_root),
            ).run()
            rendered = (run_root / "trace_behavior_view.txt").read_text(
                encoding="utf-8"
            )
            model_response = json.loads(
                (run_root / "prediction_response.json").read_text(encoding="utf-8")
            )

        self.assertEqual(prediction["benchmark"], "feedbacktrace")
        self.assertEqual(prediction["verdict"], "KEY")
        self.assertNotIn("verdict", model_response)
        self.assertEqual(client.calls, 1)
        self.assertEqual(rendered, render_trace_view(trace_graph()).text)

    def test_invalid_prediction_is_persisted_as_failed(self) -> None:
        client = OneShotClient(
            {
                "verification_point": "Confirm it.",
                "supporting_evidence_ids": ["UNKNOWN"],
                "criticality": "must_disclose",
            }
        )
        with ProjectTemporaryDirectory() as temporary:
            run_root = temporary / "run"
            view = render_trace_view(trace_graph())
            schema = load_prediction_schema(
                PROJECT_ROOT / "schemas", "feedbacktrace"
            )
            runner = TraceReview(
                view=view,
                messages=trace_messages(view, schema),
                prediction_schema=schema,
                client=client,
                store=RunStore(run_root),
            )
            with self.assertRaises(AgentLoopError):
                runner.run()
            state = json.loads(
                (run_root / "state.json").read_text(encoding="utf-8")
            )

        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["failure"], "invalid_prediction")

    def test_model_owned_verdict_is_rejected(self) -> None:
        client = OneShotClient(
            {
                "verdict": "KEY",
                "verification_point": "Confirm it.",
                "supporting_evidence_ids": ["e_3"],
                "criticality": "must_disclose",
            }
        )
        with ProjectTemporaryDirectory() as temporary:
            view = render_trace_view(trace_graph())
            schema = load_prediction_schema(
                PROJECT_ROOT / "schemas", "feedbacktrace"
            )
            runner = TraceReview(
                view=view,
                messages=trace_messages(view, schema),
                prediction_schema=schema,
                client=client,
                store=RunStore(temporary / "run"),
            )

            with self.assertRaisesRegex(AgentLoopError, "formal contract"):
                runner.run()

    def test_resume_rejects_changed_trace_view(self) -> None:
        client = OneShotClient(
            {
                "verification_point": "Confirm whether the fallback should remain disabled.",
                "supporting_evidence_ids": ["e_3"],
                "criticality": "must_disclose",
            }
        )
        with ProjectTemporaryDirectory() as temporary:
            run_root = temporary / "run"
            schema = load_prediction_schema(
                PROJECT_ROOT / "schemas", "feedbacktrace"
            )
            view = render_trace_view(trace_graph())
            TraceReview(
                view=view,
                messages=trace_messages(view, schema),
                prediction_schema=schema,
                client=client,
                store=RunStore(run_root),
            ).run()
            changed = replace(view, text=view.text + "changed\n")
            with self.assertRaises(AgentLoopError):
                TraceReview(
                    view=changed,
                    messages=trace_messages(changed, schema),
                    prediction_schema=schema,
                    client=client,
                    store=RunStore(run_root),
                ).run()


if __name__ == "__main__":
    unittest.main()
