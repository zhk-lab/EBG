from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ebg.behavior_atomization import build_behaviors
from ebg.core.errors import RelationError
from ebg.evidence_intake import build_evidence, load_visible_bundle
from ebg.relation_linking import build_edges, validate_edges
from tests.support import ProjectTemporaryDirectory, make_repo_bundle, make_trace_bundle


SCHEMA = json.loads(
    (PROJECT_ROOT / "schemas" / "edges.schema.json").read_text(encoding="utf-8")
)
VALIDATE_EDGES = Draft202012Validator(SCHEMA).validate


def build_repo(
    temporary: Path,
    files: dict[str, str],
) -> tuple[object, list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    bundle = load_visible_bundle(make_repo_bundle(temporary, repository_files=files))
    evidence = build_evidence(bundle)
    behaviors = build_behaviors(bundle, evidence)
    edges = build_edges(bundle, evidence, behaviors)
    VALIDATE_EDGES(edges)
    return bundle, evidence, behaviors, edges


class RelationLinkingTests(unittest.TestCase):
    def test_calls_and_result_feeds_use_scope_endpoints_and_evidence(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle, evidence, behaviors, edges = build_repo(
                temporary,
                {
                    "pkg/__init__.py": "",
                    "pkg/calc.py": "def produce(value):\n    return value + 1\n",
                    "pkg/app.py": (
                        "from .calc import produce\n\n"
                        "def run(value):\n"
                        "    result = produce(value)\n"
                        "    return result\n"
                    ),
                },
            )

        calls = next(item for item in edges if item["type"] == "calls")
        self.assertEqual(
            calls["from"], {"path": "pkg/app.py", "symbol": "run"}
        )
        self.assertEqual(
            calls["to"], {"path": "pkg/calc.py", "symbol": "produce"}
        )
        self.assertIsNone(calls["via"])
        feeds = next(item for item in edges if item["type"] == "feeds")
        self.assertEqual(feeds["from"], calls["to"])
        self.assertEqual(feeds["to"], calls["from"])
        self.assertEqual(feeds["via"], "result")
        known = {item["evidence_id"] for item in evidence}
        self.assertTrue(set(calls["evidence_ids"]) <= known)
        self.assertTrue(set(feeds["evidence_ids"]) <= known)
        validate_edges(bundle, evidence, behaviors, edges)

    def test_repeated_direct_calls_merge_support_into_one_edge(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            _, evidence, _, edges = build_repo(
                temporary,
                {
                    "src/app.py": (
                        "def helper(value):\n"
                        "    return value\n\n"
                        "def run(value):\n"
                        "    first = helper(value)\n"
                        "    second = helper(first)\n"
                        "    return second\n"
                    )
                },
            )

        calls = [item for item in edges if item["type"] == "calls"]
        self.assertEqual(len(calls), 1)
        by_id = {item["evidence_id"]: item["content"] for item in evidence}
        support = "".join(by_id[item] for item in calls[0]["evidence_ids"])
        self.assertIn("first = helper(value)", support)
        self.assertIn("second = helper(first)", support)

    def test_explicit_state_write_feeds_reader_scope(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            _, _, _, edges = build_repo(
                temporary,
                {
                    "src/box.py": (
                        "class Box:\n"
                        "    def set(self, value):\n"
                        "        self.value = value\n"
                        "        return self.value\n\n"
                        "    def get(self):\n"
                        "        return self.value\n"
                    )
                },
            )

        state = next(
            item
            for item in edges
            if item["type"] == "feeds" and item["via"] == "value"
        )
        self.assertEqual(state["from"]["symbol"], "Box.set")
        self.assertEqual(state["to"]["symbol"], "Box.get")

    def test_dynamic_and_transitive_calls_are_not_invented(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            _, _, _, edges = build_repo(
                temporary,
                {
                    "src/flow.py": (
                        "def c():\n"
                        "    return 1\n\n"
                        "def b():\n"
                        "    return c()\n\n"
                        "def a():\n"
                        "    return b()\n\n"
                        "def dynamic(obj):\n"
                        "    return obj.c()\n"
                    )
                },
            )

        call_pairs = {
            (item["from"]["symbol"], item["to"]["symbol"])
            for item in edges
            if item["type"] == "calls"
        }
        self.assertEqual(call_pairs, {("a", "b"), ("b", "c")})
        self.assertFalse(any("dynamic" in pair for pair in call_pairs))

    def test_trace_shared_path_only_adds_time_edge(self) -> None:
        tool = lambda turn: {
            "event_type": "tool_exchange",
            "turn_number": turn,
            "content": (
                'Tool invocation:\n{"input": {"file_path": "C:\\\\work\\\\a.py"}, '
                '"tool_name": "Read"}\n\nTool result: Read\nok'
            ),
            "tool_name": "Read",
        }
        events = [
            {"event_type": "user_prompt", "turn_number": 1, "content": "Implement the API."},
            tool(2),
            {
                "event_type": "assistant_response",
                "turn_number": 3,
                "content": "The API result was saved to reports/api-result.json.",
            },
            {
                "event_type": "user_prompt",
                "turn_number": 4,
                "content": (
                    "Start a new task. Using reports/api-result.json from the previous "
                    "result, document the API."
                ),
            },
            tool(5),
            {
                "event_type": "assistant_response",
                "turn_number": 6,
                "content": "The documentation is complete.",
            },
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=7))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)
            edges = build_edges(bundle, evidence, behaviors)
            VALIDATE_EDGES(edges)

        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["type"], "precedes")
        self.assertEqual(edges[0]["source_task_id"], "K0001")
        self.assertEqual(edges[0]["target_task_id"], "K0002")
        self.assertEqual(len(edges[0]["evidence_ids"]), 2)

    def test_trace_explicit_id_adds_reference_and_time_edges(self) -> None:
        events = [
            {"event_type": "user_prompt", "turn_number": 1, "content": "Start the evaluation."},
            {
                "event_type": "assistant_response",
                "turn_number": 2,
                "content": "Started job ID eval-42 and it is running.",
            },
            {
                "event_type": "user_prompt",
                "turn_number": 3,
                "content": (
                    "Start a new task. Stop job ID eval-42 and replace it with "
                    "job ID eval-43."
                ),
            },
            {
                "event_type": "assistant_response",
                "turn_number": 4,
                "content": "Stopped eval-42 and started eval-43.",
            },
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=5))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)
            edges = build_edges(bundle, evidence, behaviors)
            VALIDATE_EDGES(edges)

        self.assertEqual([edge["type"] for edge in edges], ["precedes", "references"])
        self.assertEqual(edges[1]["source_task_id"], "K0002")
        self.assertEqual(edges[1]["target_task_id"], "K0001")
        self.assertEqual(len(edges[1]["evidence_ids"]), 2)

    def test_trace_generic_reference_words_do_not_create_edges(self) -> None:
        events = [
            {"event_type": "user_prompt", "turn_number": 1, "content": "Build phase 1."},
            {"event_type": "assistant_response", "turn_number": 2, "content": "Phase 1 is ready."},
            {
                "event_type": "user_prompt",
                "turn_number": 3,
                "content": "Start a new task. From the table above, review phase 2.",
            },
            {"event_type": "assistant_response", "turn_number": 4, "content": "Phase 2 reviewed."},
            {
                "event_type": "user_prompt",
                "turn_number": 5,
                "content": "Start a new task. Aside from that, inspect the dashboard.",
            },
            {"event_type": "assistant_response", "turn_number": 6, "content": "Dashboard inspected."},
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=7))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)
            edges = build_edges(bundle, evidence, behaviors)

        self.assertEqual([e["type"] for e in edges], ["precedes", "precedes"])

    def test_trace_same_object_and_time_order_do_not_create_edges(self) -> None:
        events = [
            {"event_type": "user_prompt", "turn_number": 1, "content": "Read a.py."},
            {"event_type": "assistant_response", "turn_number": 2, "content": "Read it."},
            {"event_type": "user_prompt", "turn_number": 3, "content": "Rewrite a.py."},
            {"event_type": "assistant_response", "turn_number": 4, "content": "Rewrote it."},
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=5))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)
            edges = build_edges(bundle, evidence, behaviors)

        self.assertEqual([e["type"] for e in edges], ["precedes"])

    def test_validation_rejects_unknown_support_and_build_is_deterministic(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle, evidence, behaviors, edges = build_repo(
                temporary,
                {
                    "src/app.py": (
                        "def helper():\n    return 1\n\n"
                        "def run():\n    return helper()\n"
                    )
                },
            )
            self.assertEqual(edges, build_edges(bundle, evidence, behaviors))
            broken = copy.deepcopy(edges)
            broken[0]["evidence_ids"] = ["E999999"]
            with self.assertRaises(RelationError):
                validate_edges(bundle, evidence, behaviors, broken)


if __name__ == "__main__":
    unittest.main()
