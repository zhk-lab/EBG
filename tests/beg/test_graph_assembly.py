from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beg.behavior_atomization import build_behaviors
from beg.core.errors import GraphError
from beg.evidence_intake import build_evidence, load_visible_bundle
from beg.graph_assembly import build_graph, validate_graph
from beg.relation_linking import build_edges
from tests.support import ProjectTemporaryDirectory, make_repo_bundle, make_trace_bundle


SCHEMA = json.loads(
    (PROJECT_ROOT / "schemas" / "graph.schema.json").read_text(encoding="utf-8")
)
VALIDATE_GRAPH = Draft202012Validator(SCHEMA).validate


def assemble_repo(
    temporary: Path,
    files: dict[str, str],
) -> tuple[object, list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    bundle = load_visible_bundle(make_repo_bundle(temporary, repository_files=files))
    evidence = build_evidence(bundle)
    behaviors = build_behaviors(bundle, evidence)
    edges = build_edges(bundle, evidence, behaviors)
    graph = build_graph(bundle, evidence, behaviors, edges)
    VALIDATE_GRAPH(graph)
    return bundle, evidence, behaviors, edges, graph


class GraphAssemblyTests(unittest.TestCase):
    def test_repo_graph_preserves_upstream_facts_without_symbol_nodes(self) -> None:
        source = (
            "def choose(flag):\n"
            "    if flag:\n"
            "        return 1\n"
            "    raise ValueError('bad')\n\n"
            "def wrapper(flag):\n"
            "    return choose(flag)\n"
        )
        with ProjectTemporaryDirectory() as temporary:
            bundle, evidence, behaviors, edges, graph = assemble_repo(
                temporary, {"src/app.py": source}
            )

        self.assertEqual(
            set(graph),
            {"input_id", "benchmark", "evidence", "behaviors", "source_contexts", "edges"},
        )
        self.assertEqual(graph["evidence"], evidence)
        self.assertEqual(graph["behaviors"], behaviors)
        self.assertEqual(graph["edges"], edges)
        serialized = json.dumps(graph)
        self.assertNotIn("symbol_id", serialized)
        self.assertNotIn("behavior_name", serialized)
        choose = [
            item for item in graph["source_contexts"] if item["symbol"] == "choose"
        ]
        self.assertEqual(len(choose), 1)
        self.assertEqual(
            choose[0]["source"],
            "def choose(flag):\n"
            "    if flag:\n"
            "        return 1\n"
            "    raise ValueError('bad')\n",
        )
        validate_graph(bundle, evidence, behaviors, edges, graph)

    def test_configuration_and_template_contexts_are_exact_and_deduplicated(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            _, _, _, _, graph = assemble_repo(
                temporary,
                {
                    "settings.toml": "[runtime]\nmode = 'safe'\n",
                    "templates/page.jinja2": (
                        "{% block body %}\nHello {{ name }}\n{% endblock %}\n"
                    ),
                },
            )

        contexts = graph["source_contexts"]
        template = next(item for item in contexts if item["artifact_kind"] == "runtime_template")
        self.assertEqual(template["symbol"], "body")
        self.assertEqual(
            template["source"],
            "{% block body %}\nHello {{ name }}\n{% endblock %}\n",
        )
        identities = {
            (item["path"], item["symbol"], tuple(item["lines"])) for item in contexts
        }
        self.assertEqual(len(identities), len(contexts))

    def test_trace_graph_keeps_evidence_behaviors_and_scope_edges_normalized(self) -> None:
        events = [
            {"event_type": "system", "turn_number": 0, "content": "Context"},
            {"event_type": "user_prompt", "turn_number": 1, "content": "Do it"},
            {"event_type": "assistant_response", "turn_number": 2, "content": "Done"},
            {"event_type": "user_prompt", "turn_number": 3, "content": "Check"},
            {"event_type": "assistant_response", "turn_number": 4, "content": "Checked"},
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=5))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)
            edges = build_edges(bundle, evidence, behaviors)
            graph = build_graph(bundle, evidence, behaviors, edges)
            VALIDATE_GRAPH(graph)

        self.assertEqual(
            set(graph), {"input_id", "benchmark", "evidence", "behaviors", "edges"}
        )
        self.assertEqual(graph["evidence"], evidence)
        self.assertEqual(graph["behaviors"], behaviors)
        self.assertEqual(graph["evidence"][0]["locator"]["event_type"], "system")
        referenced = {
            evidence_id
            for behavior in behaviors
            for evidence_id in [
                *(item["evidence_id"] for item in behavior["demand_refs"]),
                *behavior["action_evidence_ids"],
                *(item["evidence_id"] for item in behavior["response_refs"]),
            ]
        }
        self.assertNotIn(graph["evidence"][0]["evidence_id"], referenced)

    def test_trace_graph_preserves_internal_task_edges(self) -> None:
        events = [
            {"event_type": "user_prompt", "turn_number": 1, "content": "Run it."},
            {
                "event_type": "assistant_response",
                "turn_number": 2,
                "content": "Started job ID eval-42.",
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
            },
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=5))
            evidence = build_evidence(bundle)
            behaviors = build_behaviors(bundle, evidence)
            edges = build_edges(bundle, evidence, behaviors)
            graph = build_graph(bundle, evidence, behaviors, edges)
            VALIDATE_GRAPH(graph)

        self.assertEqual(len(graph["edges"]), 2)
        self.assertEqual(graph["edges"][0]["type"], "precedes")
        self.assertEqual(graph["edges"][1]["type"], "references")
        self.assertEqual(graph["edges"][1]["source_task_id"], "K0002")
        self.assertEqual(graph["edges"][1]["target_task_id"], "K0001")

    def test_repo_with_no_observable_behavior_still_preserves_evidence(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            _, evidence, behaviors, edges, graph = assemble_repo(
                temporary, {"src/imports.py": "import os\n"}
            )

        self.assertTrue(evidence)
        self.assertEqual(behaviors, [])
        self.assertEqual(edges, [])
        self.assertEqual(graph["source_contexts"], [])

    def test_validation_rejects_changed_or_duplicated_source_context(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle, evidence, behaviors, edges, graph = assemble_repo(
                temporary, {"src/app.py": "def run():\n    return 1\n"}
            )
            changed = copy.deepcopy(graph)
            changed["source_contexts"][0]["source"] = "changed"
            with self.assertRaises(GraphError):
                validate_graph(bundle, evidence, behaviors, edges, changed)
            duplicated = copy.deepcopy(graph)
            duplicated["source_contexts"].append(
                copy.deepcopy(duplicated["source_contexts"][0])
            )
            with self.assertRaises(GraphError):
                validate_graph(bundle, evidence, behaviors, edges, duplicated)

    def test_repeated_assembly_is_deterministic(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle, evidence, behaviors, edges, graph = assemble_repo(
                temporary,
                {
                    "src/app.py": (
                        "def helper():\n    return 1\n\n"
                        "def run():\n    return helper()\n"
                    )
                },
            )
            self.assertEqual(graph, build_graph(bundle, evidence, behaviors, edges))


if __name__ == "__main__":
    unittest.main()
