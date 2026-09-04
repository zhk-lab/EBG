from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentloop.evidence import render_tool_result
from agentloop.graph_backend import GraphBackend, _render_compact_local_graph
from agentloop.errors import BackendError
from beg.behavior_atomization import build_behaviors
from beg.behavior_directory import build_ranked_directory
from beg.evidence_intake import build_evidence, load_visible_bundle
from beg.graph_assembly import build_graph
from beg.local_graph_retrieval import LocalGraphRetriever, RetrievalError
from beg.relation_linking import build_edges
from tests.support import ProjectTemporaryDirectory, make_repo_bundle, make_trace_bundle


SCHEMA = json.loads(
    (PROJECT_ROOT / "schemas" / "local_graph.schema.json").read_text(
        encoding="utf-8"
    )
)
VALIDATE_LOCAL_GRAPH = Draft202012Validator(SCHEMA).validate


def assemble(root: Path) -> tuple[object, dict, dict]:
    bundle = load_visible_bundle(root)
    evidence = build_evidence(bundle)
    behaviors = build_behaviors(bundle, evidence)
    edges = build_edges(bundle, evidence, behaviors)
    graph = build_graph(bundle, evidence, behaviors, edges)
    directory = build_ranked_directory(bundle, graph, count_tokens=len)
    return bundle, graph, directory


def repository() -> dict[str, str]:
    return {
        "pkg/api.py": (
            "from pkg.helper import helper\n\n"
            "def run(flag):\n"
            "    if flag:\n"
            "        return helper(flag)\n"
            "    raise ValueError('disabled')\n\n"
            "def secondary():\n"
            "    return 'secondary'\n"
        ),
        "pkg/helper.py": (
            "from pkg.output import emit\n\n"
            "def helper(value):\n"
            "    return emit(value)\n"
        ),
        "pkg/output.py": "def emit(value):\n    return value\n",
    }


class LocalGraphRetrievalTests(unittest.TestCase):
    def test_read_builds_complete_root_and_bounded_one_hop_paths(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph, directory = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files=repository(),
                    document="# API\nCall `run()` from pkg/api.py.\n",
                )
            )
            retriever = LocalGraphRetriever(
                bundle, graph, directory, count_tokens=len
            )
            read_id = next(
                item["read_id"]
                for item in directory["query_index"]
                if item["path"] == "pkg/api.py"
            )
            local = retriever.read(read_id)
            VALIDATE_LOCAL_GRAPH(local)

        self.assertEqual(len(local["graphs"]), 1)
        result = local["graphs"][0]
        root = result["root"]
        self.assertEqual((root["path"], root["symbol"]), ("pkg/api.py", "run"))
        self.assertEqual(
            {item["result_type"] for item in root["behaviors"]},
            {"external_call", "return", "raise"},
        )
        self.assertEqual(
            root["source"],
            "3 | def run(flag):\n"
            "4 |     if flag:\n"
            "5 |         return helper(flag)\n"
            "6 |     raise ValueError('disabled')",
        )
        self.assertEqual(len(result["paths"]), 2)
        first_step = result["paths"][0]["steps"][0]
        second_step = result["paths"][1]["steps"][0]
        self.assertIn("--feeds[", first_step["edge"])
        self.assertEqual(first_step["node"]["symbol"], "helper")
        self.assertIn("--calls[", second_step["edge"])
        self.assertEqual(second_step["node"], {"ref": "pkg/helper.py::helper@3-4"})
        self.assertNotIn("frontier", result)
        serialized = json.dumps(local, ensure_ascii=False)
        self.assertNotIn('"evidence": [', serialized)
        self.assertNotIn('"content":', serialized)
        self.assertEqual(serialized.count("def helper(value):"), 1)

    def test_exact_symbol_and_behavior_search_choose_that_root(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph, directory = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files=repository(),
                    document="# API\nUse pkg/api.py.\n",
                )
            )
            retriever = LocalGraphRetriever(
                bundle, graph, directory, count_tokens=len
            )
            symbol_hit = retriever.search("secondary")[0]
            local = retriever.read(symbol_hit["read_id"])
            behavior_id = local["graphs"][0]["root"]["behaviors"][0]["behavior_id"]
            behavior_hit = retriever.search(behavior_id)[0]

        self.assertEqual(symbol_hit["match_reason"], "exact symbol")
        self.assertRegex(symbol_hit["read_id"], r"^F\d{4,}\.S\d{4,}$")
        self.assertEqual(local["graphs"][0]["root"]["symbol"], "secondary")
        self.assertEqual(behavior_hit["match_reason"], "exact behavior ID")
        self.assertEqual(behavior_hit["matched_symbols"], ["secondary"])

    def test_multiple_roots_share_node_deduplication_across_the_read(self) -> None:
        files = {
            "pkg/api.py": (
                "from pkg.helper import helper\n\n"
                "def first():\n    return helper()\n\n"
                "def second():\n    return helper()\n"
            ),
            "pkg/helper.py": "def helper():\n    return 1\n",
        }
        document = "# API\nCall both `pkg.api.first()` and `pkg.api.second()`.\n"
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph, directory = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files=files,
                    document=document,
                )
            )
            retriever = LocalGraphRetriever(
                bundle, graph, directory, count_tokens=len
            )
            read_id = next(
                item["read_id"]
                for item in directory["query_index"]
                if item["path"] == "pkg/api.py"
            )
            local = retriever.read(read_id)
            VALIDATE_LOCAL_GRAPH(local)

        self.assertEqual(len(local["graphs"]), 2)
        serialized = json.dumps(local, ensure_ascii=False)
        self.assertEqual(serialized.count("def helper():"), 1)
        repeated = local["graphs"][1]["paths"][0]["steps"][0]["node"]
        self.assertEqual(repeated, {"ref": "pkg/helper.py::helper@1-2"})

    def test_file_read_stops_before_next_root_exceeds_atomic_budget(self) -> None:
        files = {
            "pkg/api.py": (
                "def first():\n    return 1\n\n"
                "def second():\n    return 2\n\n"
                "def third():\n    return 3\n"
            )
        }
        document = (
            "# API\nUse `pkg.api.first()`, `pkg.api.second()`, and "
            "`pkg.api.third()`.\n"
        )
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph, directory = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files=files,
                    document=document,
                )
            )
            retriever = LocalGraphRetriever(
                bundle, graph, directory, count_tokens=len
            )
            local = retriever.read(
                "F0001",
                token_budget=2,
                measure_tokens=lambda value: len(value["graphs"]),
            )
            VALIDATE_LOCAL_GRAPH(local)

        self.assertEqual(
            [item["root"]["symbol"] for item in local["graphs"]],
            ["first", "second"],
        )

    def test_root_expands_at_most_four_direct_edges(self) -> None:
        imports = "\n".join(
            f"from pkg.dep{index} import dep{index}" for index in range(6)
        )
        calls = "\n".join(f"    dep{index}()" for index in range(6))
        files = {
            "pkg/api.py": f"{imports}\n\ndef run():\n{calls}\n    return 1\n",
            **{
                f"pkg/dep{index}.py": f"def dep{index}():\n    return {index}\n"
                for index in range(6)
            },
        }
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph, directory = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files=files,
                    document="# API\nCall `pkg.api.run()`.\n",
                )
            )
            retriever = LocalGraphRetriever(
                bundle, graph, directory, count_tokens=len
            )
            read_id = next(
                item["read_id"]
                for item in directory["query_index"]
                if item["path"] == "pkg/api.py"
            )
            local = retriever.read(read_id)

        self.assertEqual(len(local["graphs"][0]["paths"]), 4)

    def test_rejects_unknown_root_tampering_and_trace(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph, directory = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files={"app.py": "def run():\n    return 1\n"},
                )
            )
            retriever = LocalGraphRetriever(
                bundle, graph, directory, count_tokens=len
            )
            with self.assertRaisesRegex(RetrievalError, "not in app.py"):
                retriever.read("F0001", root_symbols=["missing"])

            changed = copy.deepcopy(directory)
            changed["query_index"][0]["root_symbols"] = ["missing"]
            with self.assertRaisesRegex(RetrievalError, "valid Ranked Directory"):
                LocalGraphRetriever(bundle, graph, changed, count_tokens=len)

            trace_root = make_trace_bundle(
                temporary,
                [
                    {
                        "event_type": "user_prompt",
                        "turn_number": 1,
                        "content": "Do it",
                    },
                    {
                        "event_type": "assistant_response",
                        "turn_number": 2,
                        "content": "Done",
                    },
                ],
            )
            trace = load_visible_bundle(trace_root)
            with self.assertRaisesRegex(RetrievalError, "only Repo"):
                LocalGraphRetriever(trace, {}, {})

    def test_graph_backend_renders_aligned_compact_and_tracks_cross_file_spans(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph, directory = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files=repository(),
                    document="# API\nCall `run()` from pkg/api.py.\n",
                )
            )
            backend = GraphBackend(
                bundle, graph, directory, count_tokens=len
            )
            search = backend.search("run", limit=12)
            result = backend.read(
                [search.hits[0].unit_id],
                token_budget=100_000,
                max_atomic_unit_tokens=200_000,
                count_tokens=len,
            )
            rendered = render_tool_result(result)

        self.assertIn("[[LINEAR LOCAL GRAPH]]", rendered)
        self.assertIn("[DIRECT ROOT]", rendered)
        self.assertIn("Doc: API", rendered)
        self.assertIn("pkg/api.py::run@3-6", rendered)
        self.assertIn("[CONTEXT via calls]", rendered)
        self.assertIn("run@5 calls pkg/helper.py::helper@3-4", rendered)
        self.assertNotIn('"behavior_id"', rendered)
        self.assertNotIn('"graphs"', rendered)
        self.assertNotIn("[[FILE EVIDENCE]]", rendered)
        self.assertEqual(
            {span.path for span in result.displayed_spans},
            {"pkg/api.py", "pkg/helper.py"},
        )
        self.assertTrue(backend.has_id("F0001"))
        self.assertEqual(backend.initial_index_tokens, directory["token_count"])

    def test_graph_backend_rejects_a_stale_directory(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph, directory = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files={"app.py": "def run():\n    return 1\n"},
                )
            )
            directory["query_index"][0]["path"] = "other.py"
            with self.assertRaises(BackendError):
                GraphBackend(bundle, graph, directory, count_tokens=len)

    def test_empty_local_graph_does_not_claim_unseen_source_as_grounding(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph, directory = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files={"imports.py": "import os\n"},
                )
            )
            backend = GraphBackend(bundle, graph, directory, count_tokens=len)
            result = backend.read(
                ["F0001"],
                token_budget=10_000,
                max_atomic_unit_tokens=20_000,
                count_tokens=len,
            )

        self.assertEqual(result.displayed_spans, ())
        rendered = render_tool_result(result)
        self.assertIn("[[LINEAR LOCAL GRAPH]]", rendered)
        self.assertNotIn("[DIRECT ROOT]", rendered)

    def test_default_graph_backend_falls_back_to_complete_callee_symbol(self) -> None:
        files = {
            "pkg/api.py": (
                "from pkg.worker import convert\n\n"
                "def run(flag):\n"
                "    return convert(flag)\n"
            ),
            "pkg/worker.py": (
                "def convert(value):\n"
                "    if value:\n"
                "        return value + 1\n"
                "    raise ValueError('invalid')\n"
            ),
        }
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph, directory = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files=files,
                    document="# API\nCall `run()` from pkg/api.py.\n",
                )
            )
            backend = GraphBackend(bundle, graph, directory, count_tokens=len)
            read_id = next(
                item["read_id"]
                for item in directory["query_index"]
                if item["path"] == "pkg/api.py"
            )
            result = backend.read(
                [read_id],
                token_budget=100_000,
                max_atomic_unit_tokens=200_000,
                count_tokens=len,
            )
            rendered = render_tool_result(result)

        self.assertIn("run@4 calls pkg/worker.py::convert@1-4", rendered)
        self.assertIn("return value + 1", rendered)
        self.assertIn("raise ValueError('invalid')", rendered)

    def test_behavior_neighbor_keeps_matched_branch_and_default_outcome(self) -> None:
        files = {
            "pkg/gate.py": (
                "class Gate:\n"
                "    def __init__(self):\n"
                "        self.enabled = True\n\n"
                "    def choose(self):\n"
                "        if self.enabled:\n"
                "            return 1\n"
                "        return 0\n"
            )
        }
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph, directory = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files=files,
                    document="# Gate\nInitialize `Gate.__init__()`.\n",
                )
            )
            backend = GraphBackend(bundle, graph, directory, count_tokens=len)
            read_id = next(
                item["read_id"]
                for item in directory["query_index"]
                if item["path"] == "pkg/gate.py"
            )
            local = backend.local_graph(
                read_id, root_symbols=("Gate.__init__",)
            )

        choose = next(
            step["node"]
            for graph_item in local["graphs"]
            for path in graph_item["paths"]
            for step in path["steps"]
            if "Gate.choose" in step["edge"] and "ref" not in step["node"]
        )
        self.assertIn("return 1", choose["source"])
        self.assertIn("return 0", choose["source"])

    def test_compact_rendering_deduplicates_same_file_lines(self) -> None:
        rendered = _render_compact_local_graph(
            {
                "graphs": [
                    {
                        "root": {
                            "path": "same.py",
                            "symbol": "root",
                            "source": "1 | first\n2 | shared",
                        },
                        "paths": [
                            {
                                "steps": [
                                    {
                                        "edge": "root@2 calls neighbor@2-3",
                                        "node": {
                                            "path": "same.py",
                                            "symbol": "neighbor",
                                            "source": "2 | shared\n3 | last",
                                        },
                                    }
                                ]
                            }
                        ],
                    }
                ]
            }
        )

        self.assertEqual(rendered.count("2 | shared"), 1)
        self.assertIn("3 | last", rendered)

    def test_direct_symbol_root_precedes_evidence_only_root(self) -> None:
        files = {
            "pkg/worker.py": (
                "def convert(value):\n"
                "    adjusted = value + 1\n"
                "    return adjusted\n\n"
                "def notify(value):\n"
                "    print(value)\n"
            )
        }
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph, directory = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files=files,
                    document="# Worker\nCall `notify()` after computing `adjusted`.\n",
                )
            )
            backend = GraphBackend(bundle, graph, directory, count_tokens=len)

        self.assertEqual(
            backend._retriever.root_symbols_by_path["pkg/worker.py"],
            ("notify", "convert"),
        )

    def test_module_root_contains_only_document_matched_segment(self) -> None:
        files = {"pkg/settings.py": "FIRST = 1\nSECOND = 2\n"}
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph, directory = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files=files,
                    document="# Settings\n`FIRST` controls the public mode.\n",
                )
            )
            backend = GraphBackend(bundle, graph, directory, count_tokens=len)
            read_id = next(
                item["read_id"]
                for item in directory["query_index"]
                if item["path"] == "pkg/settings.py"
            )
            result = backend.read(
                [read_id],
                token_budget=100_000,
                max_atomic_unit_tokens=200_000,
                count_tokens=len,
            )
            rendered = render_tool_result(result)

        self.assertIn("1 | FIRST = 1", rendered)
        self.assertNotIn("2 | SECOND = 2", rendered)


if __name__ == "__main__":
    unittest.main()
