from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beg.behavior_atomization import build_behaviors
from beg.behavior_directory import (
    DIRECTORY_TOKEN_BUDGET,
    build_ranked_directory,
    render_ranked_directory,
    validate_ranked_directory,
)
from beg.core.errors import DirectoryError
from beg.evidence_intake import build_evidence, load_visible_bundle
from beg.graph_assembly import build_graph
from beg.relation_linking import build_edges
from scripts.main.prepare import (
    build_directory_artifact,
    validate_directory_artifact,
)
from tests.support import ProjectTemporaryDirectory, make_repo_bundle, make_trace_bundle


SCHEMA = json.loads(
    (PROJECT_ROOT / "schemas" / "ranked_directory.schema.json").read_text(
        encoding="utf-8"
    )
)
VALIDATE_SCHEMA = Draft202012Validator(SCHEMA).validate


def assemble(bundle_root: Path) -> tuple[object, dict[str, object]]:
    bundle = load_visible_bundle(bundle_root)
    evidence = build_evidence(bundle)
    behaviors = build_behaviors(bundle, evidence)
    edges = build_edges(bundle, evidence, behaviors)
    graph = build_graph(bundle, evidence, behaviors, edges)
    return bundle, graph


class RankedDirectoryTests(unittest.TestCase):
    def test_repo_uses_stable_file_ids_but_ranks_document_hit_first(self) -> None:
        files = {
            "pkg/api.py": (
                "from pkg.helper import helper\n\n"
                "def run():\n"
                "    return helper()\n"
            ),
            "pkg/helper.py": "def helper():\n    return 1\n",
            "pkg/other.py": "def unrelated():\n    return 2\n",
        }
        document = "# Public API\n\nUse `pkg.api.run()` from pkg/api.py.\n"
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph = assemble(
                make_repo_bundle(temporary, repository_files=files, document=document)
            )
            directory = build_ranked_directory(bundle, graph, count_tokens=len)
            VALIDATE_SCHEMA(directory)

        self.assertEqual(
            directory["query_index"],
            [
                {"read_id": "F0001", "path": "pkg/api.py", "root_symbols": ["run"]},
                {"read_id": "F0002", "path": "pkg/helper.py", "root_symbols": ["helper"]},
                {"read_id": "F0003", "path": "pkg/other.py", "root_symbols": ["unrelated"]},
            ],
        )
        self.assertEqual(
            [entry["path"] for entry in directory["entries"][:3]],
            ["pkg/api.py", "pkg/helper.py", "pkg/other.py"],
        )
        self.assertEqual(
            directory["entries"][0]["related_document_sections"], ["Public API"]
        )
        rendered = render_ranked_directory(directory)
        self.assertNotIn("evidence_id", rendered)
        self.assertNotIn("edge_id", rendered)
        self.assertNotIn("behavior_id", rendered)

    def test_direct_files_round_robin_across_document_sections(self) -> None:
        files = {
            "src/a.py": "def a():\n    return 1\n",
            "src/b.py": "def b():\n    return 2\n",
            "src/c.py": "def c():\n    return 3\n",
        }
        document = (
            "# Alpha\nUse src/a.py and src/b.py.\n\n"
            "# Beta\nUse src/c.py.\n"
        )
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph = assemble(
                make_repo_bundle(temporary, repository_files=files, document=document)
            )
            directory = build_ranked_directory(bundle, graph, count_tokens=len)

        self.assertEqual(
            [entry["path"] for entry in directory["entries"]],
            ["src/a.py", "src/c.py", "src/b.py"],
        )

    def test_document_can_match_a_call_name_and_qualified_config_key(self) -> None:
        files = {
            "src/app.py": "def run():\n    return publish()\n",
            "settings.toml": "[runtime]\nmode = 'safe'\n",
        }
        document = (
            "# Delivery\nCall `publish()` after processing.\n\n"
            "# Runtime\nSet `runtime.mode` to safe.\n"
        )
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph = assemble(
                make_repo_bundle(temporary, repository_files=files, document=document)
            )
            directory = build_ranked_directory(bundle, graph, count_tokens=len)

        by_path = {entry["path"]: entry for entry in directory["entries"]}
        self.assertEqual(by_path["src/app.py"]["related_document_sections"], ["Delivery"])
        self.assertEqual(by_path["settings.toml"]["related_document_sections"], ["Runtime"])

    def test_qualified_module_symbol_disambiguates_duplicate_leaf_names(self) -> None:
        files = {
            "pkg/core.py": (
                "def run():\n    return 1\n\n"
                "def fallback(flag):\n"
                "    if flag:\n        return 2\n"
                "    raise ValueError\n"
            ),
            "pkg/other.py": "def run():\n    return 3\n",
        }
        document = "# Public API\nCall `pkg.core.run()` to start.\n"
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files=files,
                    document=document,
                )
            )
            directory = build_ranked_directory(bundle, graph, count_tokens=len)

        core = next(
            item for item in directory["query_index"] if item["path"] == "pkg/core.py"
        )
        self.assertEqual(core["root_symbols"], ["run"])
        self.assertEqual(directory["entries"][0]["path"], "pkg/core.py")

    def test_file_read_keeps_multiple_unambiguous_document_roots(self) -> None:
        source = (
            "def first():\n    return 1\n\n"
            "def second():\n    return 2\n"
        )
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files={"pkg/api.py": source},
                    document=(
                        "# API\nUse both `pkg.api.first()` and "
                        "`pkg.api.second()`.\n"
                    ),
                )
            )
            directory = build_ranked_directory(bundle, graph, count_tokens=len)

        self.assertEqual(
            directory["query_index"][0]["root_symbols"],
            ["first", "second"],
        )

    def test_ambiguous_leaf_call_does_not_select_every_same_file_method(self) -> None:
        source = (
            "class First:\n"
            "    def render(self):\n        return 'first'\n\n"
            "class Second:\n"
            "    def render(self):\n        return 'second'\n"
        )
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files={"pkg/views.py": source},
                    document="# Rendering\nCall `render()` for the selected view.\n",
                )
            )
            directory = build_ranked_directory(bundle, graph, count_tokens=len)

        self.assertEqual(
            directory["query_index"][0]["root_symbols"],
            ["First.render"],
        )

    def test_qualified_class_name_does_not_expand_to_every_member(self) -> None:
        source = (
            "class Retry:\n"
            "    def __init__(self):\n        self.codes = [500]\n\n"
            "    def __call__(self, code):\n        return code in self.codes\n\n"
            "def fallback(flag):\n"
            "    if flag:\n        return 1\n"
            "    raise ValueError\n"
        )
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files={"pkg/retry.py": source},
                    document="# Retry\nClass `pkg.retry.Retry` classifies failures.\n",
                )
            )
            directory = build_ranked_directory(bundle, graph, count_tokens=len)

        self.assertEqual(
            directory["query_index"][0]["root_symbols"],
            ["Retry.__init__"],
        )

    def test_unique_backticked_code_term_selects_its_owner(self) -> None:
        source = (
            "def apply(value):\n"
            "    state_key = str(value)\n"
            "    return state_key\n\n"
            "def fallback(flag):\n"
            "    if flag:\n        return 1\n"
            "    raise ValueError\n"
        )
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files={"pkg/state.py": source},
                    document="# State\nThe `state_key` value is returned.\n",
                )
            )
            directory = build_ranked_directory(bundle, graph, count_tokens=len)

        self.assertEqual(directory["query_index"][0]["root_symbols"], ["apply"])

    def test_ambiguous_symbol_or_call_name_does_not_guess_a_file(self) -> None:
        files = {
            "src/first.py": "def run():\n    return 1\n",
            "src/second.py": "def run():\n    return 2\n",
        }
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files=files,
                    document="# Start\nCall `run()` now.\n",
                )
            )
            directory = build_ranked_directory(bundle, graph, count_tokens=len)

        self.assertTrue(
            all(not entry["related_document_sections"] for entry in directory["entries"])
        )

    def test_repo_budget_limits_display_but_query_index_keeps_every_file(self) -> None:
        files = {
            f"src/component_{index:03d}_with_a_stable_long_name.py": (
                f"def operation_{index:03d}():\n    return {index}\n"
            )
            for index in range(80)
        }
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph = assemble(
                make_repo_bundle(temporary, repository_files=files)
            )
            directory = build_ranked_directory(bundle, graph, count_tokens=len)

        self.assertLess(len(directory["entries"]), len(directory["query_index"]))
        self.assertEqual(len(directory["query_index"]), 80)
        self.assertEqual(directory["token_budget"], DIRECTORY_TOKEN_BUDGET)
        self.assertLessEqual(directory["token_count"], DIRECTORY_TOKEN_BUDGET)
        self.assertEqual(directory["token_count"], len(render_ranked_directory(directory)))

    def test_feedbacktrace_bypasses_ranked_directory(self) -> None:
        events = [
            {"event_type": "user_prompt", "turn_number": 1, "content": "First"},
            {"event_type": "assistant_response", "turn_number": 3, "content": "Done"},
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph = assemble(make_trace_bundle(temporary, events, cutoff=4))
            with self.assertRaisesRegex(DirectoryError, "bypasses Ranked Directory"):
                build_ranked_directory(bundle, graph)

    def test_validation_rejects_a_changed_entry(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files={"src/app.py": "def run():\n    return 1\n"},
                    document="Use src/app.py.\n",
                )
            )
            directory = build_ranked_directory(bundle, graph, count_tokens=len)
            changed = copy.deepcopy(directory)
            changed["entries"][0]["code_hints"] = "problem found"
            with self.assertRaises(DirectoryError):
                validate_ranked_directory(bundle, graph, changed, count_tokens=len)

    def test_repeated_build_is_deterministic(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle, graph = assemble(
                make_repo_bundle(
                    temporary,
                    repository_files={"src/app.py": "def run():\n    return 1\n"},
                )
            )
            first = build_ranked_directory(bundle, graph, count_tokens=len)
            second = build_ranked_directory(bundle, graph, count_tokens=len)

        self.assertEqual(first, second)

    def test_pipeline_writes_and_validates_structured_and_text_indexes(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle_root = make_repo_bundle(
                temporary,
                repository_files={"src/app.py": "def run():\n    return 1\n"},
                document="Use src/app.py.\n",
            )
            bundle, graph = assemble(bundle_root)
            graph_path = temporary / "graph.json"
            graph_path.write_text(
                json.dumps(graph, ensure_ascii=False), encoding="utf-8", newline=""
            )
            output = temporary / "directory"

            first = build_directory_artifact(bundle.root, graph_path, output)
            second = build_directory_artifact(bundle.root, graph_path, output)
            validated = validate_directory_artifact(bundle.root, graph_path, output)

        self.assertEqual(
            set(first["changed"]), {"ranked_directory.json", "ranked_directory.txt"}
        )
        self.assertEqual(second["changed"], [])
        self.assertTrue(validated["valid"])


if __name__ == "__main__":
    unittest.main()
