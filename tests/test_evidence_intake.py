from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from beg.core.errors import BundleError
from beg.evidence_intake import (
    build_evidence,
    load_visible_bundle,
    validate_evidence,
)
from tests.support import ProjectTemporaryDirectory, make_repo_bundle, make_trace_bundle


SCHEMA = json.loads(
    (PROJECT_ROOT / "schemas" / "evidence.schema.json").read_text(encoding="utf-8")
)
VALIDATE_EVIDENCE = Draft202012Validator(SCHEMA).validate


class EvidenceIntakeTests(unittest.TestCase):
    def test_only_production_artifacts_enter_evidence(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle_root = make_repo_bundle(
                temporary,
                repository_files={
                    "src/app.py": "def run(flag):\n    if flag:\n        return 1\n    raise ValueError('bad')\n",
                    "src/templates/page.jinja2": "{% block body %}Hello {{ name }}{% endblock %}\n",
                    "src/config/settings.toml": "[runtime]\nmode = 'safe'\n",
                    "bin/run": "#!/usr/bin/env python3\nprint('run')\n",
                    "tests/test_app.py": "def test_run():\n    assert True\n",
                    "README.md": "# Repository documentation\n",
                    "notes.txt": "ordinary document\n",
                    "gold/gold.py": "EXPECTED = True\n",
                    "patch.py": "replacement = True\n",
                    ".github/workflows/ci.yml": "steps: []\n",
                    "docs_src/conf.py": "project = 'docs'\n",
                    "pyproject.toml": "[project]\nname='runtime-package'\n",
                    "requirements/base.txt": "requests>=2\n",
                    "requirements-test.txt": "pytest>=8\n",
                },
            )
            bundle = load_visible_bundle(bundle_root)
            self.assertEqual(
                {artifact.path for artifact in bundle.repo_artifacts},
                {
                    "src/app.py",
                    "src/templates/page.jinja2",
                    "src/config/settings.toml",
                    "bin/run",
                    "pyproject.toml",
                    "requirements/base.txt",
                    "patch.py",
                },
            )
            evidence = build_evidence(bundle)
            paths = {item["locator"]["path"] for item in evidence}
            self.assertNotIn("tests/test_app.py", paths)
            self.assertNotIn("README.md", paths)
            self.assertNotIn("notes.txt", paths)
            self.assertNotIn("gold/gold.py", paths)
            self.assertIn("patch.py", paths)
            self.assertNotIn(".github/workflows/ci.yml", paths)
            self.assertNotIn("docs_src/conf.py", paths)
            self.assertNotIn("requirements-test.txt", paths)
            self.assertNotIn(bundle.task_document.path, paths)

    def test_python_evidence_preserves_complete_statements_and_real_symbols(self) -> None:
        source = (
            "from pkg import (\n"
            "    first,\n"
            "    second,\n"
            ")\n"
            "\n"
            "class Client:\n"
            "    def run(self, flag):\n"
            "        if (\n"
            "            flag\n"
            "            and self.ready\n"
            "        ):\n"
            "            value = build(\n"
            "                first, second\n"
            "            )\n"
            "            return value\n"
            "        raise ValueError('bad')\n"
        )
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(
                make_repo_bundle(temporary, repository_files={"src/app.py": source})
            )
            evidence = build_evidence(bundle)
            contents = [item["content"] for item in evidence]
            self.assertIn("from pkg import (\n    first,\n    second,\n)\n", contents)
            self.assertIn(
                "        if (\n            flag\n            and self.ready\n        ):\n",
                contents,
            )
            self.assertIn(
                "            value = build(\n                first, second\n            )\n",
                contents,
            )
            self.assertEqual(contents.count("            return value\n"), 1)
            run_items = [
                item for item in evidence if item["locator"]["symbol"] == "Client.run"
            ]
            self.assertTrue(run_items)
            self.assertNotIn("Client.run.block1", {item["locator"]["symbol"] for item in evidence})
            for item in evidence:
                VALIDATE_EVIDENCE(item)

    def test_configuration_items_are_independent_and_qualified(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(
                make_repo_bundle(
                    temporary,
                    repository_files={
                        "settings.toml": "[runtime]\nmode = 'safe'\nworkers = 2\n",
                        "requirements/base.txt": "requests>=2 \\\n    --hash=sha256:first\nhttpx==1.0\n",
                    },
                )
            )
            evidence = build_evidence(bundle)
            by_symbol = {item["locator"]["symbol"]: item["content"] for item in evidence}
            self.assertEqual(by_symbol["runtime.mode"], "mode = 'safe'\n")
            self.assertEqual(by_symbol["runtime.workers"], "workers = 2\n")
            self.assertEqual(
                by_symbol["requests"],
                "requests>=2 \\\n    --hash=sha256:first\n",
            )
            self.assertEqual(by_symbol["httpx"], "httpx==1.0\n")

    def test_non_python_source_is_one_whole_file_evidence(self) -> None:
        source = "".join(f"echo line-{number}\n" for number in range(1, 101))
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(
                make_repo_bundle(temporary, repository_files={"bin/run.sh": source})
            )
            evidence = build_evidence(bundle)
            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0]["locator"]["symbol"], "<module>")
            self.assertEqual(evidence[0]["locator"]["line_start"], 1)
            self.assertEqual(evidence[0]["locator"]["line_end"], 100)
            self.assertEqual(evidence[0]["content"], source)

    def test_trace_keeps_every_event_verbatim_and_exposes_full_locator(self) -> None:
        events = [
            {"event_type": "system", "turn_number": 0, "content": "context"},
            {"event_type": "user_prompt", "turn_number": 1, "content": "Do it."},
            {
                "event_type": "tool_exchange",
                "turn_number": 2,
                "content": "",
                "evidence_id": "raw-E2",
                "tool_name": "write_file",
            },
            {
                "event_type": "assistant_response",
                "turn_number": 3,
                "content": "Done.",
                "evidence_id": "raw-E3",
            },
        ]
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(make_trace_bundle(temporary, events, cutoff=4))
            evidence = build_evidence(bundle)
            self.assertEqual([item["content"] for item in evidence], [item["content"] for item in events])
            self.assertEqual(
                evidence[2]["locator"],
                {
                    "turn": 2,
                    "event_index": 2,
                    "event_type": "tool_exchange",
                    "tool_name": "write_file",
                    "original_evidence_id": "raw-E2",
                },
            )
            for item in evidence:
                VALIDATE_EVIDENCE(item)

    def test_same_input_rebuilds_identical_ids_content_and_order(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            bundle = load_visible_bundle(
                make_repo_bundle(
                    temporary,
                    repository_files={"b.py": "result = 2\n", "a.py": "result = 1\n"},
                )
            )
            first = build_evidence(bundle)
            second = build_evidence(bundle)
            self.assertEqual(first, second)
            self.assertEqual(
                [item["evidence_id"] for item in first],
                [f"E{index:06d}" for index in range(1, len(first) + 1)],
            )
            validate_evidence(bundle, first)

    def test_cutoff_or_non_visible_material_is_rejected_at_boundary(self) -> None:
        with ProjectTemporaryDirectory() as temporary:
            trace_root = make_trace_bundle(
                temporary,
                [{"event_type": "user_prompt", "turn_number": 5, "content": "late"}],
                cutoff=5,
            )
            with self.assertRaisesRegex(BundleError, "at or after cutoff"):
                load_visible_bundle(trace_root)


if __name__ == "__main__":
    unittest.main()
