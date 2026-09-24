import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_formal_samples.py"
SPEC = importlib.util.spec_from_file_location("build_formal_samples", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FormalPipelineTests(unittest.TestCase):
    def test_generated_gold_uses_fixed_taxonomy_and_pending_review(self):
        gold = {
            "swap_type": "ordering_precedence",
            "original_semantics": "old",
            "swapped_semantics": "new",
            "why_different": "impact",
            "evidence": ["proof"],
        }

        prepared = MODULE.prepare_generated_gold(gold)

        self.assertEqual(8, len(MODULE.SWAP_TYPES))
        self.assertEqual("model_generated", prepared["gold_source"])
        self.assertEqual("pending", prepared["adjudication_status"])

    def test_generated_gold_rejects_free_form_swap_type(self):
        gold = {
            "swap_type": "one-off label",
            "original_semantics": "old",
            "swapped_semantics": "new",
            "why_different": "impact",
            "evidence": ["proof"],
        }

        with self.assertRaises(MODULE.PipelineError):
            MODULE.prepare_generated_gold(gold)

    def test_verification_commands_are_portable(self):
        evidence = {
            section: {
                "command": f"docker --volume {MODULE.ROOT}/data/1:/workspace/project"
            }
            for section in MODULE.VERIFICATION_COMMAND_SECTIONS
        }

        MODULE.sanitize_verification_commands(evidence)

        for section in MODULE.VERIFICATION_COMMAND_SECTIONS:
            command = evidence[section]["command"]
            self.assertIn(MODULE.PORTABLE_DATASET_ROOT, command)
            self.assertNotIn(str(MODULE.ROOT), command)

    def test_replace_once_requires_one_exact_match(self):
        self.assertEqual(
            "process the first 100 records",
            MODULE.replace_once("process all records", "all", "the first 100", "original text"),
        )
        with self.assertRaises(MODULE.PipelineError):
            MODULE.replace_once("all or all", "all", "some", "original text")

    def test_document_context_selects_complete_sections(self):
        document = "## First\na\n## Second\nb\n### Child\nc\n## Third\nd\n"

        selected = MODULE.document_context(document, ["## Second"])

        self.assertEqual(selected, "## Second\nb\n### Child\nc")

    def test_parse_test_counts(self):
        self.assertEqual(
            {"passed": 65, "skipped": 2},
            MODULE.parse_test_counts("65 passed, 2 skipped in 0.4s"),
        )
        self.assertEqual({"ran": 126}, MODULE.parse_test_counts("Ran 126 tests in 0.02s"))

    def test_parse_json_object_accepts_json_fence(self):
        self.assertEqual({"ok": True}, MODULE.parse_json_object("```json\n{\"ok\": true}\n```"))

    def test_parse_json_object_rejects_empty_answer(self):
        with self.assertRaises(MODULE.PipelineError):
            MODULE.parse_json_object("  ")

    def test_write_json_is_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            MODULE.write_json(path, {"text": "语义替换"})
            self.assertEqual({"text": "语义替换"}, json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
