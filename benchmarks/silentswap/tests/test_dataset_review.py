import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

CONFIG_SPEC = importlib.util.spec_from_file_location(
    "review_config", SCRIPTS / "review_config.py"
)
CONFIG = importlib.util.module_from_spec(CONFIG_SPEC)
assert CONFIG_SPEC.loader is not None
CONFIG_SPEC.loader.exec_module(CONFIG)

VALIDATE_SPEC = importlib.util.spec_from_file_location(
    "validate_dataset", SCRIPTS / "validate_dataset.py"
)
VALIDATE = importlib.util.module_from_spec(VALIDATE_SPEC)
assert VALIDATE_SPEC.loader is not None
VALIDATE_SPEC.loader.exec_module(VALIDATE)

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


class DatasetReviewTests(unittest.TestCase):
    def test_taxonomy_covers_every_sample_once(self):
        self.assertEqual(set(range(1, 101)), set(CONFIG.SWAP_TYPE_BY_SAMPLE))
        self.assertEqual(100, len(CONFIG.SWAP_TYPE_BY_SAMPLE))
        self.assertEqual(8, len(CONFIG.SAMPLES_BY_SWAP_TYPE))

    def test_double_review_sample_is_stratified(self):
        counts = Counter(
            CONFIG.SWAP_TYPE_BY_SAMPLE[number]
            for number in CONFIG.DOUBLE_REVIEW_SAMPLES
        )

        self.assertEqual(25, len(CONFIG.DOUBLE_REVIEW_SAMPLES))
        self.assertEqual(
            {
                "input_validation_boundary": 3,
                "parsing_matching": 4,
                "ordering_precedence": 5,
                "default_null_fallback": 3,
                "exception_error_handling": 3,
                "state_identity_lifecycle": 2,
                "representation_normalization": 3,
                "dispatch_routing_aggregation": 2,
            },
            dict(counts),
        )

    def test_local_path_pattern_detects_common_host_paths(self):
        self.assertIsNotNone(VALIDATE.LOCAL_PATH.search("/Users/alice/Desktop/project"))
        self.assertIsNotNone(VALIDATE.LOCAL_PATH.search(r"C:\Users\alice\project"))
        self.assertIsNone(
            VALIDATE.LOCAL_PATH.search(
                "/home/alice/project /workspace/project /tmp/test.py"
            )
        )

    def test_failure_fingerprint_normalizes_dynamic_values(self):
        result = {
            "stdout": "",
            "stderr": "AssertionError: expected /tmp/tmpabc/value 12, got 'actual'\n",
        }

        self.assertEqual(
            "AssertionError: expected <temporary-directory>/value <number>, got '<value>'",
            VALIDATE.failure_fingerprint(result),
        )

    def test_failure_fingerprint_rejects_code_loading_errors_in_stderr(self):
        with self.assertRaisesRegex(ValueError, "code-loading error"):
            VALIDATE.failure_fingerprint(
                {"stdout": "", "stderr": "ModuleNotFoundError: missing"}
            )

    def test_failure_fingerprint_allows_intentional_import_text_in_stdout(self):
        self.assertEqual(
            "caught ImportError as documented",
            VALIDATE.failure_fingerprint(
                {"stdout": "caught ImportError as documented", "stderr": ""}
            ),
        )

    def test_failure_fingerprint_ignores_warning_only_stderr(self):
        self.assertEqual(
            "FAIL: semantic mismatch",
            VALIDATE.failure_fingerprint(
                {
                    "stdout": "observed changed behavior\nFAIL: semantic mismatch\n",
                    "stderr": (
                        "/workspace/pkg.py:4: UserWarning: deprecated API\n"
                        "  import old_api\n"
                    ),
                }
            ),
        )

    def test_failure_fingerprint_prefers_traceback_over_stdout(self):
        self.assertEqual(
            "AssertionError: expected old behavior",
            VALIDATE.failure_fingerprint(
                {
                    "stdout": "FAIL: semantic mismatch\n",
                    "stderr": (
                        "/workspace/pkg.py:4: UserWarning: deprecated API\n"
                        "  import old_api\n"
                        "Traceback (most recent call last):\n"
                        "AssertionError: expected old behavior\n"
                    ),
                }
            ),
        )

    def test_added_python_explanations_detects_comments_and_docstrings(self):
        original = "class InvalidInput(TypeError):\n    pass\n"
        swapped = (
            "class InvalidInput(TypeError):\n"
            "    \"\"\"Raised for invalid input.\"\"\"\n"
            "    # Reject unsupported values.\n"
            "    pass\n"
        )

        comments, docstrings = VALIDATE.added_python_explanations(original, swapped)

        self.assertEqual(["# Reject unsupported values."], comments)
        self.assertEqual(["Raised for invalid input."], docstrings)

    def test_added_python_explanations_allows_existing_text(self):
        source = (
            "class InvalidInput(TypeError):\n"
            "    \"\"\"Raised for invalid input.\"\"\"\n"
            "    # Existing explanation.\n"
            "    pass\n"
        )

        self.assertEqual(([], []), VALIDATE.added_python_explanations(source, source))

    @unittest.skipUnless((DATA_ROOT / "100/gold.json").is_file(), "requires constructed SilentSwap data")
    def test_all_multi_swap_evidence_passes_strict_validation(self):
        for sample_number in range(1, 101):
            VALIDATE.validate_commands(sample_number)

    @unittest.skipUnless((DATA_ROOT / "100/gold.json").is_file(), "requires constructed SilentSwap data")
    def test_localization_check_supports_current_dataset(self):
        sample = DATA_ROOT / "64"
        gold_items = json.loads((sample / "gold.json").read_text())["swaps"]
        VALIDATE.validate_localization_targets(sample, gold_items)

    @unittest.skipUnless((DATA_ROOT / "100/gold.json").is_file(), "requires constructed SilentSwap data")
    def test_localization_check_rejects_lines_from_only_one_nested_method(self):
        sample = DATA_ROOT / "64"
        gold_items = json.loads((sample / "gold.json").read_text())["swaps"]
        invalid = deepcopy(gold_items)
        invalid[2]["localization"]["line_ranges"] = [{"start": 107, "end": 108}]

        with self.assertRaisesRegex(ValueError, "localization symbol differs"):
            VALIDATE.validate_localization_targets(sample, invalid)

    @unittest.skipUnless((DATA_ROOT / "100/gold.json").is_file(), "requires constructed SilentSwap data")
    def test_sample_layouts_and_repositories_are_release_clean(self):
        for sample_number in range(1, 101):
            VALIDATE.validate_sample_layout(sample_number)
            VALIDATE.validate_repository(sample_number)


if __name__ == "__main__":
    unittest.main()
