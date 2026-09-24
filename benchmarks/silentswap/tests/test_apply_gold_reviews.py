import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "apply_gold_reviews", SCRIPTS / "apply_gold_reviews.py"
)
APPLY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(APPLY)


def completed(sample_number, role, decision="accept", semantics="same"):
    return {
        "sample_number": sample_number,
        "swap_number": 1,
        "reviewer_role": role,
        "reviewer_id": f"human_{role}",
        "review_status": "completed",
        "decision": decision,
        "proposed_swap_type": "representation_normalization",
        "original_semantics": f"old {semantics}",
        "swapped_semantics": f"new {semantics}",
        "why_different": f"impact {semantics}",
        "evidence": [f"proof {semantics}"],
    }


class ApplyGoldReviewsTests(unittest.TestCase):
    def test_matching_double_reviews_are_agreement(self):
        records = {
            (6, 1, "primary"): completed(6, "primary", decision="revise"),
            (6, 1, "secondary"): completed(6, "secondary", decision="revise"),
        }

        final, source, status = APPLY.resolve_swap(6, 1, records)

        self.assertEqual("primary", final["reviewer_role"])
        self.assertEqual("model_draft_human_revised", source)
        self.assertEqual("double_reviewed", status)

    def test_disagreement_requires_adjudicator(self):
        records = {
            (6, 1, "primary"): completed(6, "primary", semantics="one"),
            (6, 1, "secondary"): completed(6, "secondary", semantics="two"),
        }

        with self.assertRaises(ValueError):
            APPLY.resolve_swap(6, 1, records)

        records[(6, 1, "adjudicator")] = completed(
            6, "adjudicator", decision="rewrite", semantics="final"
        )
        final, source, status = APPLY.resolve_swap(6, 1, records)

        self.assertEqual("final", final["original_semantics"].split()[-1])
        self.assertEqual("human_rewritten", source)
        self.assertEqual("adjudicated_after_disagreement", status)

    def test_non_sampled_review_is_single_reviewed(self):
        records = {(1, 1, "primary"): completed(1, "primary")}

        _, source, status = APPLY.resolve_swap(1, 1, records)

        self.assertEqual("model_draft_human_verified", source)
        self.assertEqual("single_reviewed", status)

    def test_reviewer_uses_requested_human_verified_source(self):
        record = completed(1, "primary")
        record["reviewer_id"] = "human_reviewer_a"

        _, source, status = APPLY.resolve_swap(
            1, 1, {(1, 1, "primary"): record}
        )

        self.assertEqual("model_draft_human_verified", source)
        self.assertEqual("single_reviewed", status)

    def test_accept_cannot_change_gold_content(self):
        record = completed(1, "primary")
        gold = {
            "swap_type": record["proposed_swap_type"],
            "original_semantics": record["original_semantics"],
            "swapped_semantics": record["swapped_semantics"],
            "why_different": record["why_different"],
            "evidence": ["different proof"],
        }

        with self.assertRaisesRegex(ValueError, "accept review cannot change"):
            APPLY.validate_accept_content(1, 1, record, gold)


if __name__ == "__main__":
    unittest.main()
