"""Tests for the restored token counting and rank-based tertiles."""
import unittest
from types import SimpleNamespace
from count_workload import assign_bins, measure_bundle


class TokenGroupingTests(unittest.TestCase):
    def test_original_tertiles_with_deterministic_ties(self):
        rows = [{"component": "specgap", "input_id": f"s{i:03}", "search_count": 10}
                for i in reversed(range(100))]
        result = assign_bins(rows)
        self.assertEqual([sum(r["bin"] == b for r in result) for b in ("Low", "Medium", "High")],
                         [33, 33, 34])
        self.assertEqual(result[0]["input_id"], "s000")
        self.assertEqual(result[-1]["rank"], 100)

    def test_repo_counts_document_and_readable_artifacts(self):
        bundle = SimpleNamespace(benchmark="specgap",
            task_document=SimpleNamespace(content="abc"),
            repo_artifacts=[SimpleNamespace(content="defgh")], trace_events=[])
        self.assertEqual(measure_bundle(bundle, len)["search_count"], 8)

    def test_trace_counts_responses_and_tool_exchanges_only(self):
        bundle = SimpleNamespace(benchmark="feedbacktrace", task_document=None, repo_artifacts=[],
            trace_events=[SimpleNamespace(event_type=t, content=c) for t, c in
                          (("system", "abc"), ("user_prompt", "de"),
                           ("assistant_response", "fghi"), ("tool_exchange", "jklmn"))])
        result = measure_bundle(bundle, len)
        self.assertEqual(result["search_count"], 9)
        self.assertEqual(result["count_metric"], "workload_tokens")


if __name__ == "__main__":
    unittest.main()
