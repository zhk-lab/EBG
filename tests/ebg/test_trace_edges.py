"""Task-edge regressions independent of task segmentation heuristics."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ebg.relation_linking import _build_trace_edges, _EvidenceIndex


def edges_for(events):
    evidence, behaviors = [], []
    for number, (task, role, content) in enumerate(events, 1):
        eid = f"E{number:06d}"
        evidence.append({"evidence_id": eid, "source_type": "trace", "content": content,
                         "locator": {"turn": number, "event_index": number,
                                     "event_type": role}})
        behaviors.append({"task_id": task, "action_evidence_ids": [],
                          "demand_refs": [], "response_refs": []})
        if role == "tool_exchange":
            behaviors[-1]["action_evidence_ids"] = [eid]
        else:
            field = "response_refs" if role == "assistant_response" else "demand_refs"
            behaviors[-1][field] = [{"evidence_id": eid}]
    return _build_trace_edges(_EvidenceIndex(evidence), behaviors)


class TraceEdgeTests(unittest.TestCase):
    def test_reference_direction_and_interleaved_task_resume(self):
        edges = edges_for([
            ("A", "assistant_response", "Started job ID eval-42"),
            ("B", "user_prompt", "Inspect job ID eval-42"),
            ("A", "user_prompt", "Continue"),
        ])
        self.assertEqual([(e["source_task_id"], e["type"], e["target_task_id"])
                          for e in edges], [("A", "precedes", "B"), ("B", "references", "A")])

    def test_future_producer_is_not_referenced(self):
        edges = edges_for([("A", "user_prompt", "Inspect job ID eval-42"),
                           ("B", "assistant_response", "Started job ID eval-42")])
        self.assertEqual([e["type"] for e in edges], ["precedes"])

    def test_duplicate_output_and_ambiguous_reference_do_not_create_edges(self):
        edges = edges_for([("A", "assistant_response", "Started job ID eval-42"),
                           ("B", "assistant_response", "Started job ID eval-42"),
                           ("C", "user_prompt", "Inspect job ID eval-42")])
        self.assertEqual([e["type"] for e in edges], ["precedes", "precedes"])

    def test_same_value_with_different_id_type_is_not_a_reference(self):
        edges = edges_for([("A", "assistant_response", "Started job ID eval-42"),
                           ("B", "user_prompt", "Inspect run ID eval-42")])
        self.assertEqual([e["type"] for e in edges], ["precedes"])

    def test_user_supplied_id_has_no_task_output_source(self):
        edges = edges_for([("A", "user_prompt", "Inspect job ID eval-42"),
                           ("B", "user_prompt", "Inspect job ID eval-42")])
        self.assertEqual([e["type"] for e in edges], ["precedes"])

    def test_specific_artifact_url_is_referenced(self):
        url = "https://github.com/org/repo/issues/12#issuecomment-98765"
        edges = edges_for([("A", "assistant_response", url),
                           ("B", "user_prompt", "Look at " + url)])
        self.assertEqual([e["type"] for e in edges], ["precedes", "references"])

    def test_generic_url_is_not_a_reference(self):
        url = "https://github.com/org/repo"
        edges = edges_for([("A", "assistant_response", url),
                           ("B", "user_prompt", "Look at " + url)])
        self.assertEqual([e["type"] for e in edges], ["precedes"])

    def test_command_and_short_commit_number_are_not_ids(self):
        from ebg.relation_linking import _trace_keys
        self.assertEqual(_trace_keys("run python3; commit 243; shard 0; task 1; run:init"), set())
        self.assertEqual(_trace_keys('"job_id": "eval-42"'), {"job:eval-42"})

    def test_numbered_reply_references_the_actual_item(self):
        edges = edges_for([("A", "assistant_response", "1. Fix parsing\n2. Fix caching"),
                           ("B", "user_prompt", "Explain item 2 to me a little more")])
        self.assertEqual([e["type"] for e in edges], ["precedes", "references"])

    def test_topic_backreference_is_not_structural(self):
        edges = edges_for([("A", "assistant_response", "Found a bug: cache misses are never retried."),
                           ("B", "user_prompt", "Fix the bugs you mentioned earlier")])
        self.assertEqual([e["type"] for e in edges], ["precedes"])

    def test_numbered_reply_cannot_reach_past_an_unrelated_response(self):
        edges = edges_for([("A", "assistant_response", "1. Fix parsing\n2. Fix caching"),
                           ("B", "user_prompt", "Check deployment"),
                           ("B", "assistant_response", "Deployment succeeded"),
                           ("C", "user_prompt", "Explain 2")])
        self.assertTrue(all(e["type"] == "precedes" for e in edges))

    def test_shared_quoted_path_is_not_a_quoted_claim(self):
        path = "extension/tsconfig*.json"
        edges = edges_for([("A", "assistant_response", path),
                           ("B", "user_prompt", 'Check "' + path + '"')])
        self.assertEqual([e["type"] for e in edges], ["precedes"])

    def test_approval_is_not_a_reference(self):
        edges = edges_for([("A", "assistant_response", "Review found a bug"),
                           ("B", "user_prompt", "Good, continue")])
        self.assertEqual([e["type"] for e in edges], ["precedes"])

    def test_named_artifact_phrase_alone_is_not_structural(self):
        edges = edges_for([("A", "assistant_response", "The smoke-test is ready to run."),
                           ("B", "user_prompt", "run the smoke test we prepared")])
        self.assertEqual([e["type"] for e in edges], ["precedes"])

    def test_quoted_claim_can_refer_across_intervening_tasks(self):
        claim = "The confidence intervals overlap (12.7% vs 7.9%), so the results are statistically consistent."
        edges = edges_for([("A", "assistant_response", claim),
                           ("B", "user_prompt", "Update the table"),
                           ("B", "assistant_response", "Done"),
                           ("C", "user_prompt", '"' + claim + '" Where did you get these numbers?')])
        refs = [e for e in edges if e["type"] == "references"]
        self.assertEqual([(e["source_task_id"], e["target_task_id"]) for e in refs], [("C", "A")])

    def test_risk_phrase_has_no_special_rule(self):
        edges = edges_for([("A", "assistant_response", "Component extraction is the highest-risk remaining work."),
                           ("A", "assistant_response", "Review table: all earlier phases are done."),
                           ("B", "user_prompt", "As you pointed out the next part is riskier; investigate it.")])
        refs = [e for e in edges if e["type"] == "references"]
        self.assertEqual(refs, [])

    def test_repeated_id_in_tool_result_is_not_a_reference(self):
        edges = edges_for([("A", "assistant_response", "Started job ID eval-42"),
                           ("B", "tool_exchange", "Tool invocation: list jobs\nTool result: job ID eval-42")])
        self.assertEqual([e["type"] for e in edges], ["precedes"])

    def test_backward_pointer_with_multiple_sources_is_ambiguous(self):
        edges = edges_for([("A", "assistant_response", "Found a cache bug."),
                           ("B", "assistant_response", "Found a parsing bug."),
                           ("C", "user_prompt", "Fix the bugs you mentioned earlier")])
        self.assertTrue(all(e["type"] == "precedes" for e in edges))

    def test_chinese_numbered_reference(self):
        edges = edges_for([("A", "assistant_response", "1. 修复解析\n2. 修复缓存"),
                           ("B", "user_prompt", "解释第2项")])
        self.assertEqual([e["type"] for e in edges], ["precedes", "references"])

    def test_duplicate_item_labels_are_ambiguous(self):
        edges = edges_for([("A", "assistant_response", "1. Alpha\n2. Beta\n\n1. Gamma\n2. Delta"),
                           ("B", "user_prompt", "Explain item 2")])
        self.assertEqual([e["type"] for e in edges], ["precedes"])

    def test_number_without_item_marker_is_not_guessed(self):
        edges = edges_for([("A", "assistant_response", "1. Alpha\n2. Beta"),
                           ("B", "user_prompt", "We need 2 tests")])
        self.assertEqual([e["type"] for e in edges], ["precedes"])

    def test_partial_quote_is_not_fuzzy_matched(self):
        edges = edges_for([("A", "assistant_response", "The first eight words of this quoted sentence do not imply success."),
                           ("B", "user_prompt", '"The first eight words of this quoted sentence imply success."')])
        self.assertEqual([e["type"] for e in edges], ["precedes"])
