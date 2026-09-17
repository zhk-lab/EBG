from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from agentloop.errors import AgentLoopError, ContextUnfitError
from agentloop.silentswap_source_review import CONFIG, build_review, combine_judgments, load_stage1, run_review, parse_review_prediction
from agentloop.storage import RunStore
from tests.agentloop.test_runner import ScriptedClient, AlwaysRetryClient
from agentloop.benchmark_configs import repo_benchmark_config
from evaluation_core.contracts import model_prediction_schema, PredictionError
from tests.support import ProjectTemporaryDirectory


def fixture():
    bundle = SimpleNamespace(
        input_id="ss_test", benchmark="silentswap",
        task_document=SimpleNamespace(content="Original contract: zero is preserved."),
        repo_artifacts=[
            SimpleNamespace(path="z.py", content="def f(x):\n    return x or 1\n# FILE_END\n"),
            SimpleNamespace(path="a.py", content="# not read\nvalue = 1\nother = 2\n# unseen\n"),
            *[SimpleNamespace(path=f"{name}.py", content=f"# {name} complete source\n")
              for name in ("b", "c", "d", "e", "read_only")],
        ],
    )
    prediction = {"input_id": "ss_test", "benchmark": "silentswap", "swaps": [{
        "target": {"file": "z.py", "symbol": {"kind": "function", "qualified_name": ["POISON_SYMBOL"]},
                   "line_ranges": [{"start": 2, "end": 2}]},
        "code_change": "POISON_OLD_EXPLANATION", "trigger_condition": "POISON_TRIGGER",
    }]}
    state = {"input_id": "ss_test", "records": [{
        "raw_response": "POISON_HISTORY", "tool_result": {"action": "read", "units": [{
            "unit_kind": "local_graph", "span": {"path": "z.py"}, "source": "POISON_GRAPH_LABEL",
            "source_regions": [
                {"path": "a.py", "start": 2, "end": 2, "source": "POISON_RENDERED_TEXT"},
                {"path": "a.py", "start": 2, "end": 3, "source": "old rendered source"},
                {"path": "read_only.py", "start": 1, "end": 1, "source": "read-only evidence"},
            ],
        }]},
    }]}
    return bundle, prediction, state


def directory():
    return {"input_id": "ss_test", "benchmark": "silentswap", "entries": [
        {"path": f"{name}.py", "code_hints": "POISON_DIRECTORY_HINT"}
        for name in ("b", "a", "c", "d", "e", "read_only", "z")
    ]}


def answer():
    return {"swaps": [{
        "target": {"file": "a.py", "symbol": {"kind": "field", "qualified_name": ["value"]},
                   "line_ranges": [{"start": 2, "end": 2}]},
        "swap_type": "default_null_fallback", "code_change": f"before {i} -> after {i}",
        "trigger_condition": "zero", "behavioral_effect": {"before": "zero", "after": "one"},
    } for i in range(5)]}


class SourceReviewTests(unittest.TestCase):
    def test_fenced_prediction_is_parsed_without_changing_content(self):
        value = answer()
        wrapped = "Source review follows.\n\n```json\n" + json.dumps(value) + "\n```\n"
        self.assertEqual(parse_review_prediction(wrapped), value)
        for invalid in (wrapped + wrapped, "```json\n{broken}\n```", "[]",
                        "```python\n{}\n```", "prose without a JSON block"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_review_prediction(invalid)
        bundle, prediction, state = fixture()
        review = build_review(bundle, prediction, directory(), state=state)
        with ProjectTemporaryDirectory() as root:
            client = ScriptedClient([wrapped])
            result = run_review(review, client, RunStore(root))
            self.assertEqual(result["swaps"], value["swaps"])
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(run_review(review, client, RunStore(root)), result)
            self.assertEqual(len(client.calls), 1)

    def test_source_review_validates_the_visible_prediction_schema(self):
        bundle, prediction, state = fixture()
        review = build_review(bundle, prediction, directory(), state=state)
        config = repo_benchmark_config("silentswap")
        bound = config.bind_module7(ROOT / "schemas", input_id=bundle.input_id,
            task_document_name=config.task_document_filename,
            task_document=bundle.task_document.content, initial_index="EBG directory",
            prediction_schema=review.schema,
            source_texts={item.path: item.content for item in bundle.repo_artifacts})
        shown = json.loads(review.messages[1]["content"].split("PREDICTION SCHEMA\n", 1)[1])
        self.assertEqual(shown, model_prediction_schema(bound.finish_contract.prediction_schema))
        self.assertEqual(shown, model_prediction_schema(review.schema))
        invalids = [answer() for _ in range(4)]
        invalids[0]["swaps"].pop()
        invalids[1]["swaps"][0]["target"]["line_ranges"] = [[2, 2]]
        invalids[2]["swaps"][0]["target"]["symbol"]["kind"] = "invalid"
        invalids[3]["swaps"][0]["target"]["line_ranges"] = [{"start": 100, "end": 100}]
        for value in invalids:
            with self.subTest(value=value), self.assertRaises(PredictionError):
                bound.finish_contract.validate(value, input_id=bundle.input_id, observed_spans=review.spans)
        bound.finish_contract.validate(answer(), input_id=bundle.input_id, observed_spans=review.spans)
        with ProjectTemporaryDirectory() as root:
            client = ScriptedClient([json.dumps(invalids[0]), json.dumps(answer())])
            result = run_review(review, client, RunStore(root))
            self.assertEqual(len(result["swaps"]), 5)
            self.assertEqual(len(client.calls), 2)

    def test_network_exhaustion_can_resume_without_overwriting_failures(self):
        bundle, prediction, state = fixture()
        review = build_review(bundle, prediction, directory(), state=state)
        with ProjectTemporaryDirectory() as root:
            failed = AlwaysRetryClient()
            with self.assertRaises(AgentLoopError):
                run_review(review, failed, RunStore(root))
            self.assertEqual(len(failed.calls), CONFIG.network_retries + 1)
            failures = {p.name: p.read_bytes() for p in (root / "provider_failures").glob("*.json")}
            client = ScriptedClient([json.dumps(answer())])
            run_review(review, client, RunStore(root))
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(failures, {p.name: p.read_bytes() for p in (root / "provider_failures").glob("*.json")})

    def test_oversized_request_stops_before_model_call(self):
        bundle, prediction, state = fixture()
        review = build_review(bundle, prediction, directory(), state=state)
        with ProjectTemporaryDirectory() as root:
            client = ScriptedClient([])
            with patch("agentloop.silentswap_source_review.FastTokenCounter.count_messages",
                       return_value=CONFIG.physical_hard_limit + 1), self.assertRaises(ContextUnfitError):
                run_review(review, client, RunStore(root))
            self.assertEqual(client.calls, [])

    def test_stage1_history_is_selected_by_final_prediction(self):
        with ProjectTemporaryDirectory() as root:
            _, prediction, state = fixture()
            store = RunStore(root)
            store._write_json(root / "prediction.json", prediction)
            for number in (1, 2):
                attempt = root / f"attempt_{number}"
                saved = prediction if number == 2 else {**prediction, "swaps": []}
                store._write_json(attempt / "prediction.json", saved)
                store._write_json(attempt / "state.json", {**state, "status": "complete"})
            selected, history, path = load_stage1(root)
            self.assertEqual(selected, prediction)
            self.assertEqual(history["records"], state["records"])
            self.assertEqual(path, root / "attempt_2/state.json")

    def test_union_has_full_top6_and_read_files(self):
        bundle, prediction, state = fixture()
        prediction["swaps"] *= 2
        review = build_review(bundle, prediction, directory(), state=state)
        text = "\n".join(m["content"] for m in review.messages)
        self.assertNotIn("POISON", text)
        self.assertIn("3 | # FILE_END", text)
        self.assertIn("2 | value = 1\n3 | other = 2", text)
        self.assertIn("1 | # not read", text)
        self.assertIn("4 | # unseen", text)
        self.assertIn("FILE read_only.py", text)
        self.assertEqual(review.selection["full_files"], ["a.py", "b.py", "c.py", "d.py", "e.py", "read_only.py", "z.py"])
        self.assertEqual(text.count("FILE z.py"), 1)
        self.assertEqual(text.count("2 | value = 1"), 1)
        self.assertLess(text.index("FILE a.py"), text.index("FILE z.py"))
        changed = copy.deepcopy(prediction)
        changed["swaps"][0]["target"]["file"] = "unread_answer.py"
        changed["swaps"][0]["target"]["line_ranges"] = [{"start": 1, "end": 3}]
        self.assertEqual(review.messages, build_review(bundle, changed, directory(), state=state).messages)

    def test_all_reads_are_included_without_search_or_context_only_paths(self):
        bundle, prediction, state = fixture()
        bundle.repo_artifacts.append(SimpleNamespace(path="later.py", content="# full later file\n"))
        state["records"].append({"tool_result": {"action": "read", "units": [
            {"span": {"path": "later.py"}, "source_regions": [{"path": "context_only.py"}]},
            {"span": {"path": "z.py"}},
        ]}})
        state["records"].append({"tool_result": {"action": "search", "units": [
            {"span": {"path": "search_only.py"}},
        ]}})
        review = build_review(bundle, prediction, directory(), state=state)
        self.assertEqual(review.selection["read_files"], ["later.py", "z.py"])
        self.assertEqual(len(review.selection["directory_top_files"]), 6)
        self.assertEqual(len(review.selection["full_files"]), 8)
        self.assertIn("1 | # full later file", review.messages[1]["content"])
        state["input_id"] = "wrong_sample"
        with self.assertRaises(AgentLoopError):
            build_review(bundle, prediction, directory(), state=state)

    def test_overlap_does_not_duplicate_files_and_missing_source_is_rejected(self):
        bundle, prediction, state = fixture()
        state["records"][0]["tool_result"]["units"][0]["span"]["path"] = "a.py"
        review = build_review(bundle, prediction, directory(), state=state)
        self.assertEqual(len(review.selection["full_files"]), 6)
        self.assertEqual(review.messages[1]["content"].count("FILE a.py"), 1)
        ranked = directory()
        ranked["entries"][0]["path"] = "missing.py"
        with self.assertRaises(AgentLoopError):
            build_review(bundle, prediction, ranked, state=state)

    def test_stage2_targets_are_independent_and_resume_does_not_call_model(self):
        bundle, prediction, state = fixture()
        review = build_review(bundle, prediction, directory(), state=state)
        with ProjectTemporaryDirectory() as root:
            client = ScriptedClient([json.dumps(answer())])
            prediction = run_review(review, client, RunStore(root))
            self.assertEqual(prediction["swaps"][0]["target"]["file"], "a.py")
            self.assertEqual(run_review(review, client, RunStore(root)), prediction)
            self.assertEqual(len(client.calls), 1)

    def test_ungrounded_answer_gets_one_correction_and_resume_checks_inputs(self):
        bundle, prediction, state = fixture()
        review = build_review(bundle, prediction, directory(), state=state)
        invalid = answer()
        invalid["swaps"][0]["target"]["line_ranges"] = [{"start": 5, "end": 5}]
        with ProjectTemporaryDirectory() as root:
            client = ScriptedClient([json.dumps(invalid), json.dumps(answer())])
            run_review(review, client, RunStore(root))
            self.assertEqual(len(client.calls), 2)
            bundle.task_document.content += " Changed contract."
            with self.assertRaises(AgentLoopError):
                run_review(build_review(bundle, prediction, directory(), state=state), client, RunStore(root))

    def test_combined_scores_keep_stage_provenance_without_swap_alignment(self):
        first = {"input_id": "ss_test", "benchmark": "silentswap",
                 "rule_based": {"localization_score": 0.8},
                 "llm_judge": {"scores": {"location_correct": 0.5, "code_change_correct": 0}}}
        second = {"input_id": "ss_test", "benchmark": "silentswap",
                  "rule_based": {"localization_score": 0.1},
                  "llm_judge": {"scores": {"location_correct": 0, "code_change_correct": 1}}}
        result = combine_judgments(first, second)
        self.assertEqual(result["scores"], {
            "localization_score": 0.8, "location_correct": 0.5, "code_change_correct": 1,
        })
        self.assertEqual(result["metric_sources"]["code_change_correct"], "stage2")
        second["input_id"] = "ss_other"
        with self.assertRaises(ValueError):
            combine_judgments(first, second)


if __name__ == "__main__":
    unittest.main()
