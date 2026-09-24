"""Targeted checks of model-visible ablations, resumption and paired statistics."""

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run
import analyze
from ablate_behavior import EvidenceBackend, render_trace as evidence_trace
from ablate_graph import BehaviorListBackend, render_trace as behavior_trace
from ablate_task import FixedEntryBackend
from agentloop.evidence import render_tool_result
from agentloop.graph_backend import GraphBackend
from tests.ebg.test_local_graph_retrieval import assemble, repository
from tests.support import ProjectTemporaryDirectory, make_repo_bundle
from tests.tracereview.test_runner import trace_graph
from tracereview import render_trace_view


class RepoAblationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = ProjectTemporaryDirectory()
        self.root = self.temporary.__enter__()
        self.addCleanup(self.temporary.__exit__, None, None, None)
        self.bundle, self.graph, self.directory = assemble(make_repo_bundle(
            self.root, repository_files=repository(),
            document="# API\n`pkg/api.py` exposes `run` and `secondary`.\n"))
        self.original = copy.deepcopy(self.graph)
        self.kwargs = dict(token_budget=200_000, max_atomic_unit_tokens=300_000, count_tokens=len)

    def read_run(self, backend):
        read_id = backend.search("run", limit=12).hits[0].unit_id
        return backend.read([read_id], **self.kwargs)

    def test_evidence_json_preserves_existing_atoms_scopes_and_edges(self):
        full = GraphBackend(self.bundle, self.graph, self.directory, count_tokens=len)
        ablated = EvidenceBackend(self.bundle, self.graph, self.directory, count_tokens=len)
        before, after = self.read_run(full), self.read_run(ablated)
        self.assertNotEqual(render_tool_result(before), render_tool_result(after))
        payload = json.loads(after.units[0].source)
        root = payload["graphs"][0]["root"]
        self.assertEqual(root["scope"], "run")
        self.assertNotIn("behaviors", root)
        self.assertGreater(len(root["evidence"]), 1)
        originals = {e["evidence_id"]: e for e in self.graph["evidence"]}
        for item in root["evidence"]:
            self.assertEqual(set(item), {"id", "lines", "content"})
            self.assertEqual(item["content"], originals[item["id"]]["content"])
        rid = after.units[0].unit_id
        local = full.local_graph(rid, token_budget=300_000)
        self.assertEqual([s["edge"] for g in local["graphs"] for p in g["paths"] for s in p["steps"]],
                         [s["edge"] for g in payload["graphs"] for p in g["paths"] for s in p["steps"]])
        self.assertEqual(self.graph, self.original)

    def test_evidence_grounding_contains_only_displayed_atoms(self):
        backend = EvidenceBackend(self.bundle, self.graph, self.directory, count_tokens=len)
        unit = self.read_run(backend).units[0]
        payload = json.loads(unit.source)
        atoms = [e for g in payload["graphs"] for n in
                 [g["root"], *(s["node"] for p in g["paths"] for s in p["steps"])]
                 for e in n.get("evidence", [])]
        self.assertEqual(len(unit.source_regions), len(atoms))
        for atom, region in zip(atoms, unit.source_regions):
            self.assertEqual([region.start, region.end], atom["lines"])
            self.assertTrue(region.source)

    def test_no_graph_has_roles_but_no_scope_or_neighbors(self):
        backend = BehaviorListBackend(self.bundle, self.graph, self.directory, count_tokens=len)
        result = self.read_run(backend)
        payload = json.loads(result.units[0].source)
        self.assertEqual(set(payload), {"read_id", "items"})
        behaviors = [x for x in payload["items"] if "behavior_id" in x]
        self.assertTrue(behaviors)
        self.assertTrue(all("trigger" in b and "operation" in b and "result" in b for b in behaviors))
        self.assertEqual(backend._retriever.graph["edges"], [])
        self.assertNotIn("pkg/helper.py", result.units[0].source)
        self.assertEqual(self.graph, self.original)

    def test_no_task_keeps_graph_but_removes_document_preferences(self):
        backend = FixedEntryBackend(self.bundle, self.graph, self.directory, count_tokens=len)
        paths = [x["path"] for x in backend.effective_directory["entries"]]
        self.assertEqual(paths, sorted(paths, key=lambda p: (p.casefold(), p)))
        self.assertNotIn("API", backend.initial_index)
        self.assertEqual(backend._retriever.document_section_by_endpoint, {})
        self.assertEqual(backend._retriever.graph["edges"], self.graph["edges"])
        self.assertEqual(backend.priority_groups, {})
        result = self.read_run(backend)
        self.assertNotIn("Doc:", result.units[0].source)
        self.assertIn("[CONTEXT via", result.units[0].source)
        self.assertEqual(self.graph, self.original)


class TraceAblationTests(unittest.TestCase):
    def test_evidence_json_keeps_scope_edges_and_original_text(self):
        graph = trace_graph()
        full = json.loads(render_trace_view(graph).text)
        view = evidence_trace(graph)
        flat = json.loads(view.text)
        self.assertEqual(flat["current_task_id"], full["current_task_id"])
        self.assertEqual(view.evidence_ids, render_trace_view(graph).evidence_ids)
        for before, after in zip(full["task_scopes"], flat["task_scopes"]):
            self.assertEqual(before["relations"], after["relations"])
            self.assertEqual(before["status"], after["status"])
            self.assertNotIn("behaviors", after)
            self.assertTrue(after["evidence"])
            originals = {e["evidence_id"]: e["content"] for e in graph["evidence"]}
            for item in after["evidence"]:
                a, b = item["char_range"]
                self.assertEqual(item["content"], originals[item["id"]][a:b])
                self.assertIn("event_type", item)
                self.assertIn("event_index", item)
            expected = {e["evidence_id"] for b in before["behaviors"] for role in
                        ("demand", "action", "response") for e in b[role] if e["evidence_id"]}
            self.assertEqual(expected, {e["evidence_id"] for e in after["evidence"] if e["evidence_id"]})

    def test_no_graph_preserves_behaviors_and_removes_task_markers(self):
        graph = trace_graph()
        original = copy.deepcopy(graph)
        full = render_trace_view(graph)
        changed = behavior_trace(graph)
        payload = json.loads(changed.text)
        self.assertEqual(set(payload), {"input_id", "benchmark", "behaviors"})
        self.assertEqual([b["behavior_id"] for b in payload["behaviors"]],
                         [b["behavior_id"] for b in graph["behaviors"]])
        self.assertTrue(all("sequence_index" not in b for b in payload["behaviors"]))
        self.assertEqual(full.evidence_ids, changed.evidence_ids)
        self.assertEqual(graph, original)

    def test_no_task_is_not_applicable(self):
        self.assertFalse(run.supported("feedbacktrace", "no_task"))
        self.assertTrue(run.supported("specgap", "no_task"))

    def test_prompt_does_not_require_removed_trace_fields(self):
        prompt = run.prompt_text("feedbacktrace", "no_graph")
        self.assertNotIn("current_task_id", prompt)
        self.assertNotIn("sequence_index", prompt)
        evidence = run.prompt_text("feedbacktrace", "no_behavior")
        self.assertNotIn("highest `sequence_index`", evidence)
        self.assertIn("evidence_id", evidence)
        self.assertNotIn("`action` or `response` items", evidence)
        self.assertIn("`event_type`", evidence)
        baseline = (run.ROOT / "prompts/baseline/feedbacktrace.txt").read_text(encoding="utf-8")
        for candidate in (prompt, evidence):
            self.assertEqual(candidate.split("DECISION SELECTION RULES", 1)[1],
                             baseline.split("DECISION SELECTION RULES", 1)[1])


class RunnerTests(unittest.TestCase):
    def test_judge_network_recovery_extends_attempts_with_backoff(self):
        batch = run.judge.JudgeBatchConfig(experiment_name="test", experiment_root=Path("."),
                                           phase="full", benchmarks=("specgap",), arms=("graph",),
                                           deferred_retries=False)
        failed = {"status": "failed", "failure": "JudgeNetworkRetriesExhausted: HTTP 429"}
        with patch.object(run.judge, "_run_job", side_effect=[failed, failed, {"status": "complete"}]) as job, \
             patch.object(run.time, "sleep") as sleep:
            result = run.judge_sample(batch, "specgap", "sg_test", None, None)
        self.assertEqual(result["status"], "complete")
        self.assertEqual([call.args[0].network_retries for call in job.call_args_list],
                         [batch.network_retries, batch.network_retries + 1, batch.network_retries + 2])
        sleep.assert_called_once_with(30)

    def test_concurrent_model_progress_files_do_not_collide(self):
        from concurrent.futures import ThreadPoolExecutor
        with ProjectTemporaryDirectory() as root:
            paths = [run.evaluation_progress_path(root, [model], run.BENCHMARKS, ["no_behavior"])
                     for model in ("kimi-k3", "glm-5-3")]
            def update(path):
                for finished in range(20):
                    run.write_json(path, {"finished": finished, "owner": path.name})
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(update, paths))
            self.assertNotEqual(paths[0], paths[1])
            for path in paths:
                self.assertEqual(run.read_json(path), {"finished": 19, "owner": path.name})

    def test_stage_one_format_failure_uses_strict_correction(self):
        profile = {"model": "fake", "base_url": "http://127.0.0.1:1/v1",
                   "api_key_env": "FAKE_KEY", "request_options": {}}
        config = {"timeout": 10, "models": {"luna": {"silentswap": profile}}}
        failed = {"status": "failed", "failure": "finish_format_retries_exhausted", "auto_retried": True}
        with ProjectTemporaryDirectory() as root, \
             patch.object(run.predict, "_run_job", return_value=failed), \
             patch.object(run, "correct_repo_sample", return_value={"status": "complete"}) as correct:
            result = run.predict_sample(config, "silentswap", "no_task", "luna", "ss_test", root)
            self.assertEqual(result["status"], "complete")
            correct.assert_called_once()

    def test_source_review_correction_keeps_evidence_and_resumes_saved_calls(self):
        from agentloop.provider import ModelCompletion
        from tests.agentloop.test_silentswap_source_review import fixture, directory, answer
        bundle, prediction, state = fixture()
        review = run.build_review(bundle, prediction, directory(), state=state)
        bad = answer()
        bad["swaps"][0]["target"]["file"] = "unseen.py"
        replies = [json.dumps(bad), json.dumps(answer())]
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        sent = []
        def complete(messages, **kwargs):
            sent.append(messages)
            return ModelCompletion(replies.pop(0), {}, usage)
        client = SimpleNamespace(complete=complete)
        with ProjectTemporaryDirectory() as root:
            store = run.RunStore(root)
            store.save_response(1, content="{invalid", raw_response={}, usage=usage, provider_retry=0)
            result = run.correct_source_review(review, client, store, ValueError("invalid JSON"))
            self.assertEqual(result["swaps"], answer()["swaps"])
            self.assertTrue((root / "validation_correction_1/validation_error.json").exists())
            self.assertEqual(sent[0][:len(review.messages)], review.messages)
            self.assertIn("unseen.py", sent[1][-1]["content"])
            self.assertEqual(store.load_response(1)["content"], "{invalid")
            total = sum(run.predict.collect_response_usage(p)["total_tokens"]
                        for p in [root, *root.glob("validation_correction_*")])
            self.assertEqual(total, 45)
            client.complete = lambda *a, **k: self.fail("must reuse saved correction")
            self.assertEqual(run.correct_source_review(review, client, store, ValueError("invalid JSON")), result)

    def test_source_review_truncation_increases_only_correction_budget(self):
        from agentloop.provider import ModelCompletion
        from tests.agentloop.test_silentswap_source_review import fixture, directory, answer
        bundle, prediction, state = fixture()
        review = run.build_review(bundle, prediction, directory(), state=state)
        for finish, multiplier in [("length", 2), ("stop", 1)]:
            with self.subTest(finish=finish), ProjectTemporaryDirectory() as root:
                store = run.RunStore(root)
                store.save_response(1, content="", provider_retry=0, usage={},
                                    raw_response={"choices": [{"finish_reason": finish}]})
                budgets = []
                def complete(messages, *, max_output_tokens):
                    budgets.append(max_output_tokens)
                    return ModelCompletion(json.dumps(answer()), {}, {})
                run.correct_source_review(review, SimpleNamespace(complete=complete),
                                          store, ValueError("empty response"))
                self.assertEqual(budgets, [run.DEFAULT_CONFIG.max_output_tokens * multiplier])
                self.assertEqual(store.load_response(1)["content"], "")

    def test_source_review_internal_retry_recovers_truncation_budget_after_resume(self):
        from agentloop.provider import ModelCompletion
        from agentloop.silentswap_source_review import _complete, CONFIG
        for finish, multiplier in [("length", 2), ("stop", 1)]:
            with self.subTest(finish=finish), ProjectTemporaryDirectory() as root:
                store = run.RunStore(root)
                store.save_response(1, content="", provider_retry=0, usage={},
                                    raw_response={"choices": [{"finish_reason": finish}]})
                budgets = []
                def complete(messages, *, max_output_tokens):
                    budgets.append(max_output_tokens)
                    return ModelCompletion("{}", {}, {})
                self.assertEqual(_complete([{"role": "user", "content": "same task"}],
                                           SimpleNamespace(complete=complete), store, 1), "{}")
                self.assertEqual(budgets, [CONFIG.max_output_tokens * multiplier])

    def test_source_review_repeated_empty_truncation_requests_final_answer(self):
        from agentloop.provider import ModelCompletion
        from tests.agentloop.test_silentswap_source_review import fixture, directory, answer
        bundle, prediction, state = fixture()
        review = run.build_review(bundle, prediction, directory(), state=state)
        truncated = {"choices": [{"finish_reason": "length"}]}
        sent = []
        def complete(messages, *, max_output_tokens):
            sent.append(messages)
            return (ModelCompletion("", truncated, {}) if len(sent) < 3
                    else ModelCompletion(json.dumps(answer()), {}, {}))
        with ProjectTemporaryDirectory() as root:
            store = run.RunStore(root)
            store.save_response(1, content="", raw_response=truncated, usage={}, provider_retry=0)
            result = run.correct_source_review(review, SimpleNamespace(complete=complete),
                                              store, ValueError("empty response"))
            self.assertEqual(result["swaps"], answer()["swaps"])
            self.assertIn("Keep deliberation brief", sent[2][-1]["content"])
            self.assertEqual(sent[2][:len(review.messages)], review.messages)

    def test_source_review_can_continue_its_own_truncated_draft(self):
        from agentloop.provider import ModelCompletion
        from tests.agentloop.test_silentswap_source_review import fixture, directory, answer
        bundle, prediction, state = fixture()
        review = run.build_review(bundle, prediction, directory(), state=state)
        draft = "An unfinished candidate from the supplied source."
        truncated = {"choices": [{"finish_reason": "length", "message": {"reasoning_content": draft}}]}
        sent = []
        def complete(messages, *, max_output_tokens):
            sent.append(messages)
            return (ModelCompletion("", truncated, {}) if len(sent) < 5
                    else ModelCompletion(json.dumps(answer()), {}, {}))
        with ProjectTemporaryDirectory() as root:
            store = run.RunStore(root)
            store.save_response(1, content="", raw_response=truncated, usage={}, provider_retry=0)
            client = SimpleNamespace(complete=complete)
            result = run.correct_source_review(review, client, store, ValueError("empty response"))
            self.assertEqual(result["swaps"], answer()["swaps"])
            self.assertEqual(sent[4][:-2], review.messages)
            self.assertTrue(sent[4][-2]["content"].endswith(draft))
            self.assertIn("not additional source evidence", sent[4][-2]["content"])
            client.complete = lambda *a, **k: self.fail("must reuse saved continuation")
            self.assertEqual(run.correct_source_review(review, client, store, ValueError("empty response")), result)

    def test_source_review_tail_repair_preserves_content_and_contract(self):
        from tests.agentloop.test_silentswap_source_review import fixture, directory, answer
        bundle, prediction, state = fixture()
        review = run.build_review(bundle, prediction, directory(), state=state)
        valid = json.dumps(answer())
        bad_location = answer()
        bad_location["swaps"][0]["target"]["file"] = "unseen.py"
        cases = [(valid[:-2] + "}" + valid[-2:], True),
                 (valid + "}", True), (valid[:-2], False),
                 (json.dumps(bad_location) + "}", False)]
        for content, accepted in cases:
            with self.subTest(content=content), ProjectTemporaryDirectory() as root:
                store = run.RunStore(root)
                store.save_response(1, content=content, raw_response={}, usage={}, provider_retry=0)
                self.assertEqual(run.repair_source_review_format(review, store), accepted)
                self.assertEqual(store.load_response(1)["content"], content)
                self.assertEqual((root / "prediction.json").exists(), accepted)
                if accepted:
                    self.assertEqual(run.read_json(root / "prediction.json")["swaps"], answer()["swaps"])

    def test_source_review_continues_after_two_invalid_corrections(self):
        from agentloop.provider import ModelCompletion
        from tests.agentloop.test_silentswap_source_review import fixture, directory, answer
        bundle, prediction, state = fixture()
        review = run.build_review(bundle, prediction, directory(), state=state)
        invalid = json.dumps(answer()) + "}"
        sent = []
        replies = [invalid, invalid, json.dumps(answer())]

        def complete(messages, **kwargs):
            sent.append(messages)
            return ModelCompletion(replies.pop(0), {}, {})

        with ProjectTemporaryDirectory() as root:
            store = run.RunStore(root)
            store.save_response(1, content=invalid, raw_response={}, usage={}, provider_retry=0)
            client = SimpleNamespace(complete=complete)
            result = run.correct_source_review(review, client, store, ValueError("invalid JSON"))
            self.assertEqual(result["swaps"], answer()["swaps"])
            self.assertEqual(len(sent), 3)
            self.assertIn("JSON parser stopped at character", sent[2][-1]["content"])
            self.assertEqual(store.load_response(1)["content"], invalid)
            client.complete = lambda *a, **k: self.fail("must reuse saved corrections")
            self.assertEqual(run.correct_source_review(review, client, store,
                                                      ValueError("invalid JSON")), result)

    def test_evaluation_resumes_completed_stages_without_model_calls(self):
        with ProjectTemporaryDirectory() as temporary:
            root = Path(temporary)
            run.write_json(root / "runs/sg_test/batch_result.json", {"status": "complete"})
            run.write_json(root / "runs/sg_test/prediction.json", {})
            run.write_json(root / "judges/glm/sg_test/status.json", {"status": "complete"})
            run.write_json(root / "judges/glm/sg_test/result.json", {})
            with patch.object(run, "predict_sample", side_effect=AssertionError("must resume")):
                result = run.evaluation_sample({"judges": {"specgap": {"model": "glm"}}},
                                               "luna", "specgap", "no_task", "sg_test", [root], [])
            self.assertTrue(result["resumed"])

    def test_evaluation_scores_stage_one_even_if_source_review_fails(self):
        with ProjectTemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(run, "predict_sample", return_value={"status": "complete"}), \
                 patch.object(run, "source_review_sample", return_value={"status": "failed", "failure": "invalid output"}), \
                 patch.object(run, "judge_sample", return_value={"status": "complete"}) as judge:
                result = run.evaluation_sample({"judges": {"silentswap": {"model": "glm"}}},
                    "luna", "silentswap", "no_task", "ss_test", [root / "stage1", root / "stage2"],
                    [(None, None, None), (None, None, None)])
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["stages"][0]["judge"], "complete")
            self.assertNotIn("judge", result["stages"][1])
            judge.assert_called_once()

    def test_revalidate_saved_answer_preserves_cost_and_rejects_unseen_code(self):
        from evaluation_core.contracts import EvidenceSpan
        finding = {
            "finding_id": "F001", "claim": "Returns two values.",
            "verification_question": "Are both values documented?",
            "why_important": "Observable returns.", "downstream_impact": "Affects callers.",
            "code_evidence": [{"path": "a.py", "start_line": 1, "end_line": 3,
                               "symbol": "module", "explanation": "Both assignments were read."}],
        }
        for gap, expected in (("# comment", "complete"), ("hidden()", "failed")):
            with self.subTest(gap=gap), ProjectTemporaryDirectory() as temporary:
                output = Path(temporary)
                case = output / "runs/sg_test"
                previous = {"status": "failed", "auto_retried": True,
                            "actual_usage": {"total_tokens": 500},
                            "failure": "invalid_finish_grounding_or_content"}
                run.write_json(case / "batch_result.json", previous)
                state = {"turns": 2, "records": [], "config": {
                    "max_read_ids": 6, "max_search_characters": 512}, "terminal_record": {
                    "failure": previous["failure"], "raw_response": json.dumps({
                        "action": "finish", "prediction": {"findings": [finding]}})}}
                run.write_json(case / "attempt_1/state.json", state)
                bundle = SimpleNamespace(repo_artifacts=[SimpleNamespace(
                    path="a.py", content=f"a = 1\n{gap}\nb = 2\n")])
                spans = (EvidenceSpan("a.py", "module", 1, 1), EvidenceSpan("a.py", "module", 3, 3))
                with patch.object(run, "load_visible_bundle", return_value=bundle), \
                     patch.object(run.AgentLoop, "_observed_spans", return_value=spans), \
                     patch.object(run.predict, "collect_response_usage", return_value={"total_tokens": 200}):
                    result = run.revalidate_sample(output, "specgap", "sg_test")
                self.assertEqual(result["status"], expected)
                self.assertEqual(run.read_json(case / "attempt_1/state.json"), state)
                if expected == "complete":
                    self.assertEqual(result["actual_usage"], previous["actual_usage"])
                    self.assertTrue((case / "prediction.json").exists())
                    with patch.object(run, "load_visible_bundle", side_effect=AssertionError("must resume")):
                        self.assertTrue(run.revalidate_sample(output, "specgap", "sg_test")["resumed"])
                else:
                    self.assertFalse((case / "prediction.json").exists())

    def test_specgap_prompts_allow_fewer_than_two_findings(self):
        for variant in run.VARIANTS:
            prompt = run.prompt_text("specgap", variant)
            self.assertIn("There is no minimum number of findings", prompt)
            self.assertIn("return an empty findings array", prompt)
            self.assertNotIn("at least two", prompt)
            self.assertNotIn("MANDATORY MINIMUM", prompt)
        schema = run.load_prediction_schema(run.ROOT / "schemas", "specgap")
        self.assertEqual(schema["properties"]["findings"].get("minItems", 0), 0)

    def test_source_review_uses_variant_directory_and_resumes(self):
        with ProjectTemporaryDirectory() as temporary:
            profile = {"model": "fake", "base_url": "http://127.0.0.1:1/v1",
                       "api_key_env": "FAKE_KEY", "request_options": {}}
            config = {"timeout": 10, "models": {"kimi": {"silentswap": profile}}}
            original, fixed = {"entries": ["task-ranked"]}, {"entries": ["fixed"]}
            backend = SimpleNamespace(effective_directory=fixed,
                                      _retriever=SimpleNamespace(directory=original))
            review = SimpleNamespace(messages=[], selection={"directory": fixed})
            run.write_json(temporary / "stage1/runs/ss_test/prediction.json", {})
            def fake_review(_review, _client, store):
                run.write_json(store.root / "prediction.json", {"input_id": "ss_test"})
            with patch.object(run, "load_stage1", return_value=({}, {}, None)), \
                 patch.object(run, "load_repo", return_value=(None, None, backend)), \
                 patch.object(run, "build_review", return_value=review) as build, \
                 patch.object(run, "save_request"), \
                 patch.object(run.predict, "_client", return_value=object()) as client, \
                 patch.object(run, "run_review", side_effect=fake_review):
                first = run.source_review_sample(config, "kimi", "no_task", "ss_test",
                                                 temporary / "stage1", temporary / "stage2")
                self.assertEqual(first["status"], "complete")
                self.assertEqual(build.call_args.args[2], fixed)
                resumed = run.source_review_sample(config, "kimi", "no_task", "ss_test",
                                                   temporary / "stage1", temporary / "stage2")
                self.assertTrue(resumed["resumed"])
                self.assertEqual(client.call_count, 1)

    def test_judge_reuses_existing_retry_policy(self):
        batch = SimpleNamespace(deferred_retries=1)
        with patch.object(run.judge, "_run_job", return_value={"status": "failed"}), \
             patch.object(run.judge, "_needs_deferred_retry", return_value=True), \
             patch.object(run.judge, "_run_deferred_retry", return_value={"status": "complete"}) as retry:
            result = run.judge_sample(batch, "specgap", "sg_test", None, None)
            self.assertEqual(result["status"], "complete")
            retry.assert_called_once()

    def test_completed_prediction_resumes_without_client(self):
        with ProjectTemporaryDirectory() as temporary:
            output = temporary / "full"
            p = output / "runs/sg_test"
            run.write_json(p / "batch_result.json", {"status": "complete", "input_id": "sg_test"})
            run.write_json(p / "prediction.json", {"input_id": "sg_test"})
            config = {"models": {"kimi": {"specgap": {}}}}
            with patch.object(run.predict, "_run_job", side_effect=AssertionError("must not rerun")):
                self.assertTrue(run.predict_sample(config, "specgap", "full", "kimi", "sg_test", output)["resumed"])

    def test_source_review_waits_for_first_stage(self):
        with ProjectTemporaryDirectory() as root, \
             patch.object(run, "load_repo", side_effect=AssertionError("must not start stage two")):
            result = run.source_review_sample({}, "luna", "no_task", "ss_test",
                                              root / "stage1", root / "stage2")
            self.assertEqual(result["status"], "blocked")
            self.assertFalse((root / "stage2").exists())

    def test_extra_read_attempt_preserves_cost_and_history(self):
        with ProjectTemporaryDirectory() as root:
            for number in (1, 2):
                (root / f"attempt_{number}").mkdir()
            previous = {"status": "failed", "initial_failure": "invalid_finish_grounding_or_content"}
            def retry(*args, **kwargs):
                self.assertEqual(kwargs["attempt"], 3)
                return {"status": "complete", "usage": {"total_tokens": 10}}
            with patch.object(run.predict, "_run_job", side_effect=retry), \
                 patch.object(run.predict, "collect_response_usage", return_value={"total_tokens": 10}):
                result = run.retry_repo_with_reads(None, "silentswap", "ss_test", None, root, previous)
            self.assertEqual(result["actual_usage"]["total_tokens"], 30)
            self.assertEqual(result["repair_source_attempt"], 3)
            self.assertTrue(result["auto_retried"])
            self.assertEqual(run.read_json(root / "attempt_3/original_batch_result.json"), previous)

    def test_judge_correction_preserves_rules_and_reuses_saved_calls(self):
        from tests.scripts.test_judge import (
            _specgap_gold, _specgap_prediction, _perfect_specgap_response,
        )
        from agentloop.provider import ModelCompletion
        with ProjectTemporaryDirectory() as temporary:
            batch = run.judge.JudgeBatchConfig(
                experiment_name="case", experiment_root=temporary, phase="full",
                benchmarks=("specgap",), arms=("graph",))
            root = batch.judge_root / "sg_test"
            gold_path = temporary / "gold.json"
            run.write_json(gold_path, _specgap_gold())
            original = {"gold_path": str(gold_path), "prediction": _specgap_prediction(),
                        "messages": [{"role": "user", "content": "Original scoring rules"}]}
            run.write_json(root / "judge_input.json", original)
            previous = {"status": "failed", "failure": "invalid match_score for F001"}
            bad = _perfect_specgap_response()
            bad["matches"][0]["match_score"] = .9
            responses = [bad, _perfect_specgap_response()]
            sent = []

            def complete(messages, **kwargs):
                sent.append(messages)
                return ModelCompletion(json.dumps(responses.pop(0)), {},
                                       {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})

            factory = lambda: SimpleNamespace(complete=complete)
            module = run.judge._load_judge_module("specgap")
            with patch.object(run.judge, "_validate_client_profile"), \
                 patch.object(module, "load_gold_conditions", return_value=[]), \
                 patch.object(run.judge, "_judge_kwargs", return_value={"document_after": "Public calls return one."}):
                result = run.correct_judge_sample(batch, "specgap", "sg_test", module, factory, previous)
                self.assertEqual(result["status"], "complete", result.get("failure"))
                self.assertEqual(result["validation_correction"], 2)
                self.assertEqual(result["usage"]["total_tokens"], 30)
                self.assertEqual(sent[0][0], original["messages"][0])
                self.assertIn("invalid match_score", sent[1][-1]["content"])
                self.assertEqual(run.read_json(root / "judge_input.json"), original)
                run.correct_judge_sample(batch, "specgap", "sg_test", module, factory, previous)
                self.assertEqual(len(sent), 2)

    def test_trace_correction_validates_ids_preserves_failure_and_counts_usage(self):
        from agentloop.provider import ModelCompletion
        graph = trace_graph()
        input_id = graph["input_id"]
        schema = run.read_json(run.ROOT / "schemas/feedbacktrace_prediction.schema.json")
        valid = {"verification_point": "Check the changed behavior.",
                 "supporting_evidence_ids": [sorted(evidence_trace(graph).evidence_ids)[0]],
                 "criticality": "must_disclose"}
        invalid = {**valid, "supporting_evidence_ids": ["missing_id"]}
        messages = [{"role": "user", "content": "Original visible input"}]
        with ProjectTemporaryDirectory() as temporary:
            root = temporary / "run"
            run.write_json(temporary / "data/prepared/feedbacktrace/artifacts/behavior_graphs" /
                           input_id / "behavior_graph.json", graph)
            state = {"failure": "invalid_prediction", "prediction_schema": schema,
                     "config": {"max_output_tokens": 1000},
                     "terminal_record": {"raw_response": json.dumps(invalid), "provider_retry": 0}}
            run.write_json(root / "state.json", state)
            store = run.RunStore(root)
            store.save_request(1, messages=messages, token_count=10, compression={}, provider_retry=0)
            usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            store.save_response(1, content=json.dumps(invalid), raw_response={}, usage=usage, provider_retry=0)
            previous = {"status": "failed", "failure": "invalid_prediction", "usage": {}}
            config = {"timeout": 10, "models": {"luna": {"feedbacktrace": {}}}}
            client = SimpleNamespace(complete=lambda *a, **kw: ModelCompletion(json.dumps(valid), {}, usage))
            with patch.object(run, "ROOT", temporary), patch.object(run.predict, "_client", return_value=client):
                result = run.correct_trace_sample(config, "luna", "no_behavior", input_id, root, previous)
                self.assertEqual(result["status"], "complete", result.get("failure"))
                self.assertEqual(result["usage"]["total_tokens"], 30)
                self.assertEqual(run.read_json(root / "state.json"), state)
                correction_messages = run.RunStore(root / "format_correction_1").load_request(1, 0)["messages"]
                self.assertEqual(correction_messages[0], messages[0])
                self.assertIn("supporting_evidence_ids (an array of one or two", correction_messages[-1]["content"])
                self.assertIn("Do not return a trace summary", correction_messages[-1]["content"])
                client.complete = lambda *a, **kw: self.fail("saved correction must resume")
                resumed = run.correct_trace_sample(config, "luna", "no_behavior", input_id, root, previous)
                self.assertEqual(resumed["usage"]["total_tokens"], 30)

    def test_repo_correction_keeps_stage_one_schema_history_and_all_attempt_costs(self):
        from agentloop.provider import ModelCompletion
        from evaluation_core.contracts import EvidenceSpan
        schema = run.load_prediction_schema(run.ROOT / "schemas", "silentswap_location_prediction.schema.json")
        schema["properties"]["swaps"].update(minItems=5, maxItems=5)

        def answer(lines):
            return json.dumps({"action": "finish", "prediction": {"swaps": [
                {"target": {"file": "a.py", "symbol": {"kind": "module", "qualified_name": []},
                            "line_ranges": [{"start": n, "end": n}]}} for n in lines]}})

        bad, good = answer([1, 2, 3, 4, 6]), answer([1, 2, 3, 4, 5])
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        bundle = SimpleNamespace(repo_artifacts=[SimpleNamespace(path="a.py", content="x = 1\n" * 6)])
        config = {"timeout": 10, "models": {"luna": {"silentswap": {}}}}
        client = SimpleNamespace(complete=lambda *a, **kw: ModelCompletion(good, {}, usage))
        state = {"input_id": "ss_test", "status": "failed", "records": [],
                 "finish_contract": {"prediction_schema": schema},
                 "config": {"max_read_ids": 6, "max_search_characters": 512, "max_output_tokens": 1000},
                 "terminal_record": {"turn": 1, "provider_retry": 0, "raw_response": bad}}
        with ProjectTemporaryDirectory() as root:
            for number in (1, 2):
                store = run.RunStore(root / f"attempt_{number}")
                store.save_state(state)
                store.save_request(1, messages=[{"role": "user", "content": "Visible source"}],
                                   token_count=10, compression={}, provider_retry=0)
                store.save_response(1, content=bad, raw_response={}, usage=usage, provider_retry=0)
            previous = {"status": "failed", "failure": "invalid_finish_grounding_or_content"}
            with patch.object(run, "load_visible_bundle", return_value=bundle), \
                 patch.object(run.AgentLoop, "_observed_spans", return_value=(EvidenceSpan("a.py", "module", 1, 5),)), \
                 patch.object(run.predict, "_client", return_value=client):
                result = run.correct_repo_sample(config, "luna", "silentswap", "ss_test", root, previous)
                self.assertEqual(result["status"], "complete", result.get("failure"))
                self.assertEqual(result["usage"]["total_tokens"], 30)
                self.assertEqual(result["actual_usage"]["total_tokens"], 45)
                prediction, history, source = run.load_stage1(root)
                self.assertEqual(history, state)
                self.assertEqual(source, root / "attempt_2/state.json")
                self.assertEqual(len(prediction["swaps"]), 5)
                client.complete = lambda *a, **kw: self.fail("saved correction must resume")
                resumed = run.correct_repo_sample(config, "luna", "silentswap", "ss_test", root, previous)
                self.assertEqual(resumed["actual_usage"]["total_tokens"], 45)

    def test_changed_configuration_cannot_resume_old_run(self):
        with ProjectTemporaryDirectory() as temporary:
            run.make_manifest(temporary, "specgap", "full", ["sg_test"], {"model": "a"}, "same prompt")
            with self.assertRaises(ValueError):
                run.make_manifest(temporary, "specgap", "full", ["sg_test"], {"model": "b"}, "same prompt")

    def test_predict_runner_receives_correct_backend_and_prompt(self):
        with ProjectTemporaryDirectory() as temporary:
            config = {"timeout": 10, "models": {"kimi": {"specgap": {
                "model": "fake", "base_url": "http://127.0.0.1:1/v1", "api_key_env": "FAKE_KEY", "request_options": {}}}}}
            run.make_manifest(temporary, "specgap", "no_behavior", ["sg_test"], {}, "prompt")
            def fake_predict(args, **kwargs):
                self.assertIs(kwargs["backend_factory"], EvidenceBackend)
                self.assertEqual(kwargs["task_prompt"], run.prompt_text("specgap", "no_behavior"))
                run.write_json(args.output / "prediction.json", {"input_id": "sg_test"})
                return {"status": "complete", "turns": 1}
            with patch.object(run.predict, "_run", side_effect=fake_predict):
                result = run.predict_sample(config, "specgap", "no_behavior", "kimi", "sg_test", temporary)
            self.assertEqual(result["status"], "complete")

    def test_sample_selection_matches_models_and_no_secrets_saved(self):
        config = run.read_json(run.HERE / "configs/config.json")
        samples = run.read_json(run.HERE / "configs/samples.json")
        published = run.read_json(run.ROOT / "experiments/main/configs/samples.json")
        self.assertEqual(set(config["models"]), {"kimi-k3", "luna", "glm-5-3"})
        for bench in run.BENCHMARKS:
            self.assertEqual(len(samples[bench]), 100)
            self.assertEqual(set(samples[bench]), set(published[bench]))
            for profile in config["models"].values():
                self.assertNotIn("api_key", profile[bench])
                self.assertTrue(profile[bench]["source_manifest"].startswith("outputs/main/"))

    def test_profiles_match_archived_manifests_when_available(self):
        config = run.read_json(run.HERE / "configs/config.json")
        samples = run.read_json(run.HERE / "configs/samples.json")
        if not all((run.ROOT / profile["source_manifest"]).is_file()
                   for model in config["models"].values() for profile in model.values()):
            self.skipTest("requires archived main-run manifests")
        for bench, profile in config["models"]["glm-5-3"].items():
            source = run.read_json(run.ROOT / profile["source_manifest"])
            self.assertEqual(profile["model"], source["model"])
            self.assertEqual(profile["request_options"], source["prediction_requests"][bench])
        for bench in run.BENCHMARKS:
            self.assertEqual(len(samples[bench]), 100)
            for profile in config["models"].values():
                self.assertNotIn("api_key", profile[bench])
                source = run.read_json(run.ROOT / profile[bench]["source_manifest"])
                self.assertEqual(samples[bench], source["selected_ids"][bench])


class AnalysisTests(unittest.TestCase):
    def test_failed_prediction_tokens_are_counted_without_fabricating_scores(self):
        with ProjectTemporaryDirectory() as temporary:
            root = run.stage_root(temporary, "luna", "specgap", "no_behavior")
            run.write_json(root / "runs/sg_001/batch_result.json", {
                "status": "failed", "failure": "invalid_finish_grounding_or_content",
                "usage": {"total_tokens": 0},
                "actual_usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}})
            result = analyze.prediction_progress(temporary, "luna", "specgap", "no_behavior", ["sg_001", "sg_002"])
            self.assertEqual(result["prediction_status"]["stage1"], {"complete": 0, "failed": 1, "pending": 1})
            self.assertEqual(result["all_finished_prediction_tokens"]["total_tokens"], 25)
            self.assertEqual(result["prediction_failures"][0]["input_id"], "sg_001")

    def test_paired_bootstrap_direction_and_shared_samples(self):
        summary = analyze.paired_summary([(0.8, 0.6, "r1"), (0.3, 0.1, "r2")], repeats=100)
        self.assertAlmostEqual(summary["delta"], -.2)
        for value in summary["ci95"]:
            self.assertAlmostEqual(value, -.2)
        self.assertIsNone(analyze.paired_summary([(1, 0, "same"), (0, 1, "same")])["ci95"])

    def test_missing_judgments_are_incomplete_not_zero(self):
        with ProjectTemporaryDirectory() as temporary:
            config = {"models": {"kimi": {}}, "judges": {"specgap": {"model": "glm"}}}
            result = analyze.analyze(temporary, config, {"specgap": ["sg_test"]}, repeats=10)
            self.assertEqual(result["samples"], [])
            self.assertTrue(all(c["metric_means"] is None for c in result["conditions"]))
            self.assertTrue(all(c["statistics"] is None for c in result["comparisons"]))

    def test_partial_scores_are_separate_from_full_experiment_means(self):
        config = {"models": {"luna": {}}}
        row = {"metrics": {k: .5 for k in analyze.METRICS["specgap"]},
               "usage": {k: 1 for k in analyze.TOKEN_KEYS},
               "actual_usage": {k: 1 for k in analyze.TOKEN_KEYS}}
        with patch.object(analyze, "load_sample", side_effect=[row, None]):
            result = analyze.analyze(Path('.'), config, {"specgap": ["sg_001", "sg_002"]},
                                     variants=("no_behavior",))
        condition = result["conditions"][0]
        self.assertEqual(condition["completed_samples"], 1)
        self.assertIsNone(condition["metric_means"])
        self.assertEqual(condition["completed_sample_means"]["f1"], .5)
        self.assertEqual(result["comparisons"], [])

    def test_silentswap_uses_two_stages_for_scores_and_tokens(self):
        with ProjectTemporaryDirectory() as temporary:
            config = {"judges": {"silentswap": {"model": "glm"}}}
            for stage, score in (("stage1", .25), ("stage2", .9)):
                root = run.stage_root(temporary, "kimi", "silentswap", "full", stage)
                run.write_json(root / "runs/ss_test/batch_result.json", {
                    "status": "complete", "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}})
                run.write_json(root / "judges/glm/ss_test/status.json", {
                    "status": "complete", "metrics": {"localization_score": .5,
                    "location_correct": .75, "code_change_correct": score}})
            row = analyze.load_sample(temporary, config, "kimi", "silentswap", "full", "ss_test")
            self.assertEqual(row["metrics"], {"localization_score": .5, "location_correct": .75, "code_change_correct": .9})
            self.assertEqual(row["usage"]["total_tokens"], 30)
            prediction_path = root / "runs/ss_test/batch_result.json"
            failed = run.read_json(prediction_path)
            failed["status"] = "failed"
            run.write_json(prediction_path, failed)
            self.assertIsNone(analyze.load_sample(temporary, config, "kimi", "silentswap", "full", "ss_test"))
            judge_path = root / "judges/glm/ss_test/status.json"
            judged = run.read_json(judge_path)
            judged["score_origin"] = "user_requested_prediction_failure_zero"
            judged["metrics"]["code_change_correct"] = 0
            run.write_json(judge_path, judged)
            row = analyze.load_sample(temporary, config, "kimi", "silentswap", "full", "ss_test")
            self.assertEqual(row["metrics"], {"localization_score": .5, "location_correct": .75, "code_change_correct": 0})
            self.assertEqual(run.read_json(prediction_path)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
