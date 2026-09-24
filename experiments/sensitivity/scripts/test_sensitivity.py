"""Focused regression tests for depth, budgets, experiment reuse and inference."""

import unittest
from unittest.mock import patch

import run
from run import Condition, HERE, ROOT, Path, conditions, manifest, read_json
from summarize import holm, paired_statistics, summarize, usage_sum
from agentloop.errors import AgentLoopError
from agentloop.graph_backend import GraphBackend
from ebg.local_graph_retrieval import LocalGraphRetriever, RetrievalError
from tests.ebg.test_local_graph_retrieval import assemble, VALIDATE_LOCAL_GRAPH
from tests.support import ProjectTemporaryDirectory, make_repo_bundle


def chain(cycle=False):
    return {
        "a.py": "from b import beta\ndef alpha(x):\n    return beta(x)\n",
        "b.py": "from c import gamma\ndef beta(x):\n    return gamma(x)\n",
        "c.py": "from d import delta\ndef gamma(x):\n    return delta(x)\n",
        "d.py": ("from a import alpha\ndef delta(x):\n    return alpha(x)\n" if cycle else
                 "def delta(x):\n    return x\n"),
    }


def build(root, cycle=False):
    return assemble(make_repo_bundle(root, repository_files=chain(cycle), document="# Task\n`alpha` in `a.py`.\n"))


def nodes(value):
    return [n for graph in value["graphs"] for n in
            [graph["root"], *(s["node"] for p in graph["paths"] for s in p["steps"])] if "ref" not in n]


class DepthTests(unittest.TestCase):
    def test_chain_depth_and_default_equivalence(self):
        with ProjectTemporaryDirectory() as root:
            args = build(root)
            baseline = LocalGraphRetriever(*args, count_tokens=len)
            read_id = baseline.root_read_id_by_endpoint[("a.py", "alpha")]
            previous = None
            for depth in (1, 2, 3):
                retriever = LocalGraphRetriever(*args, count_tokens=len, expansion_hops=depth)
                result = retriever.read(read_id)
                VALIDATE_LOCAL_GRAPH(result)
                self.assertEqual({n["path"] for n in nodes(result)}, set("abcd"[:depth + 1][i] + ".py" for i in range(depth + 1)))
                self.assertEqual(retriever.last_read_stats["actual_depth"], depth)
                if depth == 1:
                    self.assertEqual(result, baseline.read(read_id))
                if previous:
                    self.assertEqual(result["graphs"][0]["paths"][:len(previous["graphs"][0]["paths"])],
                                     previous["graphs"][0]["paths"])
                previous = result

    def test_compact_graph_backend_preserves_groundable_deep_source(self):
        with ProjectTemporaryDirectory() as root:
            args = build(root)
            backend = GraphBackend(*args, count_tokens=len, expansion_hops=3)
            read_id = backend._retriever.root_read_id_by_endpoint[("a.py", "alpha")]
            result = backend.read([read_id], token_budget=32000, max_atomic_unit_tokens=65000, count_tokens=len)
            self.assertEqual({r.path for r in result.units[0].source_regions}, {"a.py", "b.py", "c.py", "d.py"})
            self.assertEqual(backend._retriever.last_read_stats["actual_depth"], 3)

    def test_budget_keeps_whole_one_hop_view(self):
        with ProjectTemporaryDirectory() as root:
            args = build(root)
            base = LocalGraphRetriever(*args, count_tokens=len)
            read_id = base.root_read_id_by_endpoint[("a.py", "alpha")]
            value = base.read(read_id)
            budget = base._measure_local_graph(value)
            deeper = LocalGraphRetriever(*args, count_tokens=len, expansion_hops=3)
            self.assertEqual(deeper.read(read_id, token_budget=budget), value)
            self.assertEqual(deeper.last_read_stats["actual_depth"], 1)
            self.assertGreater(deeper.last_read_stats["budget_rejected_nodes"], 0)

    def test_cycles_do_not_repeat_source_or_expand_forever(self):
        with ProjectTemporaryDirectory() as root:
            args = build(root, cycle=True)
            retriever = LocalGraphRetriever(*args, count_tokens=len, expansion_hops=3)
            result = retriever.read(retriever.root_read_id_by_endpoint[("a.py", "alpha")])
            keys = [(n["path"], n["symbol"], *n["lines"]) for n in nodes(result)]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(len(keys), 4)
            self.assertLessEqual(max(len(p["steps"]) for g in result["graphs"] for p in g["paths"]), 3)

    def test_resume_identity_changes_only_for_nondefault_depth(self):
        with ProjectTemporaryDirectory() as root:
            args = build(root)
            self.assertEqual(GraphBackend(*args, count_tokens=len).resume_identity,
                             GraphBackend(*args, count_tokens=len, expansion_hops=1).resume_identity)
            self.assertNotEqual(GraphBackend(*args, count_tokens=len).resume_identity,
                                GraphBackend(*args, count_tokens=len, expansion_hops=2).resume_identity)
            for depth in (0, 4, True, 1.5):
                with self.assertRaises(RetrievalError):
                    LocalGraphRetriever(*args, count_tokens=len, expansion_hops=depth)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(HERE / "configs/config.json")

    def test_default_groups_are_shared_and_baseline_has_no_stage2(self):
        for benchmark in run.BENCHMARKS:
            chosen = conditions(self.config, benchmark)
            self.assertEqual(len(chosen), 8)
            default = Condition(benchmark, "graph", 1, self.config["defaults"]["max_rounds"][benchmark])
            self.assertEqual(chosen.count(default), 1)
            self.assertTrue(all(not c.has_stage2 for c in chosen if c.arm == "raw"))

    def test_manifest_rejects_changed_rounds_and_profile(self):
        with ProjectTemporaryDirectory() as root:
            condition = Condition("specgap", "raw", 1, 2)
            profile = self.config["models"]["luna"]["specgap"]
            manifest(root, condition, ["sg_001"], profile)
            manifest(root, condition, ["sg_001"], profile)
            with self.assertRaises(AgentLoopError):
                manifest(root, Condition("specgap", "raw", 1, 4), ["sg_001"], profile)
            with self.assertRaises(AgentLoopError):
                manifest(root, condition, ["sg_001"], {**profile, "model": "different"})

    def test_round_override_reaches_initial_protocol(self):
        with ProjectTemporaryDirectory() as root:
            bundle, _, _ = build(root)
            args = run.argparse.Namespace(benchmark="specgap", input_id=bundle.input_id, arm="raw",
                                          artifact_root=None, output=root / "out", schema_root=ROOT / "schemas", prepare_only=True)
            with patch.object(run.predict, "load_visible_bundle", return_value=bundle), \
                 patch.object(run.predict, "prepare_initial_request", wraps=run.predict.prepare_initial_request) as prepare, \
                 patch.object(run.predict, "_client", side_effect=AssertionError("offline must not call client")):
                run.run_one(args, Condition("specgap", "raw", 1, 2))
            self.assertEqual(prepare.call_args.args[1], 2)

    def test_completed_sample_skips_api(self):
        with ProjectTemporaryDirectory() as root:
            path = root / "runs/sg_001"
            run.write_json(path / "batch_result.json", {"status": "complete", "turns": 2})
            run.write_json(path / "prediction.json", {"input_id": "sg_001"})
            with patch.object(run.predict, "_run_job", side_effect=AssertionError("must reuse completed sample")):
                result = run.predict_sample(self.config, "luna", Condition("specgap", "raw", 1, 2), "sg_001", root)
            self.assertTrue(result["resumed"])

    def test_read_budget_override_preserves_other_limits_and_stage2(self):
        with ProjectTemporaryDirectory() as root:
            bundle, _, _ = build(root)
            args = run.argparse.Namespace(benchmark="specgap", input_id=bundle.input_id, arm="raw",
                                          artifact_root=None, output=root / "out", schema_root=ROOT / "schemas", prepare_only=True)
            with patch.object(run.predict, "load_visible_bundle", return_value=bundle), \
                 patch.object(run.predict, "prepare_initial_request", wraps=run.predict.prepare_initial_request) as prepare:
                run.run_one(args, Condition("specgap", "raw", 3, 6), 32768)
            actual = prepare.call_args.kwargs["config"].public_dict()
            expected = run.DEFAULT_CONFIG.public_dict()
            expected["tool_result_budget"] = 32768
            self.assertEqual(actual, expected)
            condition = Condition("silentswap", "graph", 3, 6)
            profile = self.config["models"]["luna"]["silentswap"]
            manifest(root / "stage1", condition, ["ss_001"], profile, tool_result_budget=32768)
            manifest(root / "stage2", condition, ["ss_001"], profile, "stage2", 32768)
            self.assertEqual(read_json(root / "stage1/manifest.json")["runner_limits"]["tool_result_budget"], 32768)
            self.assertEqual(read_json(root / "stage2/manifest.json")["runner_limits"]["tool_result_budget"], 65536)
            with self.assertRaises(AgentLoopError):
                manifest(root / "stage1", condition, ["ss_001"], profile)

    def test_incomplete_pairs_do_not_report_formal_scores(self):
        with ProjectTemporaryDirectory() as root:
            result = summarize(root, self.config, {"specgap": ["sg_001", "sg_002"]},
                               models=["luna"], benchmarks=["specgap"], independent_samples=True, repeats=10)
            self.assertTrue(all(r["status"] == "incomplete" and r["p_adjusted"] is None for r in result["comparisons"]))

    def test_two_stage_scores_and_accepted_usage_are_combined(self):
        from summarize import load_sample
        with ProjectTemporaryDirectory() as root:
            condition = Condition("silentswap", "graph", 1, 2)
            first, second = [condition.root(root, "luna", stage) for stage in ("stage1", "stage2")]
            prediction = {"input_id": "ss_001"}
            state = {"input_id": "ss_001", "status": "complete", "max_rounds": 2, "turns": 2,
                     "records": [{"turn": 1, "usage": {"total_tokens": 10},
                                  "tool_result": {"action": "read", "units": []}}],
                     "terminal_record": {"turn": 2, "usage": {"total_tokens": 20}}}
            run.write_json(first / "runs/ss_001/attempt_1/state.json", state)
            run.write_json(first / "runs/ss_001/attempt_1/prediction.json", prediction)
            judge_model = self.config["judges"]["silentswap"]["model"]
            for index, stage in enumerate((first, second)):
                case = stage / "runs/ss_001"
                run.write_json(case / "prediction.json", prediction)
                run.write_json(case / "batch_result.json", {"status": "complete", "usage": {"total_tokens": 100}})
                run.write_json(stage / "judges" / judge_model / "ss_001/status.json",
                               {"status": "complete", "metrics": {"localization_score": .5,
                                "location_correct": .4, "code_change_correct": .2 if index == 0 else .8}})
            run.write_json(second / "runs/ss_001/state.json", {"format_corrections": 1})
            run.write_json(second / "runs/ss_001/responses/turn_001_format_01.json",
                           {"content": "accepted", "usage": {"total_tokens": 40}})
            row = load_sample(root, self.config, "luna", condition, "ss_001")
            self.assertEqual(row["metrics"]["code_change_correct"], .8)
            self.assertEqual(row["metrics"]["location_correct"], .4)
            self.assertEqual(row["usage"]["total_tokens"], 70)
            self.assertEqual(row["actual_usage"]["total_tokens"], 200)

    def test_stage2_refuses_changed_upstream_before_reusing_output(self):
        with ProjectTemporaryDirectory() as root:
            first, second = root / "first", root / "second"
            run.write_json(first / "runs/ss_001/batch_result.json", {"status": "complete"})
            run.write_json(second / "runs/ss_001/stage1_input.json", {"prediction": "old", "records": []})
            run.write_json(second / "runs/ss_001/batch_result.json", {"status": "complete"})
            run.write_json(second / "runs/ss_001/prediction.json", {})
            with patch.object(run, "load_stage1", return_value=({"input_id": "ss_001"}, {"records": []}, None)):
                with self.assertRaises(AgentLoopError):
                    run.stage2_sample(self.config, "luna", "ss_001", first, second)


class StatisticsTests(unittest.TestCase):
    def test_clustered_inference_and_single_cluster(self):
        pairs = [(0, 1, "same")] * 20
        self.assertIsNone(paired_statistics(pairs, repeats=20)["p_value"])
        stats = paired_statistics([(0, 1, str(i)) for i in range(6)], repeats=20)
        self.assertEqual(stats["p_value"], 1 / 64)
        self.assertEqual(stats["ci95"], [1, 1])
        self.assertEqual(paired_statistics([(1, 0, str(i)) for i in range(6)], repeats=20)["p_value"], 1)

    def test_holm_uses_full_planned_family(self):
        rows = [{"p_value": .001, "delta": 1}, {"p_value": .03, "delta": 1}]
        holm(rows, 10)
        self.assertAlmostEqual(rows[0]["p_adjusted"], .01)
        self.assertAlmostEqual(rows[1]["p_adjusted"], .27)
        self.assertTrue(rows[0]["significantly_better"])
        self.assertFalse(rows[1]["significantly_better"])

    def test_token_aliases(self):
        self.assertEqual(usage_sum([{"prompt_tokens": 3, "completion_tokens": 2},
                                    {"input_tokens": 4, "output_tokens": 1}])["total_tokens"], 10)


class ModelSelectionTests(unittest.TestCase):
    def test_plan_accepts_configured_models_and_rejects_unknown(self):
        with patch.object(run, "write_json") as write, patch("builtins.print"):
            self.assertEqual(run.main(["plan", "--model", "luna", "kimi-k3", "glm-5-3"]), 0)
        self.assertEqual(write.call_args.args[1]["models"], ["luna", "kimi-k3", "glm-5-3"])
        with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            run.main(["plan", "--model", "deepseek_flash"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
