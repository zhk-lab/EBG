"""Tests for paired evidence-search comparisons."""

import unittest
from unittest.mock import patch

import compare_models as summary
from compare_models import (
    JUDGES, METRICS, aggregate,
    search_axis, slope, slope_interval, summarize_trends,
)



class SummaryTests(unittest.TestCase):
    def test_bootstrap_keeps_constant_paired_advantage(self):
        rows = [{"model": "test", "judge": "test", "component": "specgap",
                 "bin": "Low", "input_id": str(i), "method": method,
                 "metrics": {"f1": score + (0.25 if method == "EBG" else 0)}}
                for i, score in enumerate((0, 0.25, 0.5)) for method in ("EBG", "baseline")]
        result = summary.group_medians(rows, repeats=100)[0]
        self.assertEqual(result["mean_difference"], 0.25)
        self.assertEqual(result["mean_difference_ci95"], [0.25, 0.25])

    def test_group_plot_uses_median_instead_of_mean(self):
        rows = [{"model": "test", "judge": "test", "component": "specgap",
                 "bin": "Low", "input_id": str(i), "method": method,
                 "metrics": {"f1": value if method == "EBG" else 0.2}}
                for i, value in enumerate([0.0, 0.0, 0.9])
                for method in ("EBG", "baseline")]
        result = summary.group_medians(rows, repeats=100)[0]
        self.assertEqual(result["n_pairs"], 3)
        self.assertEqual(result["EBG_median"], 0)
        self.assertEqual(result["baseline_median"], 0.2)
        self.assertEqual(result["baseline_ci95"], [0.2, 0.2])
        with self.assertRaisesRegex(ValueError, "Missing paired"):
            summary.group_medians(rows[1:], repeats=10)

    @patch.object(summary, "read")
    def test_luna_uses_component_specific_directory(self, read):
        for component, directory in (("specgap", "gpt5.6"), ("feedbacktrace", "luna")):
            read.return_value = {"input_id": "sample", "status": "complete",
                                 "metrics": {key: 0.5 for key in summary.METRICS[component]}}
            _, source = summary.load_scores(component, "baseline", "glm-5-2", "sample", model="luna")
            judge = "glm-5.2" if component == "feedbacktrace" else "glm-5-2"
            self.assertEqual(source, f"outputs/main/baseline/{component}/{directory}/judges/{judge}/sample/status.json")

    def rows(self):
        return [{"judge": "test", "component": "test", "input_id": str(i), "bin": label,
                 "method": method, "metrics": {"score": score if method == "EBG" else 0}}
                for i, (label, score) in enumerate([("Low", 0), ("Medium", 0), ("High", 1), ("High", 1)])
                for method in ("EBG", "baseline")]

    @patch.object(summary, "JUDGES", ("test",))
    @patch.object(summary, "METRICS", {"test": {"score": "Score"}})
    def test_all_is_sample_weighted(self):
        overall = summary.aggregate(self.rows())[-1]
        self.assertEqual(overall["n_pairs"], 4)
        self.assertEqual(overall["EBG"], 0.5)
        self.assertEqual(overall["difference"], 0.5)

    @patch.object(summary, "JUDGES", ("test",))
    @patch.object(summary, "METRICS", {"test": {"score": "Score"}})
    def test_missing_pair_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing paired"):
            summary.aggregate(self.rows()[1:])


class TrendTests(unittest.TestCase):
    @patch("compare_models.slope_interval", return_value=[-0.1, 0.1])
    def test_all_three_metrics_are_analyzed_per_benchmark(self, interval):
        rows = [{"model": "sol", "judge": judge, "component": component,
                 "input_id": str(i), "bin": label, "search_count": 2 ** (i + 1),
                 "method": method, "metrics": {metric: 0.1 * (i + 1) if method == "EBG" else 0
                                               for metric in metrics}}
                for judge in JUDGES for component, metrics in METRICS.items()
                for i, label in enumerate(("Low", "Medium", "High"))
                for method in ("EBG", "baseline")]
        trends = summarize_trends(rows, aggregate(rows))
        expected = {(judge, component, metric) for judge in JUDGES
                    for component, metrics in METRICS.items() for metric in metrics}
        self.assertEqual({(t["judge"], t["component"], t["metric"]) for t in trends}, expected)
        for trend in trends:
            self.assertAlmostEqual(trend["slope_difference"], 0.1)

    def test_advantage_growth_has_positive_sign(self):
        x = [1, 2, 3, 4]
        ebg = [0.8, 0.7, 0.6, 0.5]
        baseline = [0.7, 0.5, 0.3, 0.1]
        delta = [b - a for b, a in zip(ebg, baseline)]
        self.assertAlmostEqual(slope(x, delta), 0.1)
        self.assertAlmostEqual(slope(x, ebg) - slope(x, baseline), slope(x, delta))

    def test_constant_advantage_has_zero_slope(self):
        self.assertAlmostEqual(slope([1, 2, 4], [0.2, 0.2, 0.2]), 0)

    def test_no_workload_variation_rejected(self):
        with self.assertRaisesRegex(ValueError, "no variation"):
            slope([1, 1], [0, 1])

    def test_bootstrap_preserves_exact_linear_relation(self):
        lo, hi = slope_interval([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4], repeats=100)
        self.assertAlmostEqual(lo, 0.1)
        self.assertAlmostEqual(hi, 0.1)

    def test_axes_have_component_specific_units(self):
        self.assertEqual(search_axis("specgap", 4), 2)
        self.assertEqual(search_axis("silentswap", 4), 2)
        self.assertEqual(search_axis("feedbacktrace", 4), 2)


if __name__ == "__main__":
    unittest.main()
