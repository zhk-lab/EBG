import unittest
from unittest.mock import patch

from relative_gain import relative_gain, summarize


class RelativeGainTests(unittest.TestCase):
    def test_zero_baseline_is_undefined(self):
        self.assertIsNone(relative_gain(0.5, 0))
        self.assertIsNone(relative_gain(0, 0))

    def test_signed_percentage(self):
        self.assertAlmostEqual(relative_gain(0.6, 0.4), 50)
        self.assertAlmostEqual(relative_gain(0.2, 0.4), -50)

    @patch("relative_gain.METRICS", {"test": {"score": "Score"}})
    def test_ratio_of_means_not_mean_of_ratios(self):
        rows = [{"model": "m", "judge": "j", "component": "test", "bin": "Low",
                 "input_id": str(i), "method": method, "metrics": {"score": score}}
                for i, pair in enumerate(((0.2, 0.1), (0.4, 0.4)))
                for method, score in zip(("EBG", "baseline"), pair)]
        result = summarize(rows)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0]["relative_gain_pct"], 20)
        with self.assertRaisesRegex(ValueError, "Missing paired"):
            summarize(rows[1:])


if __name__ == "__main__":
    unittest.main()
