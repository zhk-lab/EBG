import unittest
from threading import Event

from pipeline import execute


class PipelineTests(unittest.TestCase):
    def test_overlap_and_benchmark_barrier(self):
        judge_started = Event()
        next_prediction = Event()
        judged = []

        def predict(benchmark, variant):
            if (benchmark, variant) == ("repo", "two"):
                self.assertTrue(judge_started.wait(5))
                next_prediction.set()
            if benchmark == "trace":
                self.assertEqual(judged[:2], [("repo", "one"), ("repo", "two")])

        def judge(benchmark, variant):
            if (benchmark, variant) == ("repo", "one"):
                judge_started.set()
                self.assertTrue(next_prediction.wait(5))
            judged.append((benchmark, variant))

        execute(predict, judge, ["repo", "trace"], ["one", "two"],
                lambda b, v: (b, v) != ("trace", "two"))
        self.assertEqual(judged, [("repo", "one"), ("repo", "two"), ("trace", "one")])

    def test_failed_prediction_is_not_judged(self):
        def fail(*args):
            raise RuntimeError("prediction failed")

        with self.assertRaisesRegex(RuntimeError, "prediction failed"):
            execute(fail, lambda *args: self.fail("judged failed predictions"),
                    ["repo"], ["one"], lambda *args: True)


if __name__ == "__main__":
    unittest.main()
