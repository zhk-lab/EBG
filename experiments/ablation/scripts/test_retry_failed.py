import unittest
import io
import json
from urllib.request import Request
from unittest.mock import patch

import run
from retry_failed import archive_failure, reconcile_prediction_usage, stream_opener
from tests.support import ProjectTemporaryDirectory


class RetryTests(unittest.TestCase):
    def test_usage_reconciliation_survives_interruption_and_is_idempotent(self):
        with ProjectTemporaryDirectory() as results:
            case = results / "model/bench/runs/sample"
            run.write_json(case / "batch_result.json", {"status": "failed"})
            run.write_json(case / "responses/turn_001.json", {
                "usage": {"prompt_tokens": 10, "completion_tokens": 2}})
            archive_failure(case, "batch_result.json", results)
            run.write_json(case / "batch_result.json", {
                "status": "complete", "usage": {"total_tokens": 5}})
            run.write_json(case / "responses/turn_001.json", {
                "usage": {"prompt_tokens": 4, "completion_tokens": 1}})
            for _ in range(2):
                reconcile_prediction_usage(case, results)
                record = run.read_json(case / "batch_result.json")
                self.assertEqual(record["actual_usage"]["total_tokens"], 17)
                self.assertEqual(record["actual_usage"]["calls"], 2)
                self.assertEqual(record["usage"], {"total_tokens": 5})
                self.assertEqual(record["status"], "complete")

    def test_stream_preserves_text_and_usage_and_rejects_truncation(self):
        chunks = [{"choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}]},
                  {"usage": {"total_tokens": 12}, "choices": []}]
        for complete in (True, False):
            response = io.BytesIO(b"".join(b"data: " + json.dumps(c).encode() + b"\n"
                                          for c in (chunks if complete else [])))
            response.headers = {"Content-Type": "text/event-stream"}
            with patch("retry_failed.urllib.request.urlopen", return_value=response):
                req = Request("http://localhost/v1/chat/completions", data=b'{"model":"test"}')
                if complete:
                    result = json.load(stream_opener(req, 10))
                    self.assertEqual(result["choices"][0]["message"]["content"], "OK")
                    self.assertEqual(result["usage"]["total_tokens"], 12)
                else:
                    with self.assertRaises(OSError):
                        stream_opener(req, 10)

    def test_failure_is_retained_and_success_is_untouched(self):
        with ProjectTemporaryDirectory() as results:
            case = results / "model/bench/runs/sample"
            failed = {"status": "failed", "actual_usage": {"total_tokens": 12}}
            run.write_json(case / "batch_result.json", failed)
            run.write_json(case / "responses/response.json", {"content": "original"})
            self.assertEqual(archive_failure(case, "batch_result.json", results), {"total_tokens": 12})
            self.assertFalse(case.exists())
            saved = results / "retry_history/model/bench/runs/sample/1"
            self.assertEqual(run.read_json(saved / "batch_result.json"), failed)
            self.assertEqual(run.read_json(saved / "responses/response.json"), {"content": "original"})
            run.write_json(case / "batch_result.json", {"status": "complete"})
            self.assertIsNone(archive_failure(case, "batch_result.json", results))
            self.assertTrue(case.exists())


if __name__ == "__main__":
    unittest.main()
