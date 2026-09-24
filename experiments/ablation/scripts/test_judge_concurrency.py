import unittest
from types import SimpleNamespace
from judge_concurrency import AdaptiveConcurrency


class ConcurrencyTests(unittest.TestCase):
    def test_rate_limit_reduces_once_per_burst_then_grows_after_stability(self):
        now = [100.0]
        gate = AdaptiveConcurrency(initial=10, clock=lambda: now[0], successes_to_grow=2)
        gate.acquire()
        gate.acquire()
        gate.release(rate_limited=True)
        self.assertEqual(gate.limit, 5)
        gate.release(rate_limited=True)
        self.assertEqual(gate.limit, 5)
        self.assertEqual(gate.rate_limits, 2)
        self.assertEqual(gate.active, 0)
        now[0] = 161
        for _ in range(2):
            gate.acquire()
            gate.release(success=True)
        self.assertEqual(gate.limit, 7)

    def test_wrapped_client_preserves_profile_and_releases_on_errors(self):
        gate = AdaptiveConcurrency()
        def fail(*args, **kwargs):
            raise RuntimeError('provider HTTP 429')
        client = SimpleNamespace(profile={'model': 'qwen'}, complete=fail)
        wrapped = gate.wrap(client)
        self.assertEqual(wrapped.profile, client.profile)
        with self.assertRaisesRegex(RuntimeError, '429'):
            wrapped.complete([], max_output_tokens=1)
        self.assertEqual(gate.limit, 2)
        self.assertEqual(gate.active, 0)

    def test_other_error_does_not_reduce_limit(self):
        gate = AdaptiveConcurrency()
        gate.acquire()
        gate.release()
        self.assertEqual(gate.limit, 5)
        self.assertEqual(gate.active, 0)


if __name__ == '__main__':
    unittest.main()
