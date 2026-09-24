import unittest

from hop_control import summarize_group


class HopControlTest(unittest.TestCase):
    def test_coverage_and_correctness_are_separate(self):
        rows = [dict(evidence_complete=True, outcome='full'),
                dict(evidence_complete=True, outcome='partial'),
                dict(evidence_complete=False, outcome='full'),
                dict(error='missing request')]
        self.assertEqual(summarize_group(rows), dict(
            total=4, verified=3, pending=1, evidence_completed=2,
            repaired=2, completed_and_repaired=1, completed_but_failed=1))

    def test_empty_group(self):
        self.assertEqual(summarize_group([])['verified'], 0)


if __name__ == '__main__':
    unittest.main()
