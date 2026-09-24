import unittest

from summarize import screen


class ScreeningTests(unittest.TestCase):
    def test_silentswap_uses_stage1_location_not_semantic_correctness(self):
        gold = {'swaps': [{}]}
        result = {'llm_judge': {
            'location_checks': [{'reference_swap_number': 1, 'fully_correct': False}],
            'code_change_checks': [{'reference_swap_number': 1, 'fully_correct': True}],
        }}
        self.assertEqual(screen('silentswap', gold, result)[0]['outcome'], 'incorrect')

    def test_specgap_partial_is_not_full_or_unmatched(self):
        gold = {'conditions': [{'condition_id': k, 'implementation_locations': [{'file': 'a.py'}]} for k in ('a', 'b', 'c')]}
        result = {'llm_judge': {'judgment': {'matches': [
            {'condition_id': 'a', 'match_score': 1},
            {'condition_id': 'b', 'match_score': 0.5},
        ]}}}
        self.assertEqual([r['outcome'] for r in screen('specgap', gold, result)],
                         ['full', 'partial', 'unmatched'])

    def test_specgap_without_location_is_excluded_not_failed(self):
        gold = {'conditions': [{'condition_id': 'a', 'implementation_locations': []}]}
        result = {'llm_judge': {'judgment': {'matches': []}}}
        self.assertEqual(screen('specgap', gold, result), [])

    def test_silentswap_matches_by_gold_id_not_list_order(self):
        gold = {'swaps': [{}, {}, {}]}
        result = {'llm_judge': {'location_checks': [
            {'reference_swap_number': 2, 'fully_correct': True},
            {'reference_swap_number': 1, 'fully_correct': False},
        ]}}
        self.assertEqual([r['outcome'] for r in screen('silentswap', gold, result)],
                         ['incorrect', 'full', 'unknown'])

    def test_trace_location_failure_does_not_define_disclosure_failure(self):
        result = {'verification_point_alignment': 1, 'evidence_location_score': 0}
        self.assertEqual(screen('feedbacktrace', {}, result)[0]['outcome'], 'full')


if __name__ == '__main__':
    unittest.main()
