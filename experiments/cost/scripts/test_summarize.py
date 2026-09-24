import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import summarize


class AcceptedTokensTests(unittest.TestCase):
    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding='utf-8')

    def test_excludes_rejected_calls_and_other_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.write(run / 'batch_result.json', {'status': 'complete', 'actual_usage': {'total_tokens': 999}})
            state = {'records': [{'usage': {'prompt_tokens': 20, 'total_tokens': 25,
                                          'prompt_tokens_details': {'cached_tokens': 12}}}],
                     'terminal_record': {'usage': {'input_tokens': 30, 'total_tokens': 40}}}
            with patch.object(summarize, 'load_stage1', return_value=({}, state, run / 'state.json')):
                self.assertEqual(summarize.accepted_tokens(run), 50)

    def test_one_shot_review_counts_only_accepted_correction(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.write(run / 'batch_result.json', {'status': 'complete'})
            self.write(run / 'state.json', {'status': 'complete'})
            for turn, tokens in [(1, 500), (2, 30)]:
                self.write(run / f'responses/turn_{turn:03d}.json',
                           {'usage': {'prompt_tokens': tokens, 'total_tokens': tokens + 10}})
            self.assertEqual(summarize.accepted_tokens(run, source_review=True), 30)

    def test_citation_repair_uses_recorded_accepted_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.write(run / 'batch_result.json', {
                'status': 'complete', 'usage_scope': 'successful_attempt_with_citation_repair',
                'usage': {'input_tokens': 50, 'total_tokens': 70}, 'actual_usage': {'input_tokens': 999}})
            self.assertEqual(summarize.accepted_tokens(run), 50)

    def test_failed_prediction_is_not_a_zero_token_success(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.write(run / 'batch_result.json', {'status': 'failed'})
            with self.assertRaisesRegex(ValueError, 'Incomplete prediction'):
                summarize.accepted_tokens(run)


if __name__ == '__main__':
    unittest.main()
