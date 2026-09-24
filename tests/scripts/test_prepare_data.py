import json
from pathlib import Path
import unittest

from scripts.prepare_data import prepare_sample
from tests.support import ProjectTemporaryDirectory, make_repo_bundle, make_trace_bundle


class DataPreparationTests(unittest.TestCase):
    def test_gold_stays_separate_and_resume_keeps_inputs(self):
        with ProjectTemporaryDirectory() as root:
            source, destination = root / 'release', root / 'prepared'
            artifacts = source / 'specgap/artifacts'
            make_repo_bundle(artifacts / 'visible_bundles', input_id='sg_001',
                             repository_files={'main.py': 'answer = 42\n'})
            hidden = artifacts / 'hidden_gold'
            hidden.mkdir()
            gold = {'input_id': 'sg_001', 'benchmark': 'specgap', 'conditions': []}
            (hidden / 'sg_001.json').write_text(json.dumps(gold))
            self.assertEqual(prepare_sample(source, destination, 'specgap', 'sg_001'), 'prepared')
            bundle = destination / 'specgap/artifacts/visible_bundles/sg_001'
            self.assertFalse((bundle / 'gold.json').exists())
            with self.assertRaises(FileExistsError):
                prepare_sample(source, destination, 'specgap', 'sg_001')
            self.assertEqual(prepare_sample(source, destination, 'specgap', 'sg_001', resume=True), 'reused')
            marker = destination / 'specgap/artifacts/preparation/sg_001.json'
            marker.unlink()
            self.assertEqual(prepare_sample(source, destination, 'specgap', 'sg_001', resume=True), 'prepared')

    def test_rejects_feedbacktrace_outside_key_release(self):
        with ProjectTemporaryDirectory() as root:
            source = root / 'release'
            artifacts = source / 'feedbacktrace/artifacts'
            make_trace_bundle(artifacts / 'visible_bundles', [
                {'event_type': 'user_prompt', 'turn_number': 1, 'content': 'Inspect the result.'},
            ], input_id='ft_001_long')
            hidden = artifacts / 'hidden_gold'
            hidden.mkdir()
            (hidden / 'ft_001_long.json').write_text(json.dumps({
                'input_id': 'ft_001_long', 'benchmark': 'feedbacktrace', 'verdict': None,
            }))
            with self.assertRaisesRegex(ValueError, 'KEY Long'):
                prepare_sample(source, root / 'prepared', 'feedbacktrace', 'ft_001_long')


if __name__ == '__main__':
    unittest.main()
