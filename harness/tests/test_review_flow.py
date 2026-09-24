from __future__ import annotations

import unittest
from unittest.mock import patch

import yaml

from codex_harness import Harness
from codex_harness.application.storage import HarnessError
from codex_harness.hooks.lifecycle import handle_hook
from tests.support import ProjectTemporaryDirectory


class ReviewFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = ProjectTemporaryDirectory()
        self.root = self.temp.__enter__()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.code = self.repo / 'train.py'
        self.code.write_text('def train():\n    return 30\n')
        (self.repo / 'PLAN.md').write_text('Keep budget 30.')
        self.harness = Harness(self.root / 'state')
        handle_hook(self.harness, {'hook_event_name': 'UserPromptSubmit', 'session_id': 's',
                    'turn_id': 't', 'cwd': str(self.repo), 'prompt': 'Follow PLAN.md and run train.py.'})

    def tearDown(self):
        self.temp.__exit__(None, None, None)

    def test_review_does_not_fetch_code_and_evidence_uses_frozen_snapshot(self):
        with patch.object(self.harness, 'build_evidence_groups', side_effect=AssertionError('eager evidence')):
            review = yaml.safe_load(self.harness.checks.review(trigger='result', focus='Report training'))
        self.assertEqual(review['plans'][0]['content'], 'Keep budget 30.')
        self.assertIsNone(review['assessment'])
        self.assertNotIn('return 30', str(review))
        for key in ('code_evidence', 'related_materials', 'table_relations'):
            self.assertNotIn(key, review)
        self.code.write_text('def train():\n    return 900\n')
        self.harness = Harness(self.root / 'state')
        evidence = yaml.safe_load(self.harness.checks.evidence(review['check_id'], 'What budget does train.py use?'))
        self.assertIn('return 30', str(evidence))
        self.assertNotIn('return 900', str(evidence))
        reread = yaml.safe_load(self.harness.checks.evidence(review['check_id'], read_ref=review['read_ref']))
        self.assertEqual(reread, review)

    def test_all_review_and_evidence_pages_use_the_evidence_reader(self):
        review = yaml.safe_load(self.harness.checks.review(trigger='result', focus='Report training'))
        check_id = review['check_id']
        self.harness.token_budget = 256
        page = yaml.safe_load(self.harness.checks.evidence(check_id, read_ref=review['read_ref']))
        self.assertIn('next', page)
        next_page = yaml.safe_load(self.harness.checks.evidence(check_id, **page['next']))
        self.assertGreater(next_page['offset'], page['offset'])
        sources = self.harness.checks.evidence(check_id, read_ref=review['all_sources']['read_ref'])
        self.assertIn('Follow PLAN.md', sources)
        self.harness.token_budget = 12000
        other = yaml.safe_load(self.harness.checks.review(trigger='result', focus='Another result'))
        with self.assertRaisesRegex(HarnessError, 'different checkpoint'):
            self.harness.checks.evidence(other['check_id'], read_ref=review['read_ref'])

    def test_record_is_small_persistent_and_does_not_collect_evidence(self):
        review = yaml.safe_load(self.harness.checks.review(trigger='result', focus='Report training'))
        check_id = review['check_id']
        self.harness.checks.evidence(check_id, 'What budget does train.py use?')
        with patch.object(self.harness, 'build_evidence_groups', side_effect=AssertionError('evidence')):
            with patch('codex_harness.application.checks.capture', side_effect=AssertionError('recapture')):
                result = yaml.safe_load(self.harness.checks.record(check_id, 'clear', 'Observed behavior matches.'))
        self.assertEqual(result['check_id'], check_id)
        self.assertEqual(result['conclusion'], 'clear')
        self.assertTrue(result['recorded'])
        self.assertNotIn('prompts', result)
        self.assertNotIn('review_protocol', result)
        self.harness = Harness(self.root / 'state')
        self.assertEqual(self.harness.checks.get(check_id)['assessment']['summary'], 'Observed behavior matches.')
        with self.assertRaises(HarnessError):
            self.harness.checks.record(check_id, 'clear', '   ')
        with self.assertRaises(HarnessError):
            self.harness.checks.record(check_id, 'invalid', 'Invalid conclusion.')
