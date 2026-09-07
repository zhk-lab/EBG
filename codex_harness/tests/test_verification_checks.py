from __future__ import annotations

import unittest
import yaml

from codex_harness import Harness
from codex_harness.hooks.lifecycle import handle_hook
from tests.support import ProjectTemporaryDirectory


class VerificationChecksTests(unittest.TestCase):
    def test_failed_verification_is_a_new_result_not_an_assumed_plan_change(self):
        with ProjectTemporaryDirectory() as root:
            repo = root / 'repo'
            repo.mkdir()
            harness = Harness(root / 'state')
            base = {'session_id': 's', 'turn_id': 't', 'cwd': str(repo)}
            handle_hook(harness, {**base, 'hook_event_name': 'UserPromptSubmit',
                                 'prompt': 'Validate the candidate before reporting.'})
            payload = {**base, 'hook_event_name': 'PostToolUse', 'tool_name': 'Bash',
                       'tool_use_id': 'passed', 'tool_input': {'command': 'python -m unittest'},
                       'tool_response': {'exit_code': 0}}
            handle_hook(harness, payload)
            old = harness.checks.all()[-1]
            harness.checks.record(check_id=old['id'], conclusion='clear', summary='Initial tests passed.')
            failed = {**payload, 'tool_use_id': 'failed',
                      'tool_response': {'exit_code': 1, 'stdout': 'Numerical tolerance exceeded.'}}
            response = handle_hook(harness, failed)
            check = harness.checks.all()[-1]
            self.assertEqual(check['trigger'], 'result')
            self.assertEqual(check['verification_call_id'], '1:failed')
            self.assertIsNone(check['assessment'])
            self.assertNotIn('decision', response)
            current = yaml.safe_load(harness.checks.review(check_id=check['id']))
            self.assertIn('Numerical tolerance exceeded.', str(current['trace']))
            stale = yaml.safe_load(harness.checks.review(check_id=old['id']))
            self.assertFalse(stale['execution_scope']['covers_latest_verification'])
            self.assertEqual(stale['execution_scope']['latest_check_id'], check['id'])
            count = len(harness.checks.all())
            self.assertEqual(handle_hook(harness, failed), {})
            self.assertEqual(len(harness.checks.all()), count)

    def test_execution_triggers_before_report_and_returns_exact_frozen_receipt(self):
        with ProjectTemporaryDirectory() as root:
            repo = root / 'repo'
            (repo / 'receipts/new').mkdir(parents=True)
            (repo / 'receipts/old').mkdir()
            (repo / 'PLAN.md').write_text('Complete a live verification.')
            (repo / 'receipts/old/result.json').write_text('{"mode":"live"}')
            harness = Harness(root / 'state')
            base = {'session_id': 's', 'turn_id': 't', 'cwd': str(repo)}
            handle_hook(harness, {**base, 'hook_event_name': 'UserPromptSubmit', 'prompt': 'Follow PLAN.md.'})
            old = yaml.safe_load(harness.checks.review(trigger='result', focus='Baseline complete'))
            (repo / 'receipts/new/result.json').write_text('{"mode":"mock","live_requests":0}')
            payload = {**base, 'hook_event_name': 'PostToolUse', 'tool_name': 'Bash',
                       'tool_use_id': 'smoke', 'tool_input': {'command': 'python scripts/smoke.py'},
                       'tool_response': {'exit_code': 0, 'stdout': 'Record: receipts/new/result.json'}}
            response = handle_hook(harness, payload)
            self.assertIn('beg_review', str(response))
            check = harness.checks.all()[-1]
            self.assertEqual(check['trigger'], 'result')
            ctx = yaml.safe_load(harness.checks.review(check_id=check['id']))
            self.assertNotIn('related_materials', ctx)
            proof = yaml.safe_load(harness.checks.evidence(check['id'], 'Did the smoke test perform live requests?'))
            items = {item['path']: item for item in proof['linked_artifacts']['items']}
            self.assertIn('receipts/new/result.json', items)
            self.assertNotIn('receipts/old/result.json', items)
            (repo / 'receipts/new/result.json').write_text('{"mode":"live","live_requests":1}')
            again = yaml.safe_load(harness.checks.evidence(check['id'], read_ref=proof['read_ref']))
            self.assertEqual(proof, again)
            stale = yaml.safe_load(harness.checks.review(check_id=old['check_id']))
            self.assertFalse(stale['execution_scope']['covers_latest_verification'])
            self.assertEqual(stale['execution_scope']['latest_check_id'], check['id'])
            count = len(harness.checks.all())
            self.assertEqual(handle_hook(harness, payload), {})
            self.assertEqual(len(harness.checks.all()), count)

    def test_reading_or_echoing_a_verification_command_does_not_trigger(self):
        with ProjectTemporaryDirectory() as root:
            repo = root / 'repo'
            repo.mkdir()
            harness = Harness(root / 'state')
            base = {'session_id': 's', 'turn_id': 't', 'cwd': str(repo)}
            handle_hook(harness, {**base, 'hook_event_name': 'UserPromptSubmit', 'prompt': 'Inspect files.'})
            for number, command in enumerate(['Get-Content scripts/smoke.py',
                                               'echo "python scripts/smoke.py"', 'python ordinary.py']):
                result = handle_hook(harness, {**base, 'hook_event_name': 'PostToolUse',
                    'tool_name': 'Bash', 'tool_use_id': str(number),
                    'tool_input': {'command': command}, 'tool_response': {'exit_code': 0}})
                self.assertEqual(result, {})
            self.assertEqual(harness.checks.all(), [])

    def test_live_receipt_is_returned_without_an_automatic_issue_verdict(self):
        with ProjectTemporaryDirectory() as root:
            repo = root / 'repo'
            repo.mkdir()
            harness = Harness(root / 'state')
            base = {'session_id': 's', 'turn_id': 't', 'cwd': str(repo)}
            handle_hook(harness, {**base, 'hook_event_name': 'UserPromptSubmit',
                                 'prompt': 'Complete a live verification.'})
            (repo / 'receipt.json').write_text('{"mode":"live","live_requests":1,"response_ids":["test-response"]}')
            handle_hook(harness, {**base, 'hook_event_name': 'PostToolUse', 'tool_name': 'Bash',
                'tool_use_id': 'live', 'tool_input': {'command': 'python scripts/smoke.py'},
                'tool_response': {'exit_code': 0, 'stdout': 'receipt.json'}})
            check = harness.checks.all()[-1]
            context = yaml.safe_load(harness.checks.review(check_id=check['id']))
            self.assertIsNone(context['assessment'])
            proof = yaml.safe_load(harness.checks.evidence(check['id'], 'Was the live verification performed?'))
            self.assertIn('"mode":"live"', str(proof['linked_artifacts']))
            self.assertTrue(context['execution_scope']['covers_latest_verification'])


if __name__ == '__main__':
    unittest.main()
