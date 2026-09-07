from __future__ import annotations

import unittest
from unittest.mock import patch

import yaml

from codex_harness import Harness
from codex_harness.application.repository import capture
from codex_harness.application.storage import HarnessError
from codex_harness.hooks.lifecycle import handle_hook
from tests.support import ProjectTemporaryDirectory


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = ProjectTemporaryDirectory()
        self.root = self.temp.__enter__()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.code = self.repo / 'train.py'
        self.code.write_text('def train():\n    return 30\n', encoding='utf-8')
        (self.repo / 'PLAN.md').write_text('Keep training budget fixed at 30.', encoding='utf-8')
        self.harness = Harness(self.root / 'state')
        self.hook('UserPromptSubmit', prompt='Follow PLAN.md and improve train.py.')

    def tearDown(self):
        self.temp.__exit__(None, None, None)

    def hook(self, name, **args):
        return handle_hook(self.harness, {'session_id': 's', 'turn_id': 't1', 'cwd': str(self.repo),
                                          'hook_event_name': name, **args})

    def context(self, **args):
        return yaml.safe_load(self.harness.checks.context(**args))

    def test_inflight_freezes_code_plans_and_trace_and_survives_restart(self):
        self.code.write_text('def train():\n    return 300\n', encoding='utf-8')
        self.hook('PostToolUse', tool_name='Bash', tool_use_id='run',
                  tool_input={'command': 'python train.py'}, tool_response={'exit_code': 0, 'accuracy': .91})
        ctx = self.context(trigger='result', focus='Adopt accuracy .91')
        self.assertEqual(ctx['plans'][0]['content'], 'Keep training budget fixed at 30.')
        self.assertEqual({e['kind'] for e in ctx['trace']}, {'tool_call', 'tool_result'})
        self.assertEqual(ctx['prompts'][0]['id'], 'P1')
        self.assertNotIn('after', self.harness.sessions.timeline('s')['turns'][0])
        call = next(e for e in ctx['trace'] if e['kind'] == 'tool_call')
        self.code.write_text('def train():\n    return 999\n', encoding='utf-8')
        (self.repo / 'PLAN.md').write_text('NEW PLAN', encoding='utf-8')
        self.hook('PostToolUse', tool_name='Bash', tool_use_id='later',
                  tool_input={'command': 'echo later'}, tool_response='LATER')
        self.harness = Harness(self.root / 'state')
        evidence = yaml.safe_load(self.harness.checks.evidence(ctx['check_id'], 'What does train.py train return?',
            [{'source_id': call['id'], 'quote': call['content']}]))
        self.assertIn('questions', evidence)
        self.assertNotIn('requirements', evidence)
        self.assertIn('trace', evidence['evidence_groups']['R1']['cited_sources'])
        self.assertIn('return 300', str(evidence))
        self.assertNotIn('return 999', str(evidence))
        view = self.harness.store.latest_view('check_' + ctx['check_id'])
        self.assertNotIn('LATER', str(view['events']))
        reread = self.context(check_id=ctx['check_id'], read_ref=ctx['read_ref'])
        self.assertEqual(ctx, reread)

    def test_older_constraints_and_linked_call_are_available(self):
        self.hook('PostToolUse', tool_name='Bash', tool_use_id='old',
                  tool_input={'command': 'python train.py'}, tool_response='OLD RESULT')
        old = self.harness.sessions.events('s')[-1]['id']
        self.hook('Stop', last_assistant_message='Continue later')
        self.hook('UserPromptSubmit', turn_id='t2', prompt='Try another candidate.')
        self.hook('PostToolUse', turn_id='t2', tool_name='Read', tool_use_id='new',
                  tool_input={'path': 'notes.txt'}, tool_response='unrelated')
        ctx = self.context(trigger='result', focus='Use old result', event_ids=[old])
        self.assertEqual(len(ctx['prompts']), 2)
        self.assertEqual(len(ctx['trace']), 2)
        self.assertNotIn('unrelated', str(ctx['trace']))
        expanded = self.context(check_id=ctx['check_id'], read_ref=ctx['all_trace']['read_ref'])
        self.assertIn('unrelated', str(expanded))

    def test_assessment_not_inferred_from_read_and_invalidated_by_changes(self):
        ctx = self.context(trigger='result', focus='Adopt result')
        self.assertIsNone(ctx['assessment'])
        done = self.context(check_id=ctx['check_id'], conclusion='issue', summary='Budget changed; disclose before adopting.')
        reused = self.context(trigger='result', focus='Adopt result')
        self.assertEqual(done['check_id'], reused['check_id'])
        self.assertEqual(done['assessment'], reused['assessment'])
        changed_claim = self.context(trigger='result', focus='All models improve')
        self.assertNotEqual(ctx['check_id'], changed_claim['check_id'])
        self.code.write_text('def train():\n    return 400\n', encoding='utf-8')
        changed_code = self.context(trigger='result', focus='Adopt result')
        self.assertIsNone(changed_code['assessment'])
        self.assertNotEqual(ctx['check_id'], changed_code['check_id'])

    def test_first_context_includes_code_and_reuses_its_frozen_evidence(self):
        ctx = self.context(trigger='result', focus='Deliver train.py with fixed budget.')
        self.assertIn('return 30', str(ctx['code_evidence']))
        self.assertIn('read_ref', ctx['evidence_expansion'])
        self.code.write_text('def train():\n    return 900\n', encoding='utf-8')
        with patch.object(self.harness, 'build_evidence_groups', side_effect=AssertionError('rebuilt')):
            with patch('codex_harness.application.checks.capture', side_effect=AssertionError('recaptured')):
                reread = self.context(check_id=ctx['check_id'])
        self.assertEqual(ctx['code_evidence'], reread['code_evidence'])
        self.assertNotIn('return 900', str(reread['code_evidence']))

    def test_stop_continuation_is_bounded_and_not_a_user_requirement(self):
        stop = self.hook('Stop', last_assistant_message='Accuracy improved.')
        self.assertEqual(stop['decision'], 'block')
        check = self.harness.checks.all()[-1]
        self.context(check_id=check['id'])
        self.assertIsNone(self.harness.checks.get(check['id'])['assessment'])
        self.hook('UserPromptSubmit', turn_id='resume', prompt=stop['reason'])
        self.context(check_id=check['id'], conclusion='uncertain', summary='Need a comparable run.')
        ctx = self.context(trigger='result', focus='Report the limitation')
        self.assertEqual(len(ctx['prompts']), 1)
        end = self.hook('Stop', turn_id='resume', stop_hook_active=True,
                        last_assistant_message='Comparable validation is missing.')
        self.assertNotIn('decision', end)

    def test_question_without_refs_uses_checkpoint_sources(self):
        ctx = self.context(trigger='result', focus='Deliver training result')
        result = yaml.safe_load(self.harness.checks.evidence(
            ctx['check_id'], 'What budget does train.py actually use?'))
        self.assertIn('return 30', str(result))
        self.assertIn('Follow PLAN.md', str(result))

    def test_context_delivers_skill_review_principles_and_freezes_them(self):
        expected = {'source': 'review fixture', 'content': 'Check the evidence for the proposed conclusion.',
                    'note': 'Review guidance, not a user requirement.'}
        with patch('codex_harness.application.checks._review_protocol', return_value=expected):
            ctx = self.context(trigger='result', focus='Report improvement')
        protocol = ctx['review_protocol']
        self.assertEqual(protocol, expected)
        with patch('codex_harness.application.checks._review_protocol', side_effect=AssertionError('reread Skill')):
            repeated = self.context(check_id=ctx['check_id'])
        self.assertEqual(protocol, repeated['review_protocol'])

    def test_review_guidance_does_not_add_a_requirement_or_verdict(self):
        plan = 'Compare the two methods. Hyperparameter search is allowed.'
        (self.repo / 'PLAN.md').write_text(plan, encoding='utf-8')
        for trigger in ('ambiguity', 'result'):
            with self.subTest(trigger=trigger):
                ctx = self.context(trigger=trigger, focus='Choose a search strategy for PLAN.md')
                self.assertTrue(ctx['review_protocol']['content'])
                self.assertEqual({item['content'] for item in ctx['plans']},
                                 {'Keep training budget fixed at 30.', plan})
                self.assertIsNone(ctx['assessment'])

    def test_repo_quote_can_anchor_followup_without_becoming_a_plan(self):
        ctx = self.context(trigger='result', focus='Deliver training result')
        view = self.harness.store.latest_view('check_' + ctx['check_id'])
        self.code.write_text('def train():\n    return 900\n', encoding='utf-8')
        ref = {'source_id': view['view_id'] + ':repo:train.py', 'quote': 'return 30'}
        result = yaml.safe_load(self.harness.checks.evidence(
            ctx['check_id'], 'What does train.py actually return?', [ref]))
        cited = result['evidence_groups']['R1']['cited_sources']
        self.assertIn('repo', cited)
        self.assertNotIn('plan', cited)
        self.assertNotIn('return 900', str(result))
        # A later question must still accept the original returned reference.
        self.harness.checks.evidence(ctx['check_id'], 'Is train.py using 30?', [ref])
        other = self.context(trigger='result', focus='A different result')
        with self.assertRaisesRegex(HarnessError, 'different checkpoint'):
            self.harness.checks.evidence(other['check_id'], 'Check budget', [ref])

    def test_large_data_does_not_fold_small_default_code_into_a_link(self):
        (self.repo / 'samples.csv').write_text('id,value\n' + 'sample,30\n' * 2000, encoding='utf-8')
        (self.repo / 'PLAN.md').write_text('Use samples.csv in train.py. Keep budget 30.', encoding='utf-8')
        self.harness.token_budget = 3000
        ctx = self.context(trigger='result', focus='Deliver train.py result')
        if 'entries' in ctx:
            code = next(e for e in ctx['entries'] if e['key'] == 'code_evidence')['content']
        else:
            code = ctx['code_evidence']
        self.assertIn('return 30', str(code))
        self.assertNotIn('sample,30', str(code))
        check = self.harness.checks.all()[-1]
        self.assertIn('samples.csv', str(check['initial_evidence']['materials']))

    def test_failure_signal_keeps_result_and_beg_calls_do_not_recurse(self):
        result = self.hook('PostToolUse', tool_name='Bash', tool_use_id='fail',
                           tool_input={'command': 'python train.py'}, tool_response={'exit_code': 1})
        self.assertNotIn('decision', result)
        self.assertIn('beg_context', result['hookSpecificOutput']['additionalContext'])
        ctx = self.context(check_id=self.harness.checks.all()[-1]['id'])
        self.assertIn('"exit_code": 1', str(ctx['trace']))
        count = len(self.harness.sessions.events('s'))
        self.hook('PostToolUse', tool_name='mcp__beg_disclose__beg_evidence', tool_use_id='self',
                  tool_input={}, tool_response={'isError': True})
        self.assertEqual(count, len(self.harness.sessions.events('s')))
        self.assertEqual(self.hook('PostToolUse', tool_name='Read', tool_use_id='text',
            tool_input={}, tool_response='The documentation says tests failed.'), {})
        checkpoints = len(self.harness.checks.all())
        replay = self.hook('PostToolUse', tool_name='Bash', tool_use_id='fail',
                           tool_input={'command': 'python train.py'}, tool_response={'exit_code': 1})
        self.assertEqual(replay, {})
        self.assertEqual(checkpoints, len(self.harness.checks.all()))

    def test_adoption_assessment_can_cover_identical_final_report(self):
        ctx = self.context(trigger='result', focus='The measured result is ready.')
        self.context(check_id=ctx['check_id'], conclusion='clear', summary='The result matches the recorded conditions.')
        self.assertEqual(self.hook('Stop', last_assistant_message='The measured result is ready.'), {})
        self.assertEqual(len(self.harness.checks.all()), 1)

    def test_unread_material_and_old_evidence_pages_remain_frozen(self):
        ctx = self.context(trigger='result', focus='Verify train.py')
        refs = [{'source_id': 'P1', 'quote': 'Follow PLAN.md and improve train.py.'}]
        first = yaml.safe_load(self.harness.checks.evidence(ctx['check_id'], 'What does train.py return?', refs))
        self.harness.checks.evidence(ctx['check_id'], 'Does train.py define an optimizer?', refs)
        reread = yaml.safe_load(self.harness.checks.evidence(ctx['check_id'], read_ref=first['read_ref']))
        self.assertEqual(first, reread)
        self.assertIsNone(self.harness.checks.get(ctx['check_id'])['assessment'])
        self.harness.token_budget = 256
        paged = yaml.safe_load(self.harness.checks.evidence(ctx['check_id'], 'What does train.py return?', refs))
        self.assertIn('read_ref', paged)

    def test_ambiguity_returns_plan_without_inventing_a_requirement(self):
        ctx = self.context(trigger='ambiguity', focus='PLAN.md does not specify a seed; use 42?')
        self.assertEqual(ctx['trigger'], 'ambiguity')
        self.assertNotIn('42', str(ctx['prompts']) + str(ctx['plans']))
        self.assertIsNone(ctx['assessment'])

    def test_invalid_refs_and_cross_checkpoint_reads_are_rejected(self):
        ctx = self.context(trigger='adjustment', focus='Switch the optimizer')
        other = self.context(trigger='result', focus='Report improvement')
        with self.assertRaises(HarnessError):
            self.context(check_id=other['check_id'], read_ref=ctx['read_ref'])
        with self.assertRaises(HarnessError):
            self.harness.checks.evidence(ctx['check_id'], 'Did training change?',
                                         [{'source_id': 'P1', 'quote': 'Invented requirement'}])
        with self.assertRaises(HarnessError):
            self.context(trigger='result', focus='adopt', event_ids=['s99999'])

    def test_capture_failure_and_event_race_leave_no_partial_checkpoint(self):
        with patch('codex_harness.application.checks.capture', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):
                self.context(trigger='result', focus='adopt')
        self.assertEqual(self.harness.checks.all(), [])
        def racing(*args, **kwargs):
            result = capture(*args, **kwargs)
            self.hook('PreToolUse', tool_name='Read', tool_use_id='race', tool_input={})
            return result
        with patch('codex_harness.application.checks.capture', side_effect=racing):
            with self.assertRaisesRegex(HarnessError, 'Events changed'):
                self.context(trigger='result', focus='adopt')
        self.assertEqual(self.harness.checks.all(), [])
        self.harness = Harness(self.root / 'state')
        self.assertIn('check_id', self.context(trigger='result', focus='adopt'))

    def test_session_switch_clears_checks_and_references(self):
        ctx = self.context(trigger='result', focus='adopt')
        self.hook('UserPromptSubmit', session_id='new', prompt='New task')
        with self.assertRaises(HarnessError):
            self.context(check_id=ctx['check_id'])
        self.assertEqual(self.harness.checks.all(), [])

    def test_small_budget_can_expand_frozen_context(self):
        self.harness.token_budget = 256
        result = self.context(trigger='result', focus='adopt')
        check = self.harness.checks.all()[-1]
        self.assertIn('entries', result)
        self.assertEqual(result['entries'][0]['content'], check['id'])
        sources = self.context(check_id=check['id'], read_ref=result['read_ref'] + '#/prompts')
        self.assertIn('Follow PLAN.md', str(sources))
