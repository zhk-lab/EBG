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

    def review(self, **args):
        return yaml.safe_load(self.harness.checks.review(**args))

    def evidence(self, **args):
        return yaml.safe_load(self.harness.checks.evidence(**args))

    def record(self, **args):
        self.harness.checks.evidence(args['check_id'], 'What budget does train.py use?')
        return yaml.safe_load(self.harness.checks.record(**args))

    def test_legacy_tasks_cannot_enter_checkpoint_evidence(self):
        review = self.review(trigger='result', focus='Check training budget')
        self.evidence(check_id=review['check_id'], question='What budget does train.py use?')
        task_id = 'check_' + review['check_id']
        for scope in ({'mode': 'current'}, {'start_prompt': 'P1', 'end_prompt': 'P1'}):
            with self.subTest(scope=scope):
                task = self.harness.store.task(task_id)
                task['scope'] = {'session_id': 's', **scope}
                self.harness.store.put_task(task)
                with self.assertRaisesRegex(HarnessError, 'requires a checkpoint'):
                    self.harness.build_evidence_groups(task_id)
                with self.assertRaisesRegex(HarnessError, 'requires a checkpoint'):
                    self.harness.refresh_task(task_id)

    def test_inflight_freezes_code_plans_and_trace_and_survives_restart(self):
        self.code.write_text('def train():\n    return 300\n', encoding='utf-8')
        self.hook('PostToolUse', tool_name='Bash', tool_use_id='run',
                  tool_input={'command': 'python train.py'}, tool_response={'exit_code': 0, 'accuracy': .91})
        ctx = self.review(trigger='result', focus='Adopt accuracy .91')
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
        self.harness.checks.evidence(ctx['check_id'], 'What does train.py return?')
        view = self.harness.store.latest_view('check_' + ctx['check_id'])
        self.assertNotIn('LATER', str(view['events']))
        reread = self.evidence(check_id=ctx['check_id'], read_ref=ctx['read_ref'])
        self.assertEqual(ctx, reread)

    def test_older_constraints_and_linked_call_are_available(self):
        self.hook('PostToolUse', tool_name='Bash', tool_use_id='old',
                  tool_input={'command': 'python train.py'}, tool_response='OLD RESULT')
        old = self.harness.sessions.events('s')[-1]['id']
        self.hook('Stop', last_assistant_message='Continue later')
        self.hook('UserPromptSubmit', turn_id='t2', prompt='Try another candidate.')
        self.hook('PostToolUse', turn_id='t2', tool_name='Read', tool_use_id='new',
                  tool_input={'path': 'notes.txt'}, tool_response='unrelated')
        ctx = self.review(trigger='result', focus='Use old result', event_ids=[old])
        self.assertEqual(len(ctx['prompts']), 2)
        self.assertEqual(len(ctx['trace']), 2)
        self.assertNotIn('unrelated', str(ctx['trace']))
        expanded = self.evidence(check_id=ctx['check_id'], read_ref=ctx['all_trace']['read_ref'])
        self.assertIn('unrelated', str(expanded))

    def test_assessment_not_inferred_from_read_and_invalidated_by_changes(self):
        ctx = self.review(trigger='result', focus='Adopt result')
        self.assertIsNone(ctx['assessment'])
        done = self.record(check_id=ctx['check_id'], conclusion='issue', summary='Budget changed; disclose before adopting.')
        reused = self.review(trigger='result', focus='Adopt result')
        self.assertEqual(done['check_id'], reused['check_id'])
        self.assertEqual(done['conclusion'], reused['assessment']['conclusion'])
        changed_claim = self.review(trigger='result', focus='All models improve')
        self.assertNotEqual(ctx['check_id'], changed_claim['check_id'])
        self.code.write_text('def train():\n    return 400\n', encoding='utf-8')
        changed_code = self.review(trigger='result', focus='Adopt result')
        self.assertIsNone(changed_code['assessment'])
        self.assertNotEqual(ctx['check_id'], changed_code['check_id'])

    def test_review_reread_does_not_recollect_or_query_code(self):
        ctx = self.review(trigger='result', focus='Deliver train.py with fixed budget.')
        self.assertNotIn('code_evidence', ctx)
        self.code.write_text('def train():\n    return 900\n', encoding='utf-8')
        with patch.object(self.harness, 'build_evidence_groups', side_effect=AssertionError('eager evidence')):
            with patch('codex_harness.application.checks.capture', side_effect=AssertionError('recaptured')):
                reread = self.review(check_id=ctx['check_id'])
        self.assertEqual(ctx, reread)

    def test_stop_continuation_is_bounded_and_not_a_user_requirement(self):
        stop = self.hook('Stop', last_assistant_message='Accuracy improved.')
        self.assertEqual(stop['decision'], 'block')
        check = self.harness.checks.all()[-1]
        self.review(check_id=check['id'])
        self.assertIsNone(self.harness.checks.get(check['id'])['assessment'])
        self.hook('UserPromptSubmit', turn_id='resume', prompt=stop['reason'])
        self.record(check_id=check['id'], conclusion='uncertain', summary='Need a comparable run.')
        ctx = self.review(trigger='result', focus='Report the limitation')
        self.assertEqual(len(ctx['prompts']), 1)
        end = self.hook('Stop', turn_id='resume', stop_hook_active=True,
                        last_assistant_message='Comparable validation is missing.')
        self.assertNotIn('decision', end)

    def test_question_without_refs_uses_checkpoint_sources(self):
        ctx = self.review(trigger='result', focus='Deliver training result')
        result = yaml.safe_load(self.harness.checks.evidence(
            ctx['check_id'], 'What budget does train.py actually use?'))
        self.assertIn('return 30', str(result))
        self.assertIn('Follow PLAN.md', str(result))

    def test_review_routes_stage_content_and_freezes_shared_rules(self):
        package = self.root / 'package'
        for name, marker in [('ebg-ambiguity', 'PROCESS ONLY'), ('ebg-adjustment', 'PROCESS ONLY'),
                             ('ebg-result-review', 'RESULT ONLY')]:
            folder = package / 'skills' / name
            folder.mkdir(parents=True)
            (folder / 'SKILL.md').write_text('## Autoresearch review\n' + marker, encoding='utf-8')
        shared = package / 'skills/ebg-review/references/review-rules.md'
        shared.parent.mkdir(parents=True)
        shared.write_text('SHARED RULES', encoding='utf-8')
        with patch('codex_harness.application.checks.resource_files', return_value=package):
            for trigger in ('ambiguity', 'adjustment', 'result'):
                with self.subTest(trigger=trigger):
                    ctx = self.review(trigger=trigger, focus='Review ' + trigger)
                    content = ctx['review_protocol']['content']
                    expected, excluded = (('RESULT ONLY', 'PROCESS ONLY') if trigger == 'result'
                                          else ('PROCESS ONLY', 'RESULT ONLY'))
                    self.assertIn(expected, content)
                    self.assertNotIn(excluded, content)
                    self.assertIn('SHARED RULES', content)
            shared.write_text('NEW SHARED RULES', encoding='utf-8')
            self.assertEqual(self.review(check_id=ctx['check_id'])['review_protocol'], ctx['review_protocol'])

    def test_context_delivers_skill_review_principles_and_freezes_them(self):
        expected = {'source': 'review fixture', 'content': 'Check the evidence for the proposed conclusion.',
                    'note': 'Review guidance, not a user requirement.'}
        with patch('codex_harness.application.checks._review_protocol', return_value=expected):
            ctx = self.review(trigger='result', focus='Report improvement')
        protocol = ctx['review_protocol']
        self.assertEqual(protocol, expected)
        with patch('codex_harness.application.checks._review_protocol', side_effect=AssertionError('reread Skill')):
            repeated = self.review(check_id=ctx['check_id'])
        self.assertEqual(protocol, repeated['review_protocol'])

    def test_review_guidance_does_not_add_a_requirement_or_verdict(self):
        plan = 'Compare the two methods. Hyperparameter search is allowed.'
        (self.repo / 'PLAN.md').write_text(plan, encoding='utf-8')
        for trigger in ('ambiguity', 'result'):
            with self.subTest(trigger=trigger):
                ctx = self.review(trigger=trigger, focus='Choose a search strategy for PLAN.md')
                self.assertTrue(ctx['review_protocol']['content'])
                self.assertEqual({item['content'] for item in ctx['plans']},
                                 {'Keep training budget fixed at 30.', plan})
                self.assertIsNone(ctx['assessment'])

    def test_repo_quote_can_anchor_followup_without_becoming_a_plan(self):
        ctx = self.review(trigger='result', focus='Deliver training result')
        self.harness.checks.evidence(ctx['check_id'], 'What does train.py return?')
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
        other = self.review(trigger='result', focus='A different result')
        with self.assertRaisesRegex(HarnessError, 'different checkpoint'):
            self.harness.checks.evidence(other['check_id'], 'Check budget', [ref])

    def test_large_data_is_only_returned_by_explicit_evidence_query(self):
        (self.repo / 'samples.csv').write_text('id,value\n' + 'sample,30\n' * 2000, encoding='utf-8')
        (self.repo / 'PLAN.md').write_text('Use samples.csv in train.py. Keep budget 30.', encoding='utf-8')
        ctx = self.review(trigger='result', focus='Deliver train.py result')
        self.assertNotIn('sample,30', str(ctx))
        self.assertNotIn('return 30', str(ctx))
        proof = self.evidence(check_id=ctx['check_id'], question='How does train.py use samples.csv?')
        self.assertIn('samples.csv', str(proof['linked_artifacts']))
        view = self.harness.store.latest_view('check_' + ctx['check_id'])
        code = self.evidence(check_id=ctx['check_id'], read_ref=view['view_id'] + ':repo:train.py')
        self.assertIn('return 30', str(code))

    def test_failure_signal_keeps_result_and_ebg_calls_do_not_recurse(self):
        result = self.hook('PostToolUse', tool_name='Bash', tool_use_id='fail',
                           tool_input={'command': 'python train.py'}, tool_response={'exit_code': 1})
        self.assertNotIn('decision', result)
        self.assertEqual(result, {})
        self.assertEqual(self.harness.checks.all(), [])
        ctx = self.review(trigger='adjustment', focus='Review failure at the end')
        self.assertIn('"exit_code": 1', str(ctx['trace']))
        count = len(self.harness.sessions.events('s'))
        self.hook('PostToolUse', tool_name='mcp__ebg_disclose__ebg_evidence', tool_use_id='self',
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
        ctx = self.review(trigger='result', focus='The measured result is ready.')
        self.record(check_id=ctx['check_id'], conclusion='clear', summary='The result matches the recorded conditions.')
        self.assertEqual(self.hook('Stop', last_assistant_message='The measured result is ready.'), {})
        self.assertEqual(len(self.harness.checks.all()), 1)
        self.assertEqual(self.harness.checks.get(ctx['check_id'])['assessment']['conclusion'], 'clear')

    def test_unread_material_and_old_evidence_pages_remain_frozen(self):
        ctx = self.review(trigger='result', focus='Verify train.py')
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
        ctx = self.review(trigger='ambiguity', focus='PLAN.md does not specify a seed; use 42?')
        self.assertEqual(ctx['trigger'], 'ambiguity')
        self.assertNotIn('42', str(ctx['prompts']) + str(ctx['plans']))
        self.assertIsNone(ctx['assessment'])

    def test_invalid_refs_and_cross_checkpoint_reads_are_rejected(self):
        ctx = self.review(trigger='adjustment', focus='Switch the optimizer')
        other = self.review(trigger='result', focus='Report improvement')
        with self.assertRaises(HarnessError):
            self.evidence(check_id=other['check_id'], read_ref=ctx['read_ref'])
        with self.assertRaises(HarnessError):
            self.harness.checks.evidence(ctx['check_id'], 'Did training change?',
                                         [{'source_id': 'P1', 'quote': 'Invented requirement'}])
        with self.assertRaises(HarnessError):
            self.review(trigger='result', focus='adopt', event_ids=['s99999'])

    def test_capture_failure_and_event_race_leave_no_partial_checkpoint(self):
        with patch('codex_harness.application.checks.capture', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):
                self.review(trigger='result', focus='adopt')
        self.assertEqual(self.harness.checks.all(), [])
        def racing(*args, **kwargs):
            result = capture(*args, **kwargs)
            self.hook('PreToolUse', tool_name='Read', tool_use_id='race', tool_input={})
            return result
        with patch('codex_harness.application.checks.capture', side_effect=racing):
            with self.assertRaisesRegex(HarnessError, 'Events changed'):
                self.review(trigger='result', focus='adopt')
        self.assertEqual(self.harness.checks.all(), [])
        self.harness = Harness(self.root / 'state')
        self.assertIn('check_id', self.review(trigger='result', focus='adopt'))

    def test_session_switch_clears_checks_and_references(self):
        ctx = self.review(trigger='result', focus='adopt')
        self.hook('UserPromptSubmit', session_id='new', prompt='New task')
        with self.assertRaises(HarnessError):
            self.review(check_id=ctx['check_id'])
        self.assertEqual(self.harness.checks.all(), [])

    def test_small_budget_can_expand_frozen_context(self):
        self.harness.token_budget = 256
        result = self.review(trigger='result', focus='adopt')
        check = self.harness.checks.all()[-1]
        self.assertIn('entries', result)
        self.assertEqual(result['entries'][0]['content'], check['id'])
        sources = self.evidence(check_id=check['id'], read_ref=result['read_ref'] + '#/prompts')
        self.assertIn('Follow PLAN.md', str(sources))
