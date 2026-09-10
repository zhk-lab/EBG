from __future__ import annotations

import unittest
from unittest.mock import patch
import yaml

from codex_harness import Harness
from codex_harness.application.runtime import runtime
from codex_harness.application.storage import HarnessError
from codex_harness.hooks.lifecycle import handle_hook, _execution_plan, _read_plan
from tests.support import ProjectTemporaryDirectory


class SessionReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = ProjectTemporaryDirectory()
        self.root = self.temp.__enter__()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        (self.repo / 'PLAN.md').write_text('Keep evaluation split fixed.', encoding='utf-8')
        (self.repo / 'train.py').write_text('def train():\n    return 30\n', encoding='utf-8')
        self.h = Harness(self.root / 'state', review_call_threshold=2)

    def tearDown(self):
        self.temp.__exit__(None, None, None)

    def hook(self, event, **fields):
        return handle_hook(self.h, dict(hook_event_name=event, session_id='s', turn_id='t',
                                        cwd=str(self.repo), **fields))

    def call(self, identifier, code=0):
        return self.hook('PostToolUse', tool_use_id=identifier, tool_name='Bash',
                         tool_input={'command': 'python train.py'}, tool_response={'exit_code': code})

    def assess(self, identifier, **kwargs):
        self.h.checks.evidence(identifier, 'What does train.py implement?')
        return self.h.checks.record(identifier, kwargs.pop('conclusion', 'clear'),
                                    kwargs.pop('summary', 'Checked the implementation.'), **kwargs)

    def test_simple_prompt_and_plan_discussion_do_not_review(self):
        self.hook('UserPromptSubmit', prompt='Explain PLAN.md; do not execute it.')
        self.hook('PostToolUse', tool_use_id='read', tool_name='Read',
                  tool_input={'path': 'PLAN.md'}, tool_response='Keep split fixed.')
        self.assertEqual(self.h.checks.all(), [])
        self.assertEqual(self.hook('Stop', last_assistant_message='Explanation'), {})

    def test_explaining_execution_is_not_execution_and_read_path_is_exact(self):
        self.assertIsNone(_execution_plan('Explain how to execute PLAN.md.'))
        self.assertIsNone(_execution_plan('请解释如何按照 Plan 执行实验。'))
        self.assertFalse(_read_plan('Read', {'path': 'NOTPLAN.md'}, 'PLAN.md'))
        self.assertTrue(_read_plan('Read', {'path': '/repo/PLAN.md'}, 'PLAN.md'))

    def test_plan_read_checks_once_and_modified_plan_checks_again(self):
        self.hook('UserPromptSubmit', prompt='Follow PLAN.md and run train.py.')
        self.assertEqual(self.h.checks.all(), [])
        call = dict(tool_use_id='read', tool_name='Read', tool_input={'path': 'PLAN.md'}, tool_response='Plan')
        self.assertIn('beg_review', str(self.hook('PostToolUse', **call)))
        self.assertEqual(self.hook('PostToolUse', **call), {})
        self.assertEqual([c['trigger'] for c in self.h.checks.all()], ['ambiguity'])
        (self.repo / 'PLAN.md').write_text('Use a different evaluation split.', encoding='utf-8')
        self.hook('PostToolUse', **{**call, 'tool_use_id': 'read2'})
        self.assertEqual(len(self.h.checks.all()), 2)

    def test_failure_deferred_and_replayed_events_not_counted_twice(self):
        self.hook('UserPromptSubmit', prompt='Run train.py.')
        self.assertEqual(self.call('a', 1), {})
        self.assertEqual(self.call('a', 1), {})
        self.assertEqual(self.h.checks.all(), [])
        with runtime(self.h.store, 's') as state:
            self.assertEqual(len(state['calls']), 1)
        stop = self.hook('Stop', last_assistant_message='Failed')
        self.assertEqual(stop['decision'], 'block')
        self.assertEqual([c['trigger'] for c in self.h.checks.all()], ['adjustment', 'result'])
        self.assertEqual(self.hook('Stop', last_assistant_message='Failed'), stop)
        self.assertEqual(len(self.h.checks.all()), 2)

    def test_count_accumulates_across_prompts_and_restart(self):
        self.hook('UserPromptSubmit', prompt='Read train.py.')
        self.call('a')
        self.assertEqual(self.hook('Stop', last_assistant_message='One call'), {})
        self.h = Harness(self.root / 'state', review_call_threshold=2)
        base = dict(session_id='s', turn_id='t2', cwd=str(self.repo))
        handle_hook(self.h, dict(base, hook_event_name='UserPromptSubmit', prompt='Continue.'))
        handle_hook(self.h, dict(base, hook_event_name='PostToolUse', tool_use_id='a', tool_name='Read',
                                tool_input={'path': 'train.py'}, tool_response='Code'))
        self.assertEqual(handle_hook(self.h, dict(base, hook_event_name='Stop', last_assistant_message='Done'))['decision'], 'block')
        handle_hook(self.h, {'session_id': 'new', 'hook_event_name': 'SessionStart'})
        with runtime(self.h.store, 'new') as state:
            self.assertEqual(state['calls'], [])

    def test_waiting_bypasses_stop_and_requires_real_answer(self):
        self.hook('UserPromptSubmit', prompt='Execute the plan below: optimize train.py.')
        check = self.h.checks.all()[0]
        self.assess(check['id'], conclusion='uncertain', summary='Which test split?', waiting_for_user=True)
        self.assertEqual(self.hook('Stop', last_assistant_message='Which test split?'), {})
        with self.assertRaisesRegex(HarnessError, 'new user reply'):
            self.h.checks.record(check['id'], 'clear', 'Fixed.', resolution='Use held-out split.')
        base = dict(session_id='s', turn_id='answer', cwd=str(self.repo))
        handle_hook(self.h, dict(base, hook_event_name='UserPromptSubmit', prompt='Use the held-out split.'))
        with runtime(self.h.store, 's') as state:
            self.assertIsNotNone(state['waiting'])
        self.h.checks.record(check['id'], 'clear', 'User specified split.', resolution='Use held-out split.')
        with runtime(self.h.store, 's') as state:
            self.assertIsNone(state['waiting'])

    def test_evidence_required_even_for_clear_and_paging_is_not_enough(self):
        self.hook('UserPromptSubmit', prompt='Inspect train.py.')
        check = self.h.checks.create('result', 'Report train.py')
        review = yaml.safe_load(self.h.checks.review(check_id=check['id']))
        self.h.checks.evidence(check['id'], read_ref=review['read_ref'])
        with self.assertRaisesRegex(HarnessError, 'beg_evidence'):
            self.h.checks.record(check['id'], 'clear', 'Looks good.')
        with patch.object(self.h, 'build_evidence_groups', side_effect=HarnessError('Unavailable')):
            with self.assertRaisesRegex(HarnessError, 'Unavailable'):
                self.h.checks.evidence(check['id'], 'What does train.py implement?')
        self.assertNotIn('evidence_queries', self.h.checks.get(check['id']))
        self.assess(check['id'])

    def test_process_note_defers_review_and_is_in_frozen_inputs(self):
        self.hook('UserPromptSubmit', prompt='Inspect train.py.')
        self.h.checks.record(note_kind='adjustment', decision_status='executed',
                             summary='Reduced sample count after timeout.')
        self.assertEqual(self.h.checks.all(), [])
        self.assertEqual(self.hook('Stop', last_assistant_message='Partial result')['decision'], 'block')
        for check in self.h.checks.all():
            self.assertEqual(check['observations'][0]['summary'], 'Reduced sample count after timeout.')

    def test_time_threshold_alone_triggers_end_review(self):
        with patch('codex_harness.application.runtime.time.time', return_value=100):
            self.hook('UserPromptSubmit', prompt='Think through the experiment.')
        with patch('codex_harness.application.runtime.time.time', return_value=401):
            self.assertEqual(self.hook('Stop', last_assistant_message='Done')['decision'], 'block')

    def test_shell_read_and_inline_plan_trigger_ambiguity(self):
        self.hook('UserPromptSubmit', prompt='按照 PLAN.md 执行实验。')
        result = self.hook('PostToolUse', tool_name='exec_command', tool_use_id='read',
                           tool_input={'cmd': 'Get-Content -Encoding UTF8 PLAN.md'}, tool_response='Plan text')
        self.assertIn('beg_review', str(result))
        self.assertEqual(self.h.checks.all()[0]['trigger'], 'ambiguity')

    def test_changed_code_in_continuation_is_not_reused(self):
        self.hook('UserPromptSubmit', prompt='Run train.py.')
        self.call('a', 1)
        stop = self.hook('Stop', last_assistant_message='Failed')
        for check in self.h.checks.all():
            self.assess(check['id'], conclusion='issue')
        base = dict(session_id='s', turn_id='resume', cwd=str(self.repo))
        handle_hook(self.h, dict(base, hook_event_name='UserPromptSubmit', prompt=stop['reason']))
        (self.repo / 'train.py').write_text('def train():\n    return 40\n', encoding='utf-8')
        handle_hook(self.h, dict(base, hook_event_name='Stop', last_assistant_message='Changed'))
        self.assertEqual(len(self.h.checks.all()), 4)
        self.assertIsNone(self.h.checks.all()[-1]['assessment'])

    def test_elapsed_excludes_time_between_turns(self):
        with patch('codex_harness.application.runtime.time.time', return_value=100):
            self.hook('UserPromptSubmit', prompt='Inspect.')
        with patch('codex_harness.application.runtime.time.time', return_value=120):
            self.hook('Stop', last_assistant_message='Done')
        with patch('codex_harness.application.runtime.time.time', return_value=10000):
            handle_hook(self.h, dict(hook_event_name='UserPromptSubmit', session_id='s', turn_id='t2',
                                    cwd=str(self.repo), prompt='Continue.'))
        with runtime(self.h.store, 's') as state:
            self.assertEqual(state['elapsed'], 20)

    def test_completed_batch_reused_in_internal_continuation(self):
        self.hook('UserPromptSubmit', prompt='Run train.py.')
        self.call('a', 1)
        stop = self.hook('Stop', last_assistant_message='Failed')
        for check in self.h.checks.all():
            self.assess(check['id'], conclusion='issue')
        base = dict(session_id='s', turn_id='resume', cwd=str(self.repo))
        handle_hook(self.h, dict(base, hook_event_name='UserPromptSubmit', prompt=stop['reason']))
        self.assertEqual(handle_hook(self.h, dict(base, hook_event_name='Stop', last_assistant_message='Limited.')), {})
        self.assertEqual(len(self.h.checks.all()), 2)

    def test_completed_batch_reused_for_same_turn_stop_continuation(self):
        self.hook('UserPromptSubmit', prompt='Run train.py.')
        self.call('a', 1)
        self.hook('Stop', last_assistant_message='Failed')
        for check in self.h.checks.all():
            self.assess(check['id'], conclusion='issue')
        self.assertEqual(self.hook('Stop', stop_hook_active=True,
                                   last_assistant_message='Failed; here is the verified limitation.'), {})
        self.assertEqual(len(self.h.checks.all()), 2)

    def test_blocked_stop_keeps_turn_open_for_followup_tools(self):
        self.hook('UserPromptSubmit', prompt='Run train.py.')
        self.call('a', 1)
        self.assertEqual(self.hook('Stop', last_assistant_message='Failed')['decision'], 'block')
        self.call('verification', 0)
        self.assertTrue(any(e.get('original_call_id') == 'verification'
                            for e in self.h.sessions.events('s')))
