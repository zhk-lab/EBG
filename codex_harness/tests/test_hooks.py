from __future__ import annotations

import unittest
from unittest.mock import patch

from codex_harness import Harness
from codex_harness.application.storage import HarnessError
from codex_harness.hooks.lifecycle import handle_hook
from codex_harness.application.repository import capture
from tests.support import ProjectTemporaryDirectory


class HookTests(unittest.TestCase):
    def setUp(self):
        self.temp = ProjectTemporaryDirectory()
        self.root = self.temp.__enter__()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.code = self.repo / 'app.py'
        self.code.write_text('VALUE = 1\n', encoding='utf-8', newline='')
        self.harness = Harness(self.root / 'state')
        self.base = {'session_id': 's', 'turn_id': 't1', 'cwd': str(self.repo)}

    def tearDown(self):
        self.temp.__exit__(None, None, None)

    def hook(self, name, **kwargs):
        return handle_hook(self.harness, {**self.base, 'hook_event_name': name, **kwargs})

    def test_records_session_without_creating_or_binding_a_task(self):
        result = self.hook('UserPromptSubmit', prompt='Change app.py')
        self.assertIn('session_id=s', result['hookSpecificOutput']['additionalContext'])
        self.assertEqual(self.harness.store.current_session(), 's')
        self.hook('UserPromptSubmit', prompt='Change app.py')
        self.code.write_text('VALUE = 2\n', encoding='utf-8', newline='')
        call = {'tool_use_id': 'c', 'tool_name': 'Edit', 'tool_input': {'path': 'app.py'}}
        self.hook('PreToolUse', **call)
        self.hook('PostToolUse', **call, tool_response='Updated')
        self.hook('PostToolUse', **call, tool_response='Updated')
        self.hook('Stop', last_assistant_message='app.py changed')
        self.hook('Stop', last_assistant_message='app.py changed')
        events = self.harness.sessions.events('s')
        self.assertEqual([e['kind'] for e in events], ['user', 'tool_call', 'tool_result', 'assistant'])
        row = self.harness.sessions.timeline('s')['turns'][0]
        for name, value in [('before', 1), ('after', 2)]:
            result = self.harness.sessions.material('s', snapshot_id=row[name], path='app.py')
            self.assertEqual(result['content'], f'VALUE = {value}\n')
        self.assertIn('session_id=s', self.hook('SessionStart')['hookSpecificOutput']['additionalContext'])

    def test_each_prompt_has_a_before_snapshot_and_preserves_followup(self):
        self.hook('UserPromptSubmit', prompt='Implement login')
        self.hook('Stop', last_assistant_message='Done')
        self.code.write_text('VALUE = 7\n', encoding='utf-8', newline='')
        self.hook('UserPromptSubmit', turn_id='t2', prompt='Now implement payments')
        rows = self.harness.sessions.timeline('s')['turns']
        self.assertEqual(len(rows), 2)
        self.assertNotIn('after', rows[1])
        self.assertEqual(self.harness.sessions.material('s', snapshot_id=rows[1]['before'], path='app.py')['content'], 'VALUE = 7\n')
        self.assertEqual([e['content'] for e in self.harness.sessions.events('s') if e['kind'] == 'user'], ['Implement login', 'Now implement payments'])

    def test_failed_start_can_resume_without_partial_prompt_or_duplicate_snapshot(self):
        with patch('codex_harness.application.sessions.capture', side_effect=RuntimeError('interrupted')):
            with self.assertRaises(RuntimeError):
                self.hook('UserPromptSubmit', prompt='Original')
        self.assertEqual(self.harness.sessions.events('s'), [])
        self.harness = Harness(self.root / 'state')
        self.hook('UserPromptSubmit', prompt='Original')
        row = self.harness.sessions.timeline('s')['turns'][0]
        self.code.write_text('VALUE = 99\n', encoding='utf-8', newline='')
        self.hook('UserPromptSubmit', prompt='Original')
        self.assertEqual(self.harness.sessions.material('s', snapshot_id=row['before'], path='app.py')['content'], 'VALUE = 1\n')
        with self.assertRaises(HarnessError):
            self.hook('UserPromptSubmit', prompt='Different')

    def test_stop_failure_resumes_and_unchanged_files_reuse_storage(self):
        self.hook('UserPromptSubmit', prompt='Check app.py')
        with patch('codex_harness.application.sessions.capture', side_effect=RuntimeError('interrupted')):
            with self.assertRaises(RuntimeError):
                self.hook('Stop', last_assistant_message='Done')
        self.harness = Harness(self.root / 'state')
        self.hook('Stop', last_assistant_message='Done')
        row = self.harness.sessions.timeline('s')['turns'][0]
        before = self.harness.sessions.snapshot('s', row['before'])
        after = self.harness.sessions.snapshot('s', row['after'])
        self.assertEqual(before['files'], after['files'])
        self.assertEqual(len(self.harness.sessions.events('s')), 2)

    def test_tool_result_replay_recovers_after_call_was_saved(self):
        self.hook('UserPromptSubmit', prompt='Change app.py')
        call = {'tool_use_id': 'c', 'tool_name': 'Edit', 'tool_input': {'path': 'app.py'}}
        self.hook('PreToolUse', **call)
        self.harness = Harness(self.root / 'state')
        self.hook('PostToolUse', **call, tool_response='Updated')
        self.assertEqual(len(self.harness.sessions.events('s')), 3)
        self.hook('PostToolUse', **{**call, 'tool_name': 'mcp__beg_disclose__beg_build_evidence_groups'}, tool_response='Evidence')
        self.assertEqual(len(self.harness.sessions.events('s')), 3)

    def test_sessions_and_reused_call_ids_stay_separate(self):
        for turn in ('t1', 't2'):
            self.hook('UserPromptSubmit', turn_id=turn, prompt='Check app.py')
            self.hook('PostToolUse', turn_id=turn, tool_use_id='same', tool_name='Read', tool_input={'path': 'app.py'}, tool_response=turn)
            self.hook('Stop', turn_id=turn, last_assistant_message='Done')
        calls = [e['call_id'] for e in self.harness.sessions.events('s') if e['kind'] == 'tool_call']
        self.assertEqual(len(set(calls)), 2)
        self.hook('UserPromptSubmit', session_id='other', prompt='Different task')
        self.assertEqual(len(self.harness.sessions.events('other')), 1)

    def test_late_stop_cannot_invent_an_old_repo_snapshot(self):
        self.hook('UserPromptSubmit', prompt='Task A')
        self.hook('UserPromptSubmit', turn_id='t2', prompt='Task B')
        with self.assertRaisesRegex(HarnessError, 'old turn'):
            self.hook('Stop', last_assistant_message='Done A')
        self.assertNotIn('after', self.harness.sessions.timeline('s')['turns'][0])

    def test_events_arriving_during_snapshot_do_not_get_an_inconsistent_cutoff(self):
        self.hook('UserPromptSubmit', prompt='Task A')
        def racing_capture(*args, **kwargs):
            result = capture(*args, **kwargs)
            self.hook('PreToolUse', tool_use_id='racing', tool_name='Read', tool_input={'path': 'app.py'})
            return result
        with patch('codex_harness.application.sessions.capture', side_effect=racing_capture):
            with self.assertRaisesRegex(HarnessError, 'Events changed'):
                self.hook('Stop', last_assistant_message='Done')
        self.assertNotIn('after', self.harness.sessions.timeline('s')['turns'][0])
        self.hook('Stop', last_assistant_message='Done')
        self.assertEqual([e['kind'] for e in self.harness.sessions.events('s')], ['user', 'tool_call', 'assistant'])
