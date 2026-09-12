from __future__ import annotations

import unittest
from unittest.mock import patch

import yaml

from codex_harness import Harness
from codex_harness.application.storage import HarnessError
from codex_harness.hooks.lifecycle import handle_hook
from tests.support import ProjectTemporaryDirectory, checkpoint_task


class CheckpointSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = ProjectTemporaryDirectory()
        self.root = self.temp.__enter__()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.code = self.repo / 'app.py'
        self.plan = self.repo / 'implementation.md'
        self.code.write_text('VALUE = 0\n', encoding='utf-8', newline='')
        self.harness = Harness(self.root / 'state')
        self.record('t1', 'Change app.py to return 1.', 1, 'Initial plan')
        self.record('t2', 'Change app.py to return 2.', 2, 'Task B plan')
        self.record('t3', 'Correction: app.py should return 3, not 2.', 3, 'Updated Task B plan')
        self.hook('UserPromptSubmit', 'review', prompt='Disclose the last task.')


    def tearDown(self):
        self.temp.__exit__(None, None, None)


    def hook(self, name, turn, **args):
        return handle_hook(self.harness, dict(session_id='s', turn_id=turn, cwd=str(self.repo), hook_event_name=name, **args))


    def record(self, turn, prompt, value, plan):
        self.hook('UserPromptSubmit', turn, prompt=prompt)
        self.code.write_text(f'VALUE = {value}\n', encoding='utf-8', newline='')
        self.plan.write_text(plan, encoding='utf-8', newline='')
        self.hook('PostToolUse', turn, tool_use_id='edit', tool_name='Edit', tool_input={'path': 'app.py'}, tool_response=f'Updated {value}')
        self.hook('Stop', turn, last_assistant_message=f'app.py changed to {value}')


    def checklist(self, context):
        return [{'id': 'R1', 'check': 'app.py should return 3, superseding 2.',
                 'refs': [{'source_id': s['id'], 'quote': s['content']} for s in context['sources'] if s['kind'] == 'user']}]


    def test_plan_provenance_is_not_invented_from_filename(self):
        context = checkpoint_task(self.harness)
        source = next(s for s in context['sources'] if s['kind'] == 'plan')
        plan_id = source['id']
        self.assertEqual(source['kind'], 'plan')
        requirements = [{'id': 'R1', 'check': 'Plan commitment', 'refs': [{'source_id': plan_id, 'quote': source['content'], 'origin': 'agent_plan'}]}]
        payload = yaml.safe_load(self.harness.build_evidence_groups(context['task_id'], requirements))
        self.assertIn('agent_plan', payload['evidence_groups']['R1']['cited_sources']['plan'][0]['source'])


    def test_new_session_clears_all_old_data_and_old_refs_but_not_repo(self):
        context = checkpoint_task(self.harness)
        payload = yaml.safe_load(self.harness.build_evidence_groups(context['task_id'], self.checklist(context)))
        original = self.code.read_bytes()
        handle_hook(self.harness, {'hook_event_name': 'SessionStart', 'session_id': 'new'})
        with self.harness.store.connect() as db:
            for table in ('files', 'snapshots', 'views', 'outputs', 'tasks', 'recorded_turns', 'recorded_events', 'recordings'):
                self.assertEqual(db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0], 0, table)
        self.assertEqual(self.code.read_bytes(), original)
        with self.assertRaises(HarnessError):
            self.harness.build_evidence_groups(context['task_id'], read_ref=payload['read_ref'])
        # A late Stop cannot recreate the previous session.
        self.hook('Stop', 'review', last_assistant_message='Late')
        self.assertEqual(self.harness.store.current_session(), 'new')
        self.assertEqual(self.harness.sessions.events('s'), [])


    def test_same_session_resume_preserves_sources_and_evidence(self):
        context = checkpoint_task(self.harness)
        original = self.harness.build_evidence_groups(context['task_id'], self.checklist(context))
        self.harness = Harness(self.root / 'state')
        handle_hook(self.harness, {'hook_event_name': 'SessionStart', 'session_id': 's', 'source': 'resume'})
        self.assertEqual(self.harness.build_evidence_groups(context['task_id']), original)


    def test_session_switch_during_capture_cannot_leave_old_records(self):
        from codex_harness.application.repository import capture
        def racing_capture(*args, **kwargs):
            self.harness.store.activate_session('new')
            return capture(*args, **kwargs)
        with patch('codex_harness.application.sessions.capture', side_effect=racing_capture):
            with self.assertRaises(HarnessError):
                self.hook('Stop', 'review', last_assistant_message='Done')
        with self.harness.store.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM files').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM snapshots').fetchone()[0], 0)


