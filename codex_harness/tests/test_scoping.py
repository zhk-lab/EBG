from __future__ import annotations

import unittest
from unittest.mock import patch

import yaml

from codex_harness import Harness
from codex_harness.application.paging import child_ref
from codex_harness.application.render import tokens
from codex_harness.application.storage import HarnessError
from codex_harness.hooks.lifecycle import handle_hook
from tests.support import ProjectTemporaryDirectory


class ScopeTests(unittest.TestCase):
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

    def select(self, **changes):
        return yaml.safe_load(self.harness.select_task(**(dict(start_prompt='P2', end_prompt='P3') | changes)))

    def checklist(self, context):
        return [{'id': 'R1', 'check': 'app.py should return 3, superseding 2.',
                 'refs': [{'source_id': s['id'], 'quote': s['content']} for s in context['sources'] if s['kind'] == 'user']}]

    def test_full_prompts_and_stable_historical_plan_versions_are_listed(self):
        listing = yaml.safe_load(self.harness.list_task_sources())
        self.assertEqual([p['id'] for p in listing['prompts']], ['P1', 'P2', 'P3', 'P4'])
        self.assertEqual(listing['prompts'][2]['content'], 'Correction: app.py should return 3, not 2.')
        self.assertFalse(listing['prompts'][-1]['complete'])
        self.assertEqual(len(listing['plans']), 3)
        ids = [p['id'] for p in listing['plans']]
        self.plan.unlink()
        self.assertEqual([p['id'] for p in yaml.safe_load(self.harness.list_task_sources())['plans']], ids)
        context = self.select(plan_ids=ids[1:])
        self.assertEqual([s['content'] for s in context['sources'] if s['kind'] == 'plan'], ['Task B plan', 'Updated Task B plan'])

    def test_continuous_scope_auto_selects_trace_and_both_snapshots(self):
        context = self.select()
        self.assertEqual([s['id'] for s in context['sources']], ['P2', 'P3'])
        self.code.write_text('VALUE = 999', encoding='utf-8')
        self.plan.unlink()
        with patch('codex_harness.application.sessions.capture', side_effect=AssertionError('live capture')):
            result = self.harness.build_evidence_groups(context['task_id'], self.checklist(context))
        view = self.harness.store.latest_view(context['task_id'])
        self.assertEqual({e['turn_id'] for e in view['events']}, {'t2', 't3'})
        diff = self.harness.material(view['view_id'], 'diff:app.py')['content']
        self.assertIn('-VALUE = 1', diff)
        self.assertIn('+VALUE = 3', diff)
        self.assertNotIn('999', result)
        self.assertEqual(self.select()['task_id'], context['task_id'])

    def test_missing_end_snapshot_or_invalid_range_is_rejected(self):
        for args in [dict(end_prompt='P4'), dict(start_prompt='P3', end_prompt='P2'), dict(start_prompt='P99'), dict(end_prompt='s3')]:
            with self.subTest(args=args), self.assertRaises(HarnessError):
                self.select(**args)

    def test_plan_created_after_end_cannot_leak_into_earlier_task(self):
        plans = yaml.safe_load(self.harness.list_task_sources())['plans']
        with self.assertRaises(HarnessError):
            self.select(start_prompt='P1', end_prompt='P1', plan_ids=[plans[-1]['id']])
        # An existing plan can legitimately predate the selected task.
        self.assertEqual(self.select(plan_ids=[plans[0]['id']])['sources'][-1]['content'], 'Initial plan')

    def test_plan_provenance_is_not_invented_from_filename(self):
        plan_id = yaml.safe_load(self.harness.list_task_sources())['plans'][1]['id']
        context = self.select(plan_ids=[plan_id])
        source = context['sources'][-1]
        self.assertEqual(source['kind'], 'plan')
        requirements = [{'id': 'R1', 'check': 'Plan commitment', 'refs': [{'source_id': plan_id, 'quote': source['content'], 'origin': 'agent_plan'}]}]
        payload = yaml.safe_load(self.harness.build_evidence_groups(context['task_id'], requirements))
        self.assertIn('agent_plan', payload['evidence_groups']['R1']['requirement']['plan'][0]['source'])

    def test_long_prompt_and_selected_sources_can_be_read_in_full(self):
        long = 'Unique prompt body. ' * 2000
        self.hook('Stop', 'review', last_assistant_message='Review done')
        self.hook('UserPromptSubmit', 'long', prompt=long)
        self.hook('Stop', 'long', last_assistant_message='Done')
        self.harness.token_budget = 500
        raw = self.harness.list_task_sources()
        self.assertLessEqual(tokens(raw), 500)
        ref = child_ref(child_ref(child_ref(yaml.safe_load(raw)['read_ref'], 'prompts'), 4), 'content')
        recovered = self.recover(lambda **kwargs: self.harness.list_task_sources(**kwargs), ref)
        self.assertEqual(recovered, long)
        selected = yaml.safe_load(self.harness.select_task('P5', 'P5'))
        ref = child_ref(child_ref(child_ref(selected['read_ref'], 'sources'), 0), 'content')
        self.assertEqual(self.recover(lambda **kwargs: self.harness.select_task(**kwargs), ref), long)

    def recover(self, call, ref):
        chunks, offset = [], 0
        while True:
            raw = call(read_ref=ref, offset=offset)
            self.assertLessEqual(tokens(raw), self.harness.token_budget)
            page = yaml.safe_load(raw)
            chunks.append(page['content'])
            if 'next' not in page:
                return ''.join(chunks)
            offset = page['next']['offset']

    def test_new_session_clears_all_old_data_and_old_refs_but_not_repo(self):
        context = self.select()
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
        context = self.select()
        original = self.harness.build_evidence_groups(context['task_id'], self.checklist(context))
        self.harness = Harness(self.root / 'state')
        handle_hook(self.harness, {'hook_event_name': 'SessionStart', 'session_id': 's', 'source': 'resume'})
        self.assertEqual(self.harness.build_evidence_groups(context['task_id']), original)

    def test_plan_first_seen_in_next_before_snapshot_is_not_a_past_version(self):
        self.hook('Stop', 'review', last_assistant_message='Done')
        self.plan.write_text('Created between turns', encoding='utf-8')
        self.hook('UserPromptSubmit', 'next', prompt='New task')
        plans = yaml.safe_load(self.harness.list_task_sources())['plans']
        with self.assertRaises(HarnessError):
            self.select(start_prompt='P4', end_prompt='P4', plan_ids=[plans[-1]['id']])

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
