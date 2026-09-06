from __future__ import annotations

import unittest

import yaml

from codex_harness import Harness
from codex_harness.application.storage import HarnessError
from codex_harness.hooks.lifecycle import handle_hook
from tests.support import ProjectTemporaryDirectory


class PlanOnlyTests(unittest.TestCase):
    def setUp(self):
        self.temp = ProjectTemporaryDirectory()
        self.root = self.temp.__enter__()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.code = self.repo / 'app.py'
        self.code.write_text('def run():\n    return 1\n', encoding='utf-8')
        self.plan = self.repo / 'BEG.md'
        self.demand = 'app.py::run must return 2.'
        self.plan.write_text(self.demand, encoding='utf-8')
        self.harness = Harness(self.root / 'state')

    def tearDown(self):
        self.temp.__exit__(None, None, None)

    def test_unmatched_requirement_has_section_navigation_and_frozen_symbol_reads(self):
        code = 'def extract():\n    return "original"\n\ndef unrelated():\n    return "other"\n'
        target = self.repo / 'evidence_intake.py'
        target.write_text(code, encoding='utf-8', newline='')
        quote = '保留输入原文和位置。'
        self.plan.write_text('# Design\n## Evidence Intake\n' + quote + '\n## Other\nElsewhere.', encoding='utf-8')
        listing = yaml.safe_load(self.harness.list_task_sources(repo_path=str(self.repo)))
        plan_id = listing['plans'][0]['id']
        task = yaml.safe_load(self.harness.select_task(plan_ids=[plan_id]))['task_id']
        payload = yaml.safe_load(self.harness.build_evidence_groups(task, [
            {'id': 'R1', 'check': '检查原文保留', 'refs': [{'source_id': plan_id, 'quote': quote}]}]))
        group = payload['evidence_groups']['R1']
        self.assertNotIn('actual', group)
        navigation = group['navigation']
        self.assertEqual([c['path'] for c in navigation['candidates']], ['evidence_intake.py'])
        self.assertIn('Evidence Intake', navigation['candidates'][0]['basis'])
        target.write_text('changed live source', encoding='utf-8')
        self.harness = Harness(self.root / 'state')
        read = lambda ref: yaml.safe_load(self.harness.build_evidence_groups(task, read_ref=ref))
        outline = read(navigation['candidates'][0]['read_ref'])
        entry = next(s for s in outline['symbols'] if s['symbol'] == 'extract')
        excerpt = read(entry['read_ref'])
        self.assertEqual(excerpt['content'], code.split('\ndef unrelated')[0])
        self.assertNotIn('other', excerpt['content'])
        index = read(payload['repository']['read_ref'])['content']
        self.assertEqual({e['path'] for e in index}, {'BEG.md', 'app.py', 'evidence_intake.py'})

    def test_unrelated_section_does_not_supply_navigation(self):
        quote = '保留现有行为。'
        self.plan.write_text('## App Module\napp.py\n## Other\n' + quote, encoding='utf-8')
        listing = yaml.safe_load(self.harness.list_task_sources(repo_path=str(self.repo)))
        plan_id = listing['plans'][0]['id']
        task = yaml.safe_load(self.harness.select_task(plan_ids=[plan_id]))['task_id']
        result = yaml.safe_load(self.harness.build_evidence_groups(task, [
            {'id': 'R1', 'check': 'app.py must work', 'refs': [{'source_id': plan_id, 'quote': quote}]}]))
        group = result['evidence_groups']['R1']
        self.assertNotIn('actual', group)
        self.assertEqual(group['navigation']['candidates'], [])

    def test_plan_without_hooks_builds_frozen_current_evidence(self):
        listing = yaml.safe_load(self.harness.list_task_sources(repo_path=str(self.repo)))
        self.assertEqual(listing['prompts'], [])
        plan_id = listing['plans'][0]['id']
        selected = yaml.safe_load(self.harness.select_task(plan_ids=[plan_id]))
        self.assertEqual(selected['scope']['mode'], 'current')
        self.assertNotIn('repo_before', selected['scope'])
        self.assertEqual(selected['sources'][0]['content'], self.demand)
        self.code.write_text('def run():\n    return 999\n', encoding='utf-8')
        self.plan.unlink()
        self.harness = Harness(self.root / 'state')
        payload = yaml.safe_load(self.harness.build_evidence_groups(selected['task_id'], [
            {'id': 'R1', 'check': self.demand, 'refs': [{'source_id': plan_id, 'quote': self.demand}]}]))
        self.assertIn('当前', payload['beg_disclose_prompt'])
        self.assertNotIn('changes', payload)
        group = payload['evidence_groups']['R1']
        self.assertIn('return 1', str(group['actual']['repo']))
        self.assertNotIn('999', str(group))
        self.assertNotIn('trace', group['actual'])
        view = self.harness.store.latest_view(selected['task_id'])
        self.assertEqual(view['events'], [])
        self.assertEqual(view['changes'], {})
        with self.assertRaises(HarnessError):
            self.harness.material(view['view_id'], 'diff:app.py')
        with self.assertRaises(HarnessError):
            self.harness.build_evidence_groups(selected['task_id'], read_ref=f"{view['view_id']}:changes")
        with self.harness.store.connect() as db:
            for table in ('recorded_turns', 'recorded_events', 'snapshots'):
                self.assertEqual(db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0], 0)

    def test_only_check_request_does_not_prevent_plan_review(self):
        handle_hook(self.harness, {'hook_event_name': 'UserPromptSubmit', 'session_id': 's',
                    'turn_id': 'review', 'cwd': str(self.repo), 'prompt': 'Check BEG.md implementation.'})
        listing = yaml.safe_load(self.harness.list_task_sources())
        self.assertFalse(listing['prompts'][0]['complete'])
        selected = yaml.safe_load(self.harness.select_task(plan_ids=[listing['plans'][0]['id']]))
        self.assertEqual([s['kind'] for s in selected['sources']], ['plan'])
        self.assertEqual(self.harness.store.task(selected['task_id'])['scope']['event_ids'], [])
        with self.assertRaises(HarnessError):
            self.harness.select_task('P1', 'P1', [listing['plans'][0]['id']])

    def test_empty_sources_partial_range_and_foreign_plan_are_rejected(self):
        listing = yaml.safe_load(self.harness.list_task_sources(repo_path=str(self.repo)))
        plan_id = listing['plans'][0]['id']
        for args in ({}, {'start_prompt': 'P1', 'plan_ids': [plan_id]},
                     {'plan_ids': [plan_id, plan_id]}, {'plan_ids': ['L99999']}):
            with self.subTest(args=args), self.assertRaises(HarnessError):
                self.harness.select_task(**args)
        foreign = self.harness.store.cache_file(str(self.root / 'elsewhere'), 'plan.md', self.demand)
        with self.assertRaises(HarnessError):
            self.harness.select_task(plan_ids=[f'L{foreign}'])
        self.plan.write_text(' \n', encoding='utf-8')
        current = yaml.safe_load(self.harness.list_task_sources(repo_path=str(self.repo)))
        empty = next(p for p in current['plans'] if p.get('version') == 'current')
        with self.assertRaises(HarnessError):
            self.harness.select_task(plan_ids=[empty['id']])

    def test_current_plan_reads_do_not_replace_historical_sources(self):
        base = {'session_id': 's', 'turn_id': 't', 'cwd': str(self.repo)}
        handle_hook(self.harness, {**base, 'hook_event_name': 'UserPromptSubmit', 'prompt': self.demand})
        handle_hook(self.harness, {**base, 'hook_event_name': 'Stop', 'last_assistant_message': 'Done'})
        self.plan.write_text('A later plan', encoding='utf-8')
        listing = yaml.safe_load(self.harness.list_task_sources(repo_path=str(self.repo)))
        later = next(p['id'] for p in listing['plans'] if p.get('version') == 'current')
        with self.assertRaises(HarnessError):
            self.harness.select_task('P1', 'P1', [later])
        selected = yaml.safe_load(self.harness.select_task('P1', 'P1'))
        self.assertEqual(selected['sources'][0]['content'], self.demand)
