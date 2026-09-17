from __future__ import annotations

import unittest

import yaml

from codex_harness import Harness
from tests.support import ProjectTemporaryDirectory, checkpoint_task


class PlanNavigationTests(unittest.TestCase):
    def setUp(self):
        self.temp = ProjectTemporaryDirectory()
        self.root = self.temp.__enter__()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.code = self.repo / 'app.py'
        self.code.write_text('def run():\n    return 1\n', encoding='utf-8')
        self.plan = self.repo / 'EBG.md'
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
        selected = checkpoint_task(self.harness, repo=self.repo, prompt='Inspect EBG.md.')
        plan_id = next(s['id'] for s in selected['sources'] if s['kind'] == 'plan')
        task = selected['task_id']
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
        self.assertEqual({e['path'] for e in index}, {'EBG.md', 'app.py', 'evidence_intake.py'})

    def test_unrelated_section_does_not_supply_navigation(self):
        quote = '保留现有行为。'
        self.plan.write_text('## App Module\napp.py\n## Other\n' + quote, encoding='utf-8')
        selected = checkpoint_task(self.harness, repo=self.repo, prompt='Inspect EBG.md.')
        plan_id = next(s['id'] for s in selected['sources'] if s['kind'] == 'plan')
        task = selected['task_id']
        result = yaml.safe_load(self.harness.build_evidence_groups(task, [
            {'id': 'R1', 'check': 'Check the quoted section.', 'refs': [{'source_id': plan_id, 'quote': quote}]}]))
        group = result['evidence_groups']['R1']
        self.assertNotIn('actual', group)
        self.assertEqual(group['navigation']['candidates'], [])

