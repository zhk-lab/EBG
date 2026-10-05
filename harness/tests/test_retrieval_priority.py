from __future__ import annotations

import json
import shutil
import unittest

import yaml

from codex_harness import Harness
from codex_harness.integration import write_integration
from tests.support import ProjectTemporaryDirectory, checkpoint_task


class RetrievalPriorityTests(unittest.TestCase):
    def setUp(self):
        self.temp = ProjectTemporaryDirectory()
        self.root = self.temp.__enter__()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.harness = Harness(self.root / 'state', token_budget=2200)

    def tearDown(self):
        self.temp.__exit__(None, None, None)

    def test_actual_selector_and_implementation_survive_first_page(self):
        (self.repo / 'engine.py').write_text(
            'import random\n\ndef queued(jobs, config):\n    return [run_job(job) for job in jobs]\n\n'
            'def run_job(job):\n    return random.Random(job["seed"])\n\n'
            'def local(jobs, config):\n    rng = random.Random(config["seed"])\n'
            '    return [rng.random() for job in jobs]\n', encoding='utf-8')
        report = {f'metric_{i}': list(range(20)) for i in range(24)}
        (self.repo / 'a_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        demand = 'Compare `a_results.json` with `engine.py`, `queued`, and `local`.'
        (self.repo / 'PLAN.md').write_text(demand, encoding='utf-8')
        self.harness.sessions.start('s', 't', str(self.repo), demand)
        self.harness.sessions.append('s', 't', {'kind': 'tool_call', 'call_id': 'c',
            'tool_name': 'Bash', 'content': json.dumps({'command': 'python engine.py --mode local'})}, 'call')
        selected = checkpoint_task(self.harness)
        plan = next(s['id'] for s in selected['sources'] if s['kind'] == 'plan')
        self.harness.build_evidence_groups(selected['task_id'], [{'id': 'R1', 'check': 'Inspect execution.',
            'refs': [{'source_id': plan, 'quote': demand}]}])
        value = self.harness.store.latest_view(selected['task_id'])
        requirement = {'id': 'R1', 'check': 'Inspect execution.',
                       'refs': [{'source_id': plan, 'kind': 'plan', 'source': 'PLAN.md@1',
                                 'content': demand, 'start': 0, 'end': len(demand)}]}
        entries = self.harness._repo_evidence(value, requirement)
        self.assertEqual(entries[0]['node'], 'engine.py::local')
        self.assertTrue(any(e.get('node') == 'engine.py::run_job' for e in entries))
        owner = 'evidence:' + selected['task_id']
        ref = self.harness.pages.save(owner, entries)
        page = yaml.safe_load(self.harness.pages.read(owner, ref))
        first = page.get('content', page.get('entries', []))
        text = str(first)
        self.assertIn('def local', text)
        self.assertIn('random.Random(config["seed"])', text)
        report_entry = next(e for e in first if e.get('node') == 'a_results.json::<file>')
        report_page = yaml.safe_load(self.harness.pages.read(owner, report_entry['content_ref']))
        self.assertIn('metric_0', report_page['content'])
        self.assertIn('next', report_page)
        tail = yaml.safe_load(self.harness.pages.read(owner, report_entry['content_ref'], report_page['next']['offset']))
        self.assertIn('metric_23', tail['content'])

    def test_agent_report_and_read_command_are_not_execution_hints(self):
        (self.repo / 'engine.py').write_text(
            'def first():\n    return 1\n\ndef second():\n    return 2\n', encoding='utf-8')
        demand = 'Inspect `engine.py`, `first`, then `second`.'
        (self.repo / 'PLAN.md').write_text(demand, encoding='utf-8')
        self.harness.sessions.start('s', 't', str(self.repo), demand)
        self.harness.sessions.append('s', 't', {'kind': 'tool_call', 'call_id': 'c',
            'tool_name': 'Bash', 'content': json.dumps({'command': 'Get-Content engine.py second'})}, 'call')
        self.harness.sessions.append('s', 't', {'kind': 'assistant',
            'content': 'I executed python engine.py --mode second.'}, 'claim')
        selected = checkpoint_task(self.harness)
        plan = next(s['id'] for s in selected['sources'] if s['kind'] == 'plan')
        self.harness.build_evidence_groups(selected['task_id'], [{'id': 'R1', 'check': demand,
            'refs': [{'source_id': plan, 'quote': demand}]}])
        value = self.harness.store.latest_view(selected['task_id'])
        requirement = {'id': 'R1', 'check': demand,
                       'refs': [{'source_id': plan, 'kind': 'plan', 'source': 'PLAN.md@1',
                                 'content': demand, 'start': 0, 'end': len(demand)}]}
        entries = self.harness._repo_evidence(value, requirement)
        self.assertEqual(entries[0]['node'], 'engine.py::first')

    def test_generated_harness_files_do_not_pollute_task_materials(self):
        write_integration(self.repo / '.codex', self.root / 'runtime', 'python')
        shutil.move(str(self.repo / '.codex/skills'), str(self.repo / '.agents/skills'))
        custom = self.repo / '.agents/skills/user-analysis/SKILL.md'
        custom.parent.mkdir(parents=True)
        custom.write_text('User research instructions.', encoding='utf-8')
        (self.repo / 'app.py').write_text('def run():\n    return 1\n', encoding='utf-8')
        (self.repo / 'PLAN.md').write_text('Inspect app.py.', encoding='utf-8')
        selected = checkpoint_task(self.harness, repo=self.repo)
        files = self.harness.store.task(selected['task_id'])['current_files']
        self.assertIn('.agents/skills/user-analysis/SKILL.md', files)
        self.assertIn('app.py', files)
        self.assertFalse(any('/ebg-' in p for p in files))
        self.assertNotIn('.codex/hooks.json', files)
        self.assertNotIn('.codex/config.toml', files)

    def test_mixed_user_configuration_remains_collected(self):
        write_integration(self.repo / '.codex', self.root / 'runtime', 'python')
        config = self.repo / '.codex/config.toml'
        config.write_text(config.read_text(encoding='utf-8') + '\n[mcp_servers.user_tool]\ncommand="user-tool"\n', encoding='utf-8')
        (self.repo / 'PLAN.md').write_text('Inspect configuration.', encoding='utf-8')
        selected = checkpoint_task(self.harness, repo=self.repo)
        self.assertIn('.codex/config.toml', self.harness.store.task(selected['task_id'])['current_files'])
