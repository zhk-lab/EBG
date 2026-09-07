from __future__ import annotations

import unittest

import yaml

from codex_harness import Harness
from codex_harness.hooks.lifecycle import handle_hook
from tests.support import ProjectTemporaryDirectory


class CollectionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = ProjectTemporaryDirectory()
        self.root = self.temp.__enter__()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.demand = 'app.py must return a sorted list.'
        (self.repo / 'PLAN.md').write_text(self.demand, encoding='utf-8')
        (self.repo / 'app.py').write_text('def run(values):\n    return sorted(values)\n', encoding='utf-8')
        self.harness = Harness(self.root / 'state', max_file_bytes=512)

    def tearDown(self):
        self.temp.__exit__(None, None, None)

    def select_plan(self):
        listing = yaml.safe_load(self.harness.list_task_sources(repo_path=str(self.repo)))
        plan_id = listing['plans'][0]['id']
        selected = yaml.safe_load(self.harness.select_task(plan_ids=[plan_id]))
        return selected['task_id'], [{'id': 'R1', 'check': self.demand,
                                     'refs': [{'source_id': plan_id, 'quote': self.demand}]}]

    def index(self, task, payload):
        result = yaml.safe_load(self.harness.build_evidence_groups(task, read_ref=payload['repository']['read_ref']))
        return {entry['path']: entry for entry in result['content']}

    def test_binary_existence_survives_frozen_plan_and_restart(self):
        library = self.repo / 'csort.dll'
        library.write_bytes(b'MZ\x00binary')
        task, requirements = self.select_plan()
        library.unlink()
        self.harness = Harness(self.root / 'state')
        payload = yaml.safe_load(self.harness.build_evidence_groups(task, requirements))
        index = self.index(task, payload)
        self.assertEqual(payload['repository']['count'], 3)
        self.assertEqual(payload['repository']['content_count'], 2)
        self.assertEqual(index['csort.dll'], {
            'path': 'csort.dll', 'exists': True, 'size_bytes': 9,
            'content_status': 'not_collected', 'reason': 'binary_extension',
        })
        self.assertIn('read_ref', index['app.py'])
        self.assertIn('未采集不等于不存在', payload['repository']['note'])

    def test_skipped_content_keeps_reason_without_inventing_content_refs(self):
        (self.repo / 'data.dat').write_bytes(b'abc\x00def')
        (self.repo / 'legacy.txt').write_bytes(b'\xff\xfe')
        (self.repo / 'large.txt').write_text('x' * 513, encoding='utf-8')
        task, requirements = self.select_plan()
        payload = yaml.safe_load(self.harness.build_evidence_groups(task, requirements))
        index = self.index(task, payload)
        for path, reason in [('data.dat', 'binary_content'), ('legacy.txt', 'non_utf8'), ('large.txt', 'size_limit')]:
            with self.subTest(path=path):
                self.assertTrue(index[path]['exists'])
                self.assertEqual(index[path]['reason'], reason)
                self.assertNotIn('read_ref', index[path])
        self.assertNotIn('missing.dll', index)

    def test_plan_only_does_not_repeat_trace_warning_per_requirement(self):
        task, requirements = self.select_plan()
        payload = yaml.safe_load(self.harness.build_evidence_groups(task, requirements))
        self.assertNotIn('未匹配到执行 Trace', payload['evidence_groups']['R1'].get('note', ''))
        self.assertIn('不自动构成', payload['beg_disclose_prompt'])

    def test_historical_inventory_is_frozen_and_binary_conversion_is_not_deletion(self):
        asset = self.repo / 'asset.dat'
        asset.write_text('original text', encoding='utf-8')
        base = {'session_id': 's', 'turn_id': 't', 'cwd': str(self.repo)}
        handle_hook(self.harness, {**base, 'hook_event_name': 'UserPromptSubmit', 'prompt': self.demand})
        asset.write_bytes(b'\x00binary')
        (self.repo / 'csort.dll').write_bytes(b'MZ\x00binary')
        handle_hook(self.harness, {**base, 'hook_event_name': 'Stop', 'last_assistant_message': 'Done'})
        selected = yaml.safe_load(self.harness.select_task('P1', 'P1'))
        task = selected['task_id']
        asset.unlink()
        (self.repo / 'csort.dll').unlink()
        payload = yaml.safe_load(self.harness.build_evidence_groups(task, [
            {'id': 'R1', 'check': self.demand, 'refs': [{'source_id': 'P1', 'quote': self.demand}]}]))
        index = self.index(task, payload)
        self.assertTrue(index['csort.dll']['exists'])
        self.assertTrue(index['asset.dat']['exists'])
        self.assertNotIn('asset.dat', self.harness.store.latest_view(task)['changes'])
        snapshot = selected['scope']['repo_after']
        listing = self.harness.sessions.material('s', snapshot_id=snapshot)
        self.assertIn('csort.dll', listing['files'])
        metadata = self.harness.sessions.material('s', snapshot_id=snapshot, path='csort.dll')
        self.assertTrue(metadata['exists'])
        self.assertNotIn('content', metadata)
        self.assertNotIn('未匹配到执行 Trace', payload['evidence_groups']['R1'].get('note', ''))
