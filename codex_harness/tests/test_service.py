from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import yaml

from codex_harness import Harness
from codex_harness.application.paging import child_ref
from codex_harness.application.render import tokens
from codex_harness.application.storage import HarnessError
from tests.support import ProjectTemporaryDirectory, checkpoint_task


CODE = "class RetryClient:\n    def send(self, request):\n        for attempt in range(3):\n            try:\n                return self.transport.send(request)\n            except RuntimeError:\n                if attempt == 2:\n                    raise\n"
PROMPT = "Modify src/retry.py RetryClient.send to retry at most twice.\nRun tests/test_retry.py to verify."


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = ProjectTemporaryDirectory()
        self.root = self.temp.__enter__()
        self.repo = self.root / 'repo'
        (self.repo / 'src').mkdir(parents=True)
        (self.repo / 'tests').mkdir()
        self.code = self.repo / 'src/retry.py'
        self.code.write_text(CODE, encoding='utf-8', newline='')
        (self.repo / 'tests/test_retry.py').write_text('def test_retry():\n    assert True\n', encoding='utf-8')
        self.harness = Harness(self.root / 'state')
        self.harness.sessions.start('s', 't1', str(self.repo), PROMPT)
        self.requirements = [
            {'id': 'R1', 'check': 'Retry at most twice.', 'refs': [{'source_id': 'P1', 'quote': PROMPT.splitlines()[0]}]},
            {'id': 'R2', 'check': 'Run the test.', 'refs': [{'source_id': 'P1', 'quote': PROMPT.splitlines()[1]}]},
        ]

    def tearDown(self):
        self.temp.__exit__(None, None, None)

    def event(self, kind, content, key, tool='Edit', call='c'):
        event = {'kind': kind, 'content': content}
        if kind.startswith('tool_'):
            event.update(tool_name=tool, call_id=call)
        self.harness.sessions.append('s', 't1', event, key)

    def select(self):
        self.harness.sessions.stop('s', 't1', None)
        context = checkpoint_task(self.harness)
        self.task = context['task_id']
        return context

    def build(self):
        if not hasattr(self, 'task'):
            self.select()
        return yaml.safe_load(self.harness.build_evidence_groups(self.task, self.requirements))

    def read(self, ref, offset=0):
        return yaml.safe_load(self.harness.build_evidence_groups(self.task, read_ref=ref, offset=offset))

    def test_groups_pair_results_and_do_not_confuse_reading_tests_with_running_them(self):
        self.code.write_text(CODE.replace('range(3)', 'range(4)').replace('attempt == 2', 'attempt == 3'), encoding='utf-8')
        self.event('tool_call', '{"path":"src/retry.py"}', 'edit')
        self.event('tool_result', 'Updated src/retry.py', 'result')
        self.event('assistant', 'RetryClient.send now retries twice.', 'reply')
        self.event('tool_call', '{"path":"tests/test_retry.py"}', 'read-test', tool='Read', call='read')
        payload = self.build()
        group = payload['evidence_groups']['R1']
        self.assertIn('range(4)', group['actual']['repo'][0]['content'])
        self.assertEqual(len(group['actual']['trace']['action']), 2)
        self.assertEqual(group['actual']['trace']['action'][1]['content'], 'Updated src/retry.py')
        self.assertNotIn('trace', payload['evidence_groups']['R2'].get('actual', {}))
        self.assertNotIn('verdict', group)
        self.assertTrue(any('-        for attempt in range(3)' in r['content'] for r in group['actual']['repo']))

    def test_old_evidence_survives_live_edits_and_checklist_revision(self):
        payload = self.build()
        ref = payload['read_ref']
        self.code.write_text('NEW LIVE CODE', encoding='utf-8')
        self.requirements = [self.requirements[0]]
        second = self.build()
        self.assertNotEqual(ref, second['read_ref'])
        self.assertEqual(self.read(ref), payload)
        self.assertIn('R2', self.read(ref)['questions'])
        self.assertNotIn('NEW LIVE CODE', json.dumps(second))

    def test_incremental_graph_and_output_survive_restart(self):
        payload = self.build()
        self.harness = Harness(self.root / 'state')
        self.assertTrue(self.harness.refresh_task(self.task)['reused'])
        self.assertEqual(self.harness.refresh_task(self.task)['files_built'], 0)
        self.assertEqual(self.build(), payload)

    def test_interrupted_construction_resumes_from_saved_file_fragments(self):
        self.select()
        from codex_harness.application import repository
        original = repository.build_evidence
        calls = []
        def interrupted(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('interrupted')
            return original(*args, **kwargs)
        with patch.object(repository, 'build_evidence', side_effect=interrupted):
            with self.assertRaises(RuntimeError):
                self.build()
        self.harness = Harness(self.root / 'state')
        self.assertIn('R1', self.build()['evidence_groups'])

    def test_invalid_reference_does_not_overwrite_saved_requirements(self):
        self.build()
        previous = self.harness.store.task(self.task)['requirements']
        with self.assertRaisesRegex(HarnessError, 'not verbatim'):
            self.harness.build_evidence_groups(self.task, [{'id': 'R1', 'check': 'Anything', 'refs': [{'source_id': 'P1', 'quote': 'invented'}]}])
        self.assertEqual(self.harness.store.task(self.task)['requirements'], previous)

    def test_no_fallback_from_generic_demand_to_unrelated_code(self):
        self.harness.sessions.stop('s', 't1', None)
        self.harness.sessions.start('s', 't2', str(self.repo), 'Preserve compatibility.')
        self.harness.sessions.stop('s', 't2', None)
        self.task = checkpoint_task(self.harness)['task_id']
        self.requirements = [{'id': 'R1', 'check': 'Preserve compatibility.',
                              'refs': [{'source_id': 'P2', 'quote': 'compatibility'}]}]
        group = self.build()['evidence_groups']['R1']
        self.assertNotIn('actual', group)
        self.assertIn('note', group)

    def test_large_evidence_directory_preserves_source_and_direct_content_link(self):
        self.code.write_text(CODE + '# ' + 'long text ' * 1000, encoding='utf-8')
        self.requirements = [{'id': 'R1', 'check': 'File context',
                              'refs': [{'source_id': 'P1', 'quote': 'src/retry.py'}]}]
        payload = self.build()
        self.harness.token_budget = 500
        ref = child_ref(child_ref(child_ref(child_ref(payload['read_ref'], 'evidence_groups'), 'R1'), 'actual'), 'repo')
        raw = self.harness.build_evidence_groups(self.task, read_ref=ref)
        self.assertLessEqual(tokens(raw), 500)
        page = yaml.safe_load(raw)
        entry = page['entries'][0]
        self.assertIn('src/retry.py', entry['source'])
        self.assertGreater(entry['content_chars'], 1000)
        text = self.read(entry['content_ref'])
        self.assertTrue(text['content'].startswith('class RetryClient:'))

    def test_plain_word_does_not_reenter_through_stub_context(self):
        self.code.write_text('def document():\n    pass\n', encoding='utf-8')
        self.harness.sessions.stop('s', 't1', None)
        self.harness.sessions.start('s', 't2', str(self.repo), 'Users read the document')
        self.harness.sessions.stop('s', 't2', None)
        task = checkpoint_task(self.harness)['task_id']
        result = yaml.safe_load(self.harness.build_evidence_groups(task, [
            {'id': 'R1', 'check': 'Preserve reading',
             'refs': [{'source_id': 'P2', 'quote': 'Users read the document'}]}]))
        self.assertNotIn('actual', result['evidence_groups']['R1'])

    def test_deleted_file_is_kept_as_diff_without_target_writes(self):
        self.code.unlink()
        before = {p: p.read_bytes() for p in self.repo.rglob('*') if p.is_file()}
        payload = self.build()
        group = payload['evidence_groups']['R1']
        self.assertTrue(any('-class RetryClient' in r['content'] for r in group['actual']['repo']))
        self.assertEqual(before, {p: p.read_bytes() for p in self.repo.rglob('*') if p.is_file()})

    def test_pending_verification_has_call_without_result(self):
        self.event('tool_call', '{"command":"pytest tests/test_retry.py"}', 'pending', tool='Bash')
        group = self.build()['evidence_groups']['R2']
        self.assertEqual(len(group['actual']['trace']['action']), 1)
        self.assertIn('note', group)

    def test_other_path_or_ambiguous_basename_does_not_match(self):
        (self.repo / 'other').mkdir()
        (self.repo / 'other/retry.py').write_text('VALUE = 1', encoding='utf-8')
        self.event('tool_call', '{"path":"other/retry.py"}', 'wrong')
        self.event('tool_call', '{"path":"retry.py"}', 'ambiguous', call='other')
        self.assertNotIn('trace', self.build()['evidence_groups']['R1'].get('actual', {}))

    def test_one_hop_neighbor_keeps_original_and_relation(self):
        self.code.write_text('from helper import work\nclass RetryClient:\n    def send(self, request):\n        return work(request)\n', encoding='utf-8')
        (self.repo / 'src/helper.py').write_text('def work(request):\n    return request\n', encoding='utf-8', newline='')
        entries = self.build()['evidence_groups']['R1']['actual']['repo']
        neighbors = [r for r in entries if 'context' in r]
        self.assertTrue(neighbors)
        self.assertIn('calls', neighbors[0]['context'])
        self.assertEqual(neighbors[0]['content'], 'def work(request):\n    return request\n')

    def test_explicit_path_and_plain_function_keep_one_hop_dependency(self):
        self.code.write_text('from helper import work\n\ndef send():\n    return work()\n', encoding='utf-8')
        (self.repo / 'src/helper.py').write_text('def work():\n    return 5\n', encoding='utf-8')
        self.harness.sessions.stop('s', 't1', None)
        prompt = 'Implement src/retry.py::send to return 2.'
        self.harness.sessions.start('s', 't2', str(self.repo), prompt)
        self.harness.sessions.stop('s', 't2', None)
        task = checkpoint_task(self.harness)['task_id']
        result = yaml.safe_load(self.harness.build_evidence_groups(task, [
            {'id': 'R1', 'check': prompt, 'refs': [{'source_id': 'P2', 'quote': prompt}]}]))
        entries = result['evidence_groups']['R1']['actual']['repo']
        self.assertTrue(any('return 5' in r['content'] and 'context' in r for r in entries))

    def test_unmatched_claim_is_available_without_fabricating_requirement_association(self):
        self.event('assistant', 'All tests passed. Everything is complete.', 'claim')
        result = self.build()
        for group in result['evidence_groups'].values():
            self.assertNotIn('response', group.get('actual', {}).get('trace', {}))
        context = result['trace_context']
        self.assertIn('未直接关联', context['note'])
        claim = context['response'][0]
        self.assertEqual(claim['content'], 'All tests passed. Everything is complete.')
        self.assertEqual(self.read(claim['read_ref'])['content'], claim['content'])

    def test_reading_tests_is_retained_as_context_not_verification(self):
        self.event('tool_call', '{"path":"tests/test_retry.py"}', 'read', tool='Read')
        self.event('tool_result', 'def test_retry(): pass', 'read-result', tool='Read')
        self.requirements = [self.requirements[1]]
        result = self.build()
        self.assertNotIn('action', result['evidence_groups']['R2'].get('actual', {}).get('trace', {}))
        actions = result['trace_context']['action']
        self.assertEqual(len(actions), 2)
        self.assertIn('Read', actions[0]['source'])
        self.assertEqual(actions[1]['content'], 'def test_retry(): pass')

    def test_module_neighbor_does_not_expand_unrelated_functions(self):
        self.code.write_text("LIMIT = 3\nclass RetryClient:\n    def send(self, request):\n        return LIMIT\n\ndef unrelated():\n    return 'UNRELATED_BODY'\n", encoding='utf-8', newline='')
        entries = self.build()['evidence_groups']['R1']['actual']['repo']
        module = [e for e in entries if '::<module>' in e['source']]
        self.assertTrue(module)
        self.assertTrue(all('UNRELATED_BODY' not in e['content'] for e in module))

    def test_changes_directory_includes_unrelated_files(self):
        (self.repo / 'unrelated.py').write_text('VALUE = 9\n', encoding='utf-8')
        payload = self.build()
        entries = self.read(payload['changes']['read_ref'])['content']
        added = next(e for e in entries if e['path'] == 'unrelated.py')
        self.assertIn('+VALUE = 9', self.read(added['read_ref'])['content'])

    def test_crlf_and_long_single_line_are_recoverable_within_budget(self):
        content = (CODE + '# ' + 'long text ' * 1500).replace('\n', '\r\n')
        self.code.write_bytes(content.encode('utf-8'))
        payload = self.build()
        view = self.harness.store.latest_view(self.task)['view_id']
        self.harness.token_budget = 400
        root = self.read(view + ':repo:src/retry.py')
        reference = next(e['read_ref'] for e in root['entries'] if e['key'] == 'content')
        chunks, offset = [], 0
        while True:
            raw = self.harness.build_evidence_groups(self.task, read_ref=reference, offset=offset)
            self.assertLessEqual(tokens(raw), 400)
            page = yaml.safe_load(raw)
            chunks.append(page['content'])
            if 'next' not in page:
                break
            offset = page['next']['offset']
        self.assertEqual(''.join(chunks), content)

    def test_huge_checklist_and_group_directories_are_paged_without_loss(self):
        self.requirements = [dict(self.requirements[0], id=f'R{i}', check='requirement ' * 50) for i in range(1, 30)]
        self.select()
        self.harness.token_budget = 700
        initial = self.harness.build_evidence_groups(self.task, self.requirements)
        self.assertLessEqual(tokens(initial), 700)
        payload = yaml.safe_load(initial)
        ref = child_ref(payload['read_ref'], 'evidence_groups')
        seen, offset = [], 0
        while True:
            raw = self.harness.build_evidence_groups(self.task, read_ref=ref, offset=offset)
            self.assertLessEqual(tokens(raw), 700)
            page = yaml.safe_load(raw)
            seen.extend(e['key'] for e in page['entries'])
            if 'next' not in page:
                break
            offset = page['next']['offset']
        self.assertEqual(seen, [f'R{i}' for i in range(1, 30)])

    def test_evidence_reference_cannot_cross_task_and_old_ref_survives_retry(self):
        payload = self.build()
        self.harness.sessions.start('s', 't2', str(self.repo), 'Another task')
        self.harness.sessions.stop('s', 't2', 'Done')
        other = checkpoint_task(self.harness)['task_id']
        with self.assertRaises(HarnessError):
            self.harness.build_evidence_groups(other, read_ref=payload['read_ref'])
        self.assertEqual(self.read(payload['read_ref']), payload)

    def test_concurrent_checklist_update_does_not_overwrite(self):
        self.select()
        first, stale = self.harness.store.task(self.task), self.harness.store.task(self.task)
        self.harness.store.put_task(first)
        with self.assertRaisesRegex(HarnessError, 'concurrently'):
            self.harness.store.put_task(stale)

    def test_small_budget_keeps_checklist_reachable_and_references_intact(self):
        self.code.write_text(CODE.replace('range(3)', 'range(4)'), encoding='utf-8')
        full = self.build()
        self.harness.token_budget = 256
        raw = self.harness.build_evidence_groups(self.task)
        self.assertLessEqual(tokens(raw), 256)
        page = yaml.safe_load(raw)
        root = page.get('root_ref', page['read_ref'])
        checklist = self.read(child_ref(root, 'questions'))
        self.assertEqual(checklist['R1'], self.requirements[0]['check'])
        linked = self.read(child_ref(full['read_ref'], 'changes'))
        self.assertEqual(linked['content']['read_ref'], full['changes']['read_ref'])

    def test_diff_marks_missing_final_newline(self):
        self.code.write_text('VALUE = 9', encoding='utf-8')
        self.build()
        view = self.harness.store.latest_view(self.task)
        diff = self.harness.material(view['view_id'], 'diff:src/retry.py')['content']
        self.assertIn('+VALUE = 9\n\\ No newline at end of file\n', diff)
