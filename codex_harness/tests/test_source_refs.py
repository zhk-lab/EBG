from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import yaml

from codex_harness import Harness
from codex_harness.application.repository import construct_graph
from codex_harness.application.source_refs import SourceReader
from codex_harness.application.trace import build_trace
from tests.support import ProjectTemporaryDirectory, checkpoint_task


class SourceReferenceTests(unittest.TestCase):
    def test_paged_node_directory_keeps_component_and_relation_reference(self):
        with ProjectTemporaryDirectory() as root:
            harness = Harness(root / 'state', token_budget=256)
            harness.store.activate_session('s')
            node = {'component': 'C1', 'node': 'app.py::start', 'role': 'seed', 'root': 'app.py::start',
                    'source': 'app.py@1-500', 'content': 'return "large payload"\n' * 500,
                    'relations': [{'from': 'app.py::start', 'to': 'app.py::neighbor', 'type': 'calls'}]}
            ref = harness.pages.save('review:s', [node])
            page = yaml.safe_load(harness.pages.read('review:s', ref))
            entry = page['entries'][0]
            self.assertEqual(entry['component'], 'C1')
            self.assertEqual(entry['node'], 'app.py::start')
            self.assertEqual(harness.pages.resolve('review:s', entry['relations_ref']), node['relations'])

    def test_cached_graph_has_no_bodies_and_reads_exact_frozen_lines_after_restart(self):
        with ProjectTemporaryDirectory() as root:
            repo = root / 'repo'
            repo.mkdir()
            source = 'def first():\r\n    return "frozen 汉字"\r\n\r\ndef second():\r\n    return first()'
            (repo / 'app.py').write_bytes(source.encode('utf-8'))
            harness = Harness(root / 'state')
            selected = checkpoint_task(harness, repo=repo, prompt='Inspect `app.py` and `second`.')
            files = harness.store.task(selected['task_id'])['current_files']
            graph, contexts, _, built = construct_graph(harness.store, repo, files)
            self.assertGreater(built, 0)
            for item in graph['evidence']:
                self.assertNotIn('content', item)
                self.assertIn('source_ref', item)
            for item in [*contexts, *graph['source_contexts']]:
                self.assertNotIn('source', item)
            fragment = harness.store.file(files['app.py'])['fragment']
            self.assertNotIn('frozen 汉字', json.dumps(fragment, ensure_ascii=False))
            expected = [SourceReader(harness.store).read(e['source_ref']) for e in graph['evidence']]
            (repo / 'app.py').write_text('live replacement', encoding='utf-8')
            harness = Harness(root / 'state')
            with patch('codex_harness.application.repository.build_evidence', side_effect=AssertionError('rebuilt')):
                rebuilt, _, _, count = construct_graph(harness.store, repo, files)
            self.assertEqual(count, 0)
            self.assertEqual(graph, rebuilt)
            self.assertEqual(expected, [SourceReader(harness.store).read(e['source_ref']) for e in rebuilt['evidence']])
            self.assertTrue(any('\r\n' in body for body in expected))
            context = next(c for c in contexts if c['symbol'] == 'second')
            self.assertEqual(SourceReader(harness.store).context(context), 'def second():\r\n    return first()')

    def test_saved_output_uses_reference_and_pages_exact_original(self):
        with ProjectTemporaryDirectory() as root:
            harness = Harness(root / 'state', token_budget=256)
            harness.store.activate_session('s')
            content = ''.join(f'line {i} 汉字\r\n' for i in range(300))
            file_id = harness.store.cache_file(str(root), 'large.py', content)
            ref = harness.pages.save('review:s', {'source': 'large.py', 'content': content,
                                                 '_source_ref': {'file_id': file_id, 'lines': [1, 300]}})
            with harness.store.connect() as db:
                saved = db.execute('SELECT data FROM outputs WHERE id=?', (ref,)).fetchone()[0]
            self.assertNotIn('line 299', saved)
            harness = Harness(root / 'state', token_budget=256)
            self.assertEqual(harness.pages.resolve('review:s', ref)['content'], content)
            parts = []
            offset = 0
            while True:
                page = yaml.safe_load(harness.pages.read('review:s', ref + '#/content', offset))
                parts.append(page['content'])
                if 'next' not in page:
                    break
                offset = page['next']['offset']
            self.assertEqual(''.join(parts), content)

    def test_trace_graph_references_original_call_and_result(self):
        events = [
            {'id': 'T1', 'kind': 'tool_call', 'content': '{"command":"pytest"}', 'tool_name': 'Bash',
             'call_id': 'c', 'turn': 1, 'seq': 1},
            {'id': 'T2', 'kind': 'tool_result', 'content': 'passed', 'call_id': 'c', 'turn': 1, 'seq': 2},
        ]
        graph = build_trace(events)
        self.assertEqual(graph['evidence'][0]['event_refs'], ['T1', 'T2'])
        self.assertNotIn('content', graph['evidence'][0])
