from __future__ import annotations

import unittest

import yaml

from codex_harness import Harness
from codex_harness.application.storage import HarnessError
from tests.support import ProjectTemporaryDirectory, checkpoint_task


SCORER = '''LIMIT = 3

def assess(predictions, data):
    actual = data["targets"][LIMIT:]
    quality_loss = sum(abs(a - b) for a, b in zip(actual, predictions))
    return quality_loss

def unrelated():
    return "unrelated payload"
'''

TRAINER = '''from scoring import assess
from optimizer import search

def run(data):
    initial = fit_training(data)

    def objective(parameters):
        coefficients = parameters[2:]
        predictions = predict(coefficients, data)
        return assess(predictions, data)

    result = search(objective, initial)
    return result

def unrelated():
    return "unrelated payload"
'''


class EvidenceSelectionTests(unittest.TestCase):
    def test_complete_one_hop_is_grouped_and_does_not_expand_neighbors(self):
        source = 'def launch():\n    return ' + ' + '.join(f'helper{i}()' for i in range(7)) + '\n\n'
        source += '\n\n'.join(f'def helper{i}():\n    return ' + ('outside()' if i == 0 else str(i))
                              for i in range(7))
        source += '\n\ndef outside():\n    return "outside second hop"\n'
        (self.repo / 'app.py').write_text(source, encoding='utf-8')
        _, result = self.build(['Inspect `app.py` and `launch`.'])
        entries = result['evidence_groups']['R1']['actual']['repo']
        self.assertEqual(entries[0]['node'], 'app.py::launch')
        self.assertEqual({e['component'] for e in entries}, {'C1'})
        self.assertEqual({e['root'] for e in entries}, {'app.py::launch'})
        self.assertEqual({e['node'] for e in entries}, {'app.py::launch', *(f'app.py::helper{i}' for i in range(7))})
        calls = [r for r in entries[0]['relations'] if r['type'] == 'calls']
        self.assertEqual(len(calls), 7)
        self.assertTrue(all(r['from'] == 'app.py::launch' for r in calls))
        self.assertNotIn('outside second hop', str(entries))

    def test_shared_neighbor_degree_root_and_original_document_tie_order(self):
        (self.repo / 'app.py').write_text(
            'def first():\n    return shared()\n\ndef second():\n    return shared() + extra()\n\n'
            'def shared():\n    return 1\n\ndef extra():\n    return 2\n', encoding='utf-8')
        _, result = self.build(['Inspect `app.py`, `first` and `second`.'])
        entries = result['evidence_groups']['R1']['actual']['repo']
        self.assertEqual(entries[0]['node'], 'app.py::second')
        self.assertEqual({e['node'] for e in entries if e['role'] == 'seed'}, {'app.py::first', 'app.py::second'})
        self.assertEqual(sum('def shared()' in e.get('content', '') for e in entries), 1)
        shared = next(e for e in entries if e['node'] == 'app.py::shared')
        self.assertEqual({r['from'] for r in shared['relations'] if r['type'] == 'calls'},
                         {'app.py::first', 'app.py::second'})

    def test_every_scope_using_explicit_value_is_a_seed(self):
        (self.repo / 'app.py').write_text(
            'def first():\n    return budget_limit\n\ndef second():\n    return budget_limit + 1\n', encoding='utf-8')
        _, result = self.build(['Inspect `app.py` uses of `budget_limit`.'])
        entries = result['evidence_groups']['R1']['actual']['repo']
        self.assertEqual({e['node'] for e in entries if e['role'] == 'seed'}, {'app.py::first', 'app.py::second'})

    def test_root_tie_uses_original_document_order_not_quote_order(self):
        (self.repo / 'app.py').write_text(
            'def first():\n    return shared()\n\ndef second():\n    return shared()\n\n'
            'def shared():\n    return 1\n', encoding='utf-8')
        quotes = ['Inspect `app.py` and `second`.', 'Inspect `app.py` and `first`.']
        task, _ = self.build(quotes)
        source = next(s['id'] for s in self.harness.store.task(task)['sources'] if s['kind'] == 'plan')
        result = yaml.safe_load(self.harness.build_evidence_groups(task, [
            {'id': 'R1', 'check': 'Compare the two functions.',
             'refs': [{'source_id': source, 'quote': quote} for quote in reversed(quotes)]}]))
        entries = result['evidence_groups']['R1']['actual']['repo']
        self.assertEqual(entries[0]['node'], 'app.py::second')
        self.assertEqual(len(entries[0]['document_refs']), 1)

    def test_nested_seeds_keep_identity_and_share_enclosing_source(self):
        task, result = self.build(['Inspect `runner.py`, `run` and `run.objective`.'])
        entries = result['evidence_groups']['R1']['actual']['repo']
        nodes = {entry['node'] for entry in entries if entry.get('role') == 'seed'}
        self.assertTrue({'runner.py::run', 'runner.py::run.objective'} <= nodes)
        contents = '\n'.join(entry.get('content', '') for entry in entries)
        self.assertEqual(contents.count('coefficients = parameters[2:]'), 1)
        shared = next(entry for entry in entries if entry.get('included_in'))
        original = yaml.safe_load(self.harness.build_evidence_groups(task, read_ref=shared['content_ref']))
        self.assertIn('coefficients = parameters[2:]', original['content'])

    def setUp(self):
        self.temp = ProjectTemporaryDirectory()
        self.root = self.temp.__enter__()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        (self.repo / 'scoring.py').write_text(SCORER, encoding='utf-8', newline='')
        (self.repo / 'runner.py').write_text(TRAINER, encoding='utf-8', newline='')
        self.harness = Harness(self.root / 'state')

    def tearDown(self):
        self.temp.__exit__(None, None, None)

    def build(self, quotes):
        (self.repo / 'PLAN.md').write_text('\n'.join(quotes), encoding='utf-8', newline='')
        selected = checkpoint_task(self.harness, repo=self.repo)
        plan = next(s['id'] for s in selected['sources'] if s['kind'] == 'plan')
        task = selected['task_id']
        requirements = [{'id': f'R{i}', 'check': quote, 'refs': [{'source_id': plan, 'quote': quote}]}
                        for i, quote in enumerate(quotes, 1)]
        return task, yaml.safe_load(self.harness.build_evidence_groups(task, requirements))

    def test_metric_value_links_scorer_to_nested_caller_and_enclosing_control(self):
        _, result = self.build(['Keep `quality_loss` unchanged.'])
        entries = result['evidence_groups']['R1']['actual']['repo']
        contents = '\n'.join(entry.get('content', '') for entry in entries)
        self.assertIn('actual = data["targets"][LIMIT:]', contents)
        self.assertIn('coefficients = parameters[2:]', contents)
        self.assertIn('result = search(objective, initial)', contents)
        self.assertNotIn('unrelated payload', contents)
        self.assertTrue(any('run.objective' in entry.get('context', '') for entry in entries))
        self.assertTrue(any('calls' in entry.get('context', '') for entry in entries))

    def test_output_label_is_an_anchor_but_prose_and_docstrings_are_not(self):
        (self.repo / 'scoring.py').write_text('def show(value):\n    print(f"measure_total: {value}")\n', encoding='utf-8')
        (self.repo / 'notes.py').write_text('"""measure_total is mentioned only in documentation."""\n', encoding='utf-8')
        _, result = self.build(['Record `measure_total`.', 'Users read the document.'])
        entries = result['evidence_groups']['R1']['actual']['repo']
        self.assertTrue(any('measure_total:' in entry.get('content', '') for entry in entries))
        self.assertFalse(any('notes.py' in entry['source'] for entry in entries))
        self.assertNotIn('actual', result['evidence_groups']['R2'])

    def test_acronym_matches_code_value_without_changing_ambiguous_symbol_rules(self):
        (self.repo / 'scoring.py').write_text('def assess(x):\n    rmse = x ** 0.5\n    return rmse\n', encoding='utf-8')
        _, result = self.build(['Preserve RMSE.'])
        self.assertIn('rmse =', str(result['evidence_groups']['R1'].get('actual', {})))

    def test_duplicate_file_content_is_readable_by_reference_in_later_group(self):
        task, result = self.build(['Do not edit `scoring.py`.', 'Inspect all of `scoring.py`.'])
        first = result['evidence_groups']['R1']['actual']['repo']
        later = result['evidence_groups']['R2']['actual']['repo']
        self.assertTrue(any(entry.get('content') == SCORER for entry in first))
        shared = next(entry for entry in later if entry.get('content_ref'))
        self.assertNotIn('content', shared)
        (self.repo / 'scoring.py').write_text('changed live content', encoding='utf-8')
        read = yaml.safe_load(self.harness.build_evidence_groups(task, read_ref=shared['content_ref']))
        self.assertEqual(read['content'], SCORER)

    def test_bad_quote_identifies_requirement_and_reference_without_accepting_it(self):
        task, _ = self.build(['Keep `quality_loss` unchanged.'])
        source = next(s['id'] for s in self.harness.store.task(task)['sources'] if s['kind'] == 'plan')
        with self.assertRaisesRegex(HarnessError, r'R2.*refs\[0\].*not verbatim'):
            self.harness.build_evidence_groups(task, [
                {'id': 'R2', 'check': 'A summary', 'refs': [{'source_id': source, 'quote': 'Keep ... unchanged.'}]}])

    def test_short_quote_keeps_explicit_file_on_the_same_source_line(self):
        task, _ = self.build(['`runner.py` is the only editable implementation.',
                              'Other material: `scoring.py`.'])
        source = next(s['id'] for s in self.harness.store.task(task)['sources'] if s['kind'] == 'plan')
        result = yaml.safe_load(self.harness.build_evidence_groups(task, [
            {'id': 'R1', 'check': 'Edit only the permitted implementation.',
             'refs': [{'source_id': source, 'quote': 'the only editable implementation'}]}]))
        entries = result['evidence_groups']['R1']['actual']['repo']
        self.assertTrue(any(TRAINER == e.get('content') for e in entries))
        entry = next(e for e in entries if e.get('content') == TRAINER)
        self.assertIn('runner.py', entry['anchor']['content'])
        self.assertNotIn('scoring.py', entry['anchor']['content'])

    def test_lf_quote_of_crlf_plan_preserves_original_bytes(self):
        task, _ = self.build(['First line.\r\nSecond line.'])
        source = next(s['id'] for s in self.harness.store.task(task)['sources'] if s['kind'] == 'plan')
        result = yaml.safe_load(self.harness.build_evidence_groups(task, [
            {'id': 'R1', 'check': 'Two lines',
             'refs': [{'source_id': source, 'quote': 'First line.\nSecond line.'}]}]))
        self.assertEqual(result['evidence_groups']['R1']['cited_sources']['plan'][0]['content'],
                         'First line.\r\nSecond line.')

    def test_value_anchors_do_not_resolve_ambiguous_definitions(self):
        for path in ('scoring.py', 'runner.py'):
            (self.repo / path).write_text('def run():\n    return 1\n', encoding='utf-8')
        _, result = self.build(['Change `run`.'])
        self.assertNotIn('actual', result['evidence_groups']['R1'])

    def test_comment_and_docstring_acronyms_are_not_value_evidence(self):
        (self.repo / 'scoring.py').write_text(
            '# RMSE is a comment\ndef assess():\n    """RMSE documentation."""\n    return 1\n',
            encoding='utf-8')
        _, result = self.build(['Preserve RMSE.'])
        self.assertNotIn('actual', result['evidence_groups']['R1'])

    def test_containing_excerpts_keep_relations_without_duplicate_bodies(self):
        _, result = self.build(['In `runner.py`, inspect `objective`.'])
        entries = result['evidence_groups']['R1']['actual']['repo']
        contents = '\n'.join(entry.get('content', '') for entry in entries)
        self.assertEqual(contents.count('coefficients = parameters[2:]'), 1)
        self.assertTrue(any('calls' in entry.get('context', '') for entry in entries))

    def test_go_symbols_and_local_calls_are_available_to_ebg(self):
        (self.repo / 'app.go').write_text(
            'package app\n\nfunc helper(value int) int {\n\treturn value + 1\n}\n\n'
            'func Run(value int) int {\n\treturn helper(value)\n}\n',
            encoding='utf-8',
        )
        _, result = self.build(['Inspect `app.go` and `Run`.'])
        entries = result['evidence_groups']['R1']['actual']['repo']
        nodes = {entry['node'] for entry in entries}
        self.assertIn('app.go::Run', nodes)
        self.assertIn('app.go::helper', nodes)
        run = next(entry for entry in entries if entry['node'] == 'app.go::Run')
        self.assertTrue(any(relation['type'] == 'calls' and relation['to'] == 'app.go::helper'
                            for relation in run['relations']))

    def test_typescript_and_javascript_function_symbols_are_available_to_ebg(self):
        (self.repo / 'app.ts').write_text(
            'export function helper(value: number): number {\n'
            '  return value + 1\n}\n\n'
            'export const tsRun = (value: number) => helper(value)\n',
            encoding='utf-8',
        )
        (self.repo / 'util.js').write_text(
            'function helper(value) { return value + 1; }\n'
            'class Runner {\n'
            '  execute(value) { return helper(value); }\n'
            '}\n'
            'function jsRun(value) { return helper(value); }\n',
            encoding='utf-8',
        )
        _, result = self.build(['Inspect `app.ts`, `tsRun`, and `util.js`, `Runner.execute`, `jsRun`.'])
        entries = result['evidence_groups']['R1']['actual']['repo']
        nodes = {entry['node'] for entry in entries}
        self.assertIn('app.ts::tsRun', nodes)
        self.assertIn('util.js::Runner.execute', nodes)
        self.assertIn('util.js::jsRun', nodes)
        ts_run = next(entry for entry in entries if entry['node'] == 'app.ts::tsRun')
        js_run = next(entry for entry in entries if entry['node'] == 'util.js::jsRun')
        method = next(entry for entry in entries if entry['node'] == 'util.js::Runner.execute')
        self.assertTrue(any(relation['type'] == 'calls' for relation in ts_run['relations']))
        self.assertTrue(any(relation['type'] == 'calls' for relation in js_run['relations']))
        self.assertTrue(any(relation['type'] == 'calls' for relation in method['relations']))
