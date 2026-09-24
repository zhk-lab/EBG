from __future__ import annotations

import json
import unittest

import yaml

from codex_harness import Harness
from tests.support import ProjectTemporaryDirectory, checkpoint_task


class ResearchContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = ProjectTemporaryDirectory()
        self.root = self.temp.__enter__()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.plan = 'Compare the delivered model with baseline under fixed training conditions.'
        (self.repo / 'PLAN.md').write_text(self.plan, encoding='utf-8')
        self.harness = Harness(self.root / 'state')

    def tearDown(self):
        self.temp.__exit__(None, None, None)

    def record(self, name, value):
        (self.repo / name).write_text(json.dumps(value, indent=2), encoding='utf-8')

    def build(self):
        selected = checkpoint_task(self.harness, repo=self.repo)
        plan = next(s['id'] for s in selected['sources'] if s['kind'] == 'plan')
        self.task = selected['task_id']
        payload = yaml.safe_load(self.harness.build_evidence_groups(self.task, [{
            'id': 'R1', 'check': self.plan,
            'refs': [{'source_id': plan, 'quote': self.plan}],
        }]))
        return payload

    def experiment(self, identifier='run-a', **changes):
        return {'record_type': 'experiment', 'experiment_id': identifier,
                'revision': 'rev-a', 'metric': {'name': 'loss', 'value': .4},
                'parameters': {'steps': 50}, 'training_row_ids': ['a', 'b'],
                'evaluation_row_ids': ['c', 'd'], **changes}

    def claim(self, **changes):
        return {'record_type': 'claim', 'experiment_id': 'run-a',
                'revision': 'rev-a', 'metric': {'name': 'loss', 'value': .4},
                'conditions': {'steps': 50}, **changes}

    def test_pairs_claim_with_receipt_and_preserves_frozen_source(self):
        self.record('receipt.json', self.experiment(parameters={'steps': 500},
                    training_row_ids=['a', 'c', 'c']))
        self.record('claim.json', self.claim(revision='rev-b'))
        payload = self.build()
        item = payload['research_context']['claims'][0]
        self.assertEqual(item['candidate']['status'], 'unique')
        comparisons = {c['field']: c for c in item['claim_comparisons']}
        self.assertFalse(comparisons['revision']['equal'])
        self.assertFalse(comparisons['conditions.steps']['equal'])
        receipt = item['candidate']['record']
        self.assertEqual(receipt['row_id_overlap']['count'], 1)
        self.assertEqual(receipt['row_id_overlap']['examples'], ['c'])
        self.record('receipt.json', self.experiment())
        raw = yaml.safe_load(self.harness.build_evidence_groups(
            self.task, read_ref=receipt['read_ref']))
        self.assertIn('500', raw['content'])
        self.assertNotIn('verdict', json.dumps(item))

    def test_duplicate_ids_remain_ambiguous_not_first_match(self):
        self.record('a.json', self.experiment())
        self.record('b.json', self.experiment(revision='rev-b'))
        self.record('claim.json', self.claim())
        item = self.build()['research_context']['claims'][0]
        self.assertEqual(item['candidate']['status'], 'ambiguous')
        self.assertEqual(len(item['candidate']['records']), 2)
        self.assertNotIn('claim_comparisons', item)

    def test_valid_change_disjoint_rows_and_missing_information(self):
        self.record('base.json', self.experiment('base', parameters={'steps': 50, 'rate': .1}))
        self.record('candidate.json', self.experiment(parameters={'steps': 50, 'rate': .2}))
        self.record('claim.json', self.claim(baseline_experiment_id='base', conditions={'steps': 50, 'seed': 7}))
        item = self.build()['research_context']['claims'][0]
        self.assertEqual(item['candidate']['record']['row_id_overlap']['count'], 0)
        comparisons = {c['field']: c for c in item['claim_comparisons']}
        self.assertEqual(comparisons['conditions.seed']['status'], 'missing_in_record')
        self.assertTrue(comparisons['conditions.steps']['equal'])
        changed = [c for c in item['baseline_comparisons'] if c.get('equal') is False]
        self.assertEqual([c['field'] for c in changed], ['parameters.rate'])

    def test_no_claim_still_indexes_experiments_without_inventing_one(self):
        self.record('receipt.json', self.experiment(training_row_ids=None))
        context = self.build()['research_context']
        self.assertEqual(context['claims'], [])
        self.assertEqual(context['experiments'][0]['row_id_overlap']['status'], 'unavailable')

    def test_jsonl_and_pagination_keep_context_reachable(self):
        (self.repo / 'receipts.jsonl').write_text(json.dumps(self.experiment()) + '\n', encoding='utf-8')
        self.record('claim.json', self.claim())
        full = self.build()
        self.harness.token_budget = 1000
        page = yaml.safe_load(self.harness.build_evidence_groups(self.task, read_ref=full['read_ref']))
        self.assertIn('research_context', page)
        context = page['research_context']
        if 'read_ref' in context:
            self.assertEqual(self.harness.pages.resolve('evidence:' + self.task, context['read_ref']),
                             full['research_context'])
        self.assertIn('@1', full['research_context']['claims'][0]['candidate']['record']['source'])

    def test_unrelated_json_is_not_reinterpreted(self):
        self.record('metadata.json', {'metric': .4, 'parameters': {'steps': 50}})
        self.assertNotIn('research_context', self.build())

    def test_population_change_with_same_path_counts_duplicates(self):
        self.record('base.json', self.experiment('base', evaluation_source='data/eval.csv',
                    evaluation_row_ids=['a', 'b', 'b', 'c']))
        self.record('candidate.json', self.experiment(evaluation_source='data/eval.csv',
                    evaluation_row_ids=['b', 'a']))
        self.record('claim.json', self.claim(baseline_experiment_id='base'))
        item = self.build()['research_context']['claims'][0]
        population = item['evaluation_population']
        self.assertFalse(population['equal'])
        self.assertEqual(population['removed_count'], 2)
        self.assertEqual(population['added_count'], 0)
        self.assertEqual(population['removed_examples'], ['b', 'c'])

    def test_population_reordering_is_equal_but_missing_ids_are_unknown(self):
        self.record('base.json', self.experiment('base', evaluation_row_ids=['b', 'c']))
        self.record('candidate.json', self.experiment(evaluation_row_ids=['c', 'b']))
        self.record('claim.json', self.claim(baseline_experiment_id='base'))
        self.assertTrue(self.build()['research_context']['claims'][0]['evaluation_population']['equal'])
        self.record('candidate.json', self.experiment(evaluation_row_ids=None))
        self.assertEqual(self.build()['research_context']['claims'][0]['evaluation_population']['status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
