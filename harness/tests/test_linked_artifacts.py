from __future__ import annotations

import unittest

import yaml

from codex_harness import Harness
from codex_harness.application.artifacts import linked_artifacts
from codex_harness.application.render import tokens
from codex_harness.hooks.lifecycle import handle_hook
from tests.support import ProjectTemporaryDirectory


class LinkedArtifactsTests(unittest.TestCase):
    def test_follows_manifest_links_from_selected_plan_and_reads_frozen_content(self):
        with ProjectTemporaryDirectory() as root:
            repo = root / 'repo'
            (repo / 'inputs/lineage').mkdir(parents=True)
            (repo / 'PLAN.md').write_text('Compare models. Source manifest: inputs/export.json.')
            (repo / 'inputs/export.json').write_text('{"index": "lineage/map.csv"}')
            (repo / 'inputs/lineage/map.csv').write_text('item,origin\na,b\n')
            (repo / 'unused.json').write_text('{"irrelevant": true}')
            (repo / 'train.py').write_text('def train():\n    return 1\n')
            harness = Harness(root / 'state')
            handle_hook(harness, {'hook_event_name': 'UserPromptSubmit', 'session_id': 's',
                                 'turn_id': 't', 'cwd': str(repo), 'prompt': 'Follow PLAN.md.'})
            context = yaml.safe_load(harness.checks.review(trigger='result', focus='Adopt model.'))
            # The later disk state must not replace the checkpoint evidence.
            (repo / 'inputs/lineage/map.csv').write_text('item,origin\na,changed\n')
            result = yaml.safe_load(harness.checks.evidence(context['check_id'], 'What does train.py do?',
                [{'source_id': 'P1', 'quote': 'Follow PLAN.md.'}]))
            items = {x['path']: x for x in result['linked_artifacts']['items']}
            self.assertIn('inputs/export.json', items)
            mapping = items['inputs/lineage/map.csv']
            self.assertEqual(mapping['from'], 'inputs/export.json')
            self.assertIn('a,b', mapping['content'])
            self.assertNotIn('unused.json', items)
            full = harness.checks.evidence(context['check_id'], read_ref=mapping['read_ref'])
            self.assertIn('a,b', full)
            self.assertNotIn('changed', full)
            harness.token_budget = 1200
            paged = yaml.safe_load(harness.checks.evidence(context['check_id'], 'What does train.py do?',
                [{'source_id': 'P1', 'quote': 'Follow PLAN.md.'}]))
            self.assertIn('linked_artifacts', paged)
            if 'read_ref' in paged['linked_artifacts']:
                expanded = harness.checks.evidence(context['check_id'],
                    read_ref=paged['linked_artifacts']['read_ref'])
                self.assertIn('inputs/export.json', expanded)

    def test_ambiguous_basenames_are_not_resolved_arbitrarily_and_cycles_stop(self):
        class Files:
            def file(self, value):
                return {'content': value}

        view = {'view_id': 'V1', 'files': {'a/x.json': '"b/y.json"',
                'b/y.json': '"a/x.json"', 'b/x.json': '{}', 'unrelated.json': '{}'}}
        result = linked_artifacts(Files(), view, [('', 'Read x.json and a/x.json.')])
        self.assertEqual({x['path'] for x in result['items']}, {'a/x.json', 'b/y.json'})
        self.assertEqual(result['ambiguous_references'][0]['candidates'], ['a/x.json', 'b/x.json'])
        self.assertFalse(result['limited'])

    def test_paged_evidence_keeps_table_relations_visible_and_readable(self):
        with ProjectTemporaryDirectory() as root:
            repo = root / 'repo'
            repo.mkdir()
            (repo / 'PLAN.md').write_text('Inspect train.py and batch.csv, jobs.csv, catalog.csv.')
            (repo / 'train.py').write_text('"""' + 'implementation detail\n' * 5000 + '"""\n')
            (repo / 'batch.csv').write_text('sample_id,job_key\na,j1\nb,j2\nc,j3\n')
            (repo / 'jobs.csv').write_text('job_key,source_key\nj1,p1\nj2,p2\nj3,p3\n')
            (repo / 'catalog.csv').write_text('id,partition\np1,internal\np2,internal\np3,external\n')
            harness = Harness(root / 'state', token_budget=6000)
            handle_hook(harness, {'hook_event_name': 'UserPromptSubmit', 'session_id': 's',
                                 'turn_id': 't', 'cwd': str(repo), 'prompt': 'Follow PLAN.md.'})
            review = yaml.safe_load(harness.checks.review(trigger='result', focus='Report input provenance.'))
            evidence = yaml.safe_load(harness.checks.evidence(review['check_id'], 'Where did the inputs originate?'))
            relations = evidence['linked_artifacts']['table_relations']
            group = next(x for x in relations['groups'] if x['from'] == 'batch.csv')
            self.assertEqual({x['value']: x['rows'] for x in group['counts']},
                             {'internal': 2, 'external': 1})
            self.assertIn('not declared foreign keys', relations['note'])
            full = yaml.safe_load(harness.checks.evidence(review['check_id'], read_ref=relations['read_ref']))
            self.assertEqual(full['groups'], relations['groups'])
            harness.token_budget = 1200
            output = harness.checks.evidence(review['check_id'], read_ref=evidence['read_ref'])
            self.assertLessEqual(tokens(output), harness.token_budget)
            small = yaml.safe_load(output)
            navigation = small['linked_artifacts']['table_relations']
            self.assertEqual(navigation['read_ref'], relations['read_ref'])
            self.assertEqual(navigation['group_count'], len(relations['groups']))

    def test_large_material_is_explicitly_partial_and_expansion_is_bounded(self):
        class Files:
            def file(self, value):
                return {'content': value}

        view = {'view_id': 'V1', 'files': {'large.csv': 'id,value\n' + 'a,1234567890\n' * 1000,
                                         'other.json': '{}'}}
        result = linked_artifacts(Files(), view, [('', 'large.csv other.json')], limit=1)
        self.assertTrue(result['items'][0]['excerpt_only'])
        self.assertEqual(result['items'][0]['read_ref'], 'V1:repo:large.csv')
        self.assertTrue(result['limited'])


if __name__ == '__main__':
    unittest.main()
