from __future__ import annotations

import unittest

from codex_harness.application.table_evidence import table_evidence


class TableEvidenceTests(unittest.TestCase):
    def collect(self, files, **kwargs):
        class Store:
            def file(self, value):
                return {'content': value}
        return table_evidence(Store(), {'view_id': 'V1', 'files': files}, list(files), **kwargs)

    def fixture(self, last='external'):
        return {
            'batch.csv': 'sample_id,job_key\na,j1\nb,j2\nc,j3\n',
            'jobs.csv': 'job_key,source_key\nj1,p1\nj2,p2\nj3,p3\n',
            'catalog.tsv': f'id,partition\np1,internal\np2,internal\np3,{last}\n'.replace(',', '\t'),
        }

    def test_joins_renamed_keys_and_returns_counts_with_original_rows(self):
        result = self.collect(self.fixture())
        entry = next(x for x in result['groups'] if x['from'] == 'batch.csv')
        self.assertEqual(entry['matched_rows'], 3)
        self.assertEqual(entry['unmatched_rows'], 0)
        self.assertEqual(len(entry['joins']), 2)
        groups = {x['value']: x for x in entry['counts']}
        self.assertEqual(groups['internal']['rows'], 2)
        self.assertEqual(groups['external']['rows'], 1)
        self.assertEqual([r['content'] for r in groups['external']['example']],
                         ['c,j3', 'j3,p3', 'p3\texternal'])
        self.assertEqual(groups['external']['example'][0]['source'], 'batch.csv@4')
        self.assertIn('V1:repo:catalog.tsv', str(entry))

    def test_normal_categories_do_not_invent_a_problem(self):
        result = self.collect(self.fixture('internal'))
        entry = next(x for x in result['groups'] if x['from'] == 'batch.csv')
        self.assertEqual([(x['value'], x['rows']) for x in entry['counts']], [('internal', 3)])
        self.assertNotIn('violation', str(result))

    def test_duplicate_keys_and_missing_matches_remain_explicit(self):
        files = self.fixture()
        files['catalog.tsv'] += 'p3\tother\n'
        result = self.collect(files)
        self.assertFalse(result['groups'])
        self.assertIn('catalog.tsv.id', str(result['notes']))
        files = self.fixture()
        files['catalog.tsv'] = 'id\tpartition\np1\tinternal\np2\tinternal\n'
        entry = next(x for x in self.collect(files)['groups'] if x['from'] == 'batch.csv')
        self.assertEqual((entry['matched_rows'], entry['unmatched_rows']), (2, 1))

    def test_limits_and_non_identifier_columns_do_not_create_joins(self):
        files = {'x.csv': 'id,score\na,1\nb,2\n', 'y.csv': 'id,score,kind\nx,1,red\ny,2,blue\n'}
        self.assertFalse(self.collect(files)['groups'])
        limited = self.collect(self.fixture(), max_rows=2)
        self.assertFalse(limited['groups'])
        self.assertTrue(limited['notes'])

    def test_quoted_multiline_rows_keep_physical_source_locations(self):
        files = {'batch.csv': 'id,note\na,"two\nlines"\nb,plain\n',
                 'index.csv': 'id,partition\na,first\nb,second\n'}
        entry = next(x for x in self.collect(files)['groups'] if x['from'] == 'batch.csv')
        examples = {x['value']: x['example'][0] for x in entry['counts']}
        self.assertEqual(examples['first']['source'], 'batch.csv@2-3')
        self.assertEqual(examples['first']['content'], 'a,"two\nlines"')
        self.assertEqual(examples['second']['source'], 'batch.csv@4')


if __name__ == '__main__':
    unittest.main()
