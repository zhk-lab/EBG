import json
from pathlib import Path
import tempfile
import unittest

from scripts.export_dataset_preview import make_row, repository_tree


class DatasetPreviewTests(unittest.TestCase):
    def fixture(self, root, benchmark, input_id, files, gold):
        artifacts = root / benchmark / 'artifacts'
        bundle = artifacts / 'visible_bundles' / input_id
        bundle.mkdir(parents=True)
        (bundle / 'input_manifest.json').write_text(json.dumps({
            'input_id': input_id, 'benchmark': benchmark, 'visible_files': list(files)}))
        for name, content in files.items():
            p = bundle / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding='utf-8')
        (artifacts / 'hidden_gold').mkdir()
        (artifacts / 'hidden_gold' / f'{input_id}.json').write_text(json.dumps({
            'input_id': input_id, 'benchmark': benchmark, **gold}))

    def test_document_is_complete_and_evidence_keeps_condition_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root, 'specgap', 'sg_001', {'documents/task.md': 'x' * 5000,
                         'repository/src/a.py': ''}, {'conditions': [{
                'condition_id': 'kc_1', 'normalized_condition': 'Keep defaults.',
                'implementation_locations': [{'file': 'src/a.py'}]}]})
            row = make_row(root, 'specgap', 'sg_001')
            self.assertEqual(len(row['document']), 5000)
            self.assertIn('[kc_1]', row['missing_requirements'])
            self.assertIn('[kc_1]', row['evidence'])
            self.assertIn('a.py', row['repository_tree'])
            self.assertNotIn('task.md', row['repository_tree'])

    def test_changes_merge_expected_and_actual(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root, 'silentswap', 'ss_001', {'documents/task.md': 'Task'},
                         {'swaps': [{'original_semantics': 'Retry.',
                          'swapped_semantics': 'Exit.', 'why_different': 'Fails early.',
                          'evidence': ['code evidence']}]})
            row = make_row(root, 'silentswap', 'ss_001')
            self.assertIn('Expected: Retry.', row['semantic_changes'])
            self.assertIn('Changed: Exit.', row['semantic_changes'])
            self.assertIn('[change_1]', row['evidence'])
            self.assertNotIn('expected_behavior', row)

    def test_trace_evidence_matches_and_target_feedback_is_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events = [{'event_type': 'user_prompt', 'turn_number': 1, 'content': 'x' * 5000},
                      {'event_type': 'assistant_response', 'turn_number': 2,
                       'content': 'Done.', 'evidence_id': 'e_2'}]
            self.fixture(root, 'feedbacktrace', 'ft_001_long',
                         {'trace/model_input.json': json.dumps({'events': events})},
                         {'verification_point': 'Confirm.', 'evidence_ids': ['e_2'],
                          'criticality': 'must_disclose', 'target_user_feedback': 'SECRET_TARGET'})
            row = make_row(root, 'feedbacktrace', 'ft_001_long')
            self.assertGreater(len(row['trace']), 5000)
            self.assertIn('Turn 2 | Assistant | e_2', row['evidence'])
            self.assertNotIn('SECRET_TARGET', str(row))
            self.assertEqual(row['criticality'], 'must_disclose')
            self.assertEqual(list(row), ['id', 'trace', 'verification_point', 'evidence', 'criticality'])

    def test_tree_only_contains_visible_repository_files(self):
        tree = repository_tree(['documents/task.md', 'repository/src/a.py', 'repository/README.md'])
        self.assertEqual(tree, 'repository/\n|-- src/\n|   `-- a.py\n`-- README.md')


if __name__ == '__main__':
    unittest.main()
