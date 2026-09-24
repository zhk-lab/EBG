import json
from pathlib import Path
import tempfile
import unittest

from scripts.export_dataset_preview import make_row, PREVIEW_LIMIT


class DatasetPreviewTests(unittest.TestCase):
    def test_preview_uses_only_manifest_document_and_bounds_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / 'specgap/artifacts/visible_bundles/sg_001'
            (bundle / 'documents').mkdir(parents=True)
            (bundle / 'documents/task.md').write_text('x' * (PREVIEW_LIMIT + 10))
            (bundle / 'input_manifest.json').write_text(json.dumps({
                'input_id': 'sg_001', 'benchmark': 'specgap',
                'visible_files': ['documents/task.md'],
            }))
            row = make_row(root, 'specgap', 'sg_001')
            self.assertEqual(len(row['input_preview']), PREVIEW_LIMIT)
            self.assertTrue(row['preview_truncated'])
            self.assertEqual(row['visible_file_count'], 1)
            self.assertEqual(row['archive'], 'specgap.tar.gz')

    def test_feedback_trace_keeps_chronological_visible_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / 'feedbacktrace/artifacts/visible_bundles/ft_001_long'
            (bundle / 'trace').mkdir(parents=True)
            (bundle / 'input_manifest.json').write_text(json.dumps({
                'input_id': 'ft_001_long', 'benchmark': 'feedbacktrace',
                'visible_files': ['trace/model_input.json'],
            }))
            (bundle / 'trace/model_input.json').write_text(json.dumps({'events': [
                {'event_type': 'user_prompt', 'content': 'Inspect the tests.'},
                {'event_type': 'tool_result', 'content': {'passed': True}},
            ]}))
            row = make_row(root, 'feedbacktrace', 'ft_001_long')
            self.assertEqual(row['input_preview'],
                             '[user_prompt] Inspect the tests.\n\n[tool_result] {"passed": true}')
            self.assertFalse(row['preview_truncated'])
            self.assertNotIn('events', row)


if __name__ == '__main__':
    unittest.main()
