import unittest

import yaml

from codex_harness import Harness
from codex_harness.application.storage import HarnessError
from tests.support import ProjectTemporaryDirectory, checkpoint_task


class HistoryMaterialTests(unittest.TestCase):
    def setUp(self):
        self.temp = ProjectTemporaryDirectory()
        self.root = self.temp.__enter__()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.harness = Harness(self.root / 'state')
        self.code = self.repo / 'train.py'
        self.code.write_text('version = 0\n', encoding='utf-8', newline='')
        for number in (1, 2, 3):
            turn = f'turn-{number}'
            self.harness.sessions.start('session', turn, str(self.repo), 'Keep train.py results comparable.')
            self.code.write_text(f'version = {number}\n', encoding='utf-8', newline='')
            self.harness.sessions.stop('session', turn, f'Finished {number}.')
        selected = checkpoint_task(self.harness)
        self.task = selected['task_id']
        self.requirements = [{'id': 'R1', 'check': 'Keep results comparable.',
                              'refs': [{'source_id': 'P2', 'quote': 'Keep train.py results comparable.'}]}]
        self.harness.build_evidence_groups(self.task, self.requirements)
        self.view = self.harness.store.latest_view(self.task)['view_id']

    def tearDown(self):
        self.temp.__exit__(None, None, None)

    def test_intermediate_version_read_is_frozen_and_scoped(self):
        index = self.harness.material(self.view, 'history:index')
        self.assertEqual([row['prompt'] for row in index], ['P1', 'P2', 'P3'])
        self.assertEqual(index[1]['after']['snapshot'], 'S4')
        self.code.write_text('version = 99\n', encoding='utf-8')
        answer = yaml.safe_load(self.harness.build_evidence_groups(
            self.task, read_ref=f'{self.view}:snapshot_file:S4:train.py'))
        self.assertEqual(answer['content'], 'version = 2\n')
        with self.assertRaisesRegex(HarnessError, 'outside this frozen task'):
            self.harness.material(self.view, 'snapshot_file:S999:train.py')

    def test_history_navigation_survives_paging(self):
        self.harness.token_budget = 900
        result = yaml.safe_load(self.harness.build_evidence_groups(self.task))
        self.assertEqual(result['history_context']['read_ref'], f'{self.view}:history:index')
        directory = self.harness.material(self.view, 'snapshot:S4')
        self.assertEqual(directory['files'][0]['read_ref'], f'{self.view}:snapshot_file:S4:train.py')


if __name__ == '__main__':
    unittest.main()
