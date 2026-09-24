import tempfile
import unittest
from pathlib import Path

from engine import fit
from experiment import run
from profiles import configurations, select_result


class ExperimentTests(unittest.TestCase):
    def test_selection_retains_best_observation(self):
        records = [{'accuracy': .7}, {'accuracy': .8}, {'accuracy': .75}]
        self.assertIs(select_result(records), records[1])

    def test_profile_expansion(self):
        profile = dict(optimizer='sgd', rates=[.01, .1], decays=[0, .01], momenta=[0])
        self.assertEqual(len(list(configurations(profile))), 4)

    def test_momentum_zero_matches_sgd(self):
        config = dict(optimizer='sgd', rate=.04, decay=.001, momentum=0.)
        self.assertEqual(fit(config)['accuracy'], fit(dict(config, optimizer='momentum'))['accuracy'])

    def test_resume_does_not_duplicate_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = run('reference', output)
            original = (output / 'trials.jsonl').read_text()
            self.assertEqual(run('reference', output), first)
            self.assertEqual((output / 'trials.jsonl').read_text(), original)
