from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from codex_harness.integration import write_integration
from tests.support import ProjectTemporaryDirectory


class IntegrationTests(unittest.TestCase):
    def test_setup_copies_the_skills_linked_workflow_reference(self):
        with ProjectTemporaryDirectory() as root:
            package = root / 'package'
            skill = package / 'skills' / 'beg-disclose'
            (skill / 'references').mkdir(parents=True)
            (skill / 'SKILL.md').write_text(
                'For evidence lookup, read [workflow](references/tool-workflows.md).', encoding='utf-8')
            reference = skill / 'references' / 'tool-workflows.md'
            reference.write_text('Use the original sources and frozen evidence.', encoding='utf-8')
            shared = skill / 'references' / 'review-rules.md'
            shared.write_text('Common evidence and disclosure rules.', encoding='utf-8')
            result_skill = package / 'skills' / 'beg-result-review'
            result_skill.mkdir()
            (result_skill / 'SKILL.md').write_text(
                'Read [rules](../beg-disclose/references/review-rules.md).', encoding='utf-8')
            output = root / 'integration'
            with patch('codex_harness.integration.files', return_value=package):
                write_integration(output, root / 'state', sys.executable)
            installed = output / 'skills' / 'beg-disclose'
            self.assertEqual((installed / 'SKILL.md').read_bytes(), (skill / 'SKILL.md').read_bytes())
            self.assertEqual((installed / 'references' / reference.name).read_bytes(), reference.read_bytes())
            self.assertEqual((installed / 'references' / shared.name).read_bytes(), shared.read_bytes())
            self.assertEqual((output / 'skills/beg-result-review/SKILL.md').read_bytes(),
                             (result_skill / 'SKILL.md').read_bytes())
