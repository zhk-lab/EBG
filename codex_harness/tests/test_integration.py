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
                'For full review, read [workflow](references/tool-workflows.md).', encoding='utf-8')
            reference = skill / 'references' / 'tool-workflows.md'
            reference.write_text('Use the original sources and frozen evidence.', encoding='utf-8')
            output = root / 'integration'
            with patch('codex_harness.integration.files', return_value=package):
                write_integration(output, root / 'state', sys.executable)
            installed = output / 'skills' / 'beg-disclose'
            self.assertEqual((installed / 'SKILL.md').read_bytes(), (skill / 'SKILL.md').read_bytes())
            self.assertEqual((installed / 'references' / reference.name).read_bytes(), reference.read_bytes())
