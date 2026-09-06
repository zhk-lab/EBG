from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from tests.support import ProjectTemporaryDirectory


class PortabilityTests(unittest.TestCase):
    def test_copied_folder_runs_when_original_project_imports_are_blocked(self):
        source = Path(__file__).resolve().parents[1]
        with ProjectTemporaryDirectory() as root:
            isolated = root / "isolated"
            shutil.copytree(source, isolated, ignore=shutil.ignore_patterns(
                ".tmp-tests", ".state", ".venv", "__pycache__", "build", "dist", "*.egg-info",
            ))
            program = """
import importlib.abc
import json
import sys
from pathlib import Path

class BlockOriginalProject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'beg', 'agentloop', 'evaluation_core', 'tracereview', 'tests'}:
            raise ImportError('Original project import is unavailable: ' + fullname)

sys.meta_path.insert(0, BlockOriginalProject())
sys.path.insert(0, str(Path.cwd() / 'src'))
from codex_harness.tools.mcp_server import create_server
from codex_harness import Harness
from codex_harness.hooks.lifecycle import handle_hook
import yaml
repo = Path('fixture')
repo.mkdir()
(repo / 'app.py').write_text('def run():\\n    return 1\\n', encoding='utf-8')
harness = Harness('state')
base = dict(session_id='s', turn_id='t', cwd=str(repo.resolve()))
handle_hook(harness, dict(base, hook_event_name='UserPromptSubmit', prompt='Change app.py run.'))
handle_hook(harness, dict(base, hook_event_name='Stop', last_assistant_message='Done'))
assert yaml.safe_load(harness.list_task_sources())['prompts'][0]['id'] == 'P1'
context = yaml.safe_load(harness.select_task('P1', 'P1'))
requirements = [{'id':'R1','check':'Change app.py run.', 'refs':[{'source_id':'P1','quote':'Change app.py run.'}]}]
result = yaml.safe_load(harness.build_evidence_groups(context['task_id'], requirements))
assert 'R1' in result['evidence_groups']
assert Harness('state').refresh_task(context['task_id'])['reused']
for name, module in list(sys.modules.items()):
    if name.startswith('codex_harness') and getattr(module, '__file__', None):
        assert Path(module.__file__).resolve().is_relative_to(Path.cwd()), name
print(json.dumps(result))
"""
            completed = subprocess.run([sys.executable, "-I", "-c", program], cwd=isolated,
                                       capture_output=True, text=True, encoding="utf-8", timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("R1", json.loads(completed.stdout)["evidence_groups"])
