from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def checkpoint_task(harness, *, repo=None, prompt='Inspect PLAN.md.', trigger='result'):
    """Freeze a real checkpoint, leaving graph construction to the test."""
    if repo is not None:
        harness.sessions.start('s', 't', str(repo), prompt)
    check = harness.checks.create(trigger, prompt)
    with patch.object(harness, 'build_evidence_groups', return_value=''):
        harness.checks.evidence(check['id'], 'Inspect the recorded evidence.')
    task = harness.store.task('check_' + check['id'])
    return {'task_id': task['id'], 'scope': task['scope'], 'sources': task['sources']}


class ProjectTemporaryDirectory:
    def __enter__(self) -> Path:
        directory = PROJECT_ROOT / ".state" / "tests"
        directory.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(dir=directory)
        return Path(self._temporary.name)

    def __exit__(self, *args: Any) -> None:
        self._temporary.cleanup()
