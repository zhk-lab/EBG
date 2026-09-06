from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectTemporaryDirectory:
    def __enter__(self) -> Path:
        directory = PROJECT_ROOT / ".state" / "tests"
        directory.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(dir=directory)
        return Path(self._temporary.name)

    def __exit__(self, *args: Any) -> None:
        self._temporary.cleanup()
