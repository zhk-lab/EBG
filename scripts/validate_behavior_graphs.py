"""Validate existing BEG behavior graph outputs."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.graph_cli import main


if __name__ == "__main__":
    raise SystemExit(main(["validate", *sys.argv[1:]]))
