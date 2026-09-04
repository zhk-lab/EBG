"""Command-line entry point for resumable batch predictions."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.batch_prediction import main


if __name__ == "__main__":
    raise SystemExit(main())
