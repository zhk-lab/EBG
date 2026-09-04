from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.audit_repo_backend import _read_token_metrics


class RepoBackendAuditTests(unittest.TestCase):
    def test_read_size_metrics_use_32k_and_64k_boundaries(self) -> None:
        metrics = _read_token_metrics([32_768, 32_769, 65_536, 65_537])

        self.assertEqual(metrics["read_units_over_32k"], 3)
        self.assertEqual(metrics["read_units_over_64k"], 1)
        self.assertNotIn("read_units_over_36k", metrics)


if __name__ == "__main__":
    unittest.main()
