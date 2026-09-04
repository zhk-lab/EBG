from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.audit_gold_retrieval import _gold_locations, _source_covered


class GoldRetrievalAuditTests(unittest.TestCase):
    def test_repo_gold_formats_become_the_same_location_shape(self) -> None:
        specgap = _gold_locations(
            {
                "benchmark": "specgap",
                "conditions": [
                    {
                        "condition_id": "kc_001",
                        "implementation_locations": [
                            {
                                "file": "src/app.py",
                                "symbol": "Client.run",
                                "line_ranges": [{"start": 4, "end": 7}],
                            }
                        ],
                    }
                ],
            }
        )
        silentswap = _gold_locations(
            {
                "benchmark": "silentswap",
                "swaps": [
                    {
                        "localization": {
                            "file": "src/app.py",
                            "symbol": {
                                "kind": "method",
                                "qualified_name": ["Client", "run"],
                            },
                            "line_ranges": [{"start": 4, "end": 7}],
                        }
                    }
                ],
            }
        )

        self.assertEqual(specgap[0]["path"], silentswap[0]["path"])
        self.assertEqual(specgap[0]["symbol"], silentswap[0]["symbol"])
        self.assertEqual(specgap[0]["line_ranges"], [[4, 7]])

    def test_coverage_requires_every_gold_range_inside_returned_source(self) -> None:
        location = {
            "path": "src/app.py",
            "symbol": "run",
            "line_ranges": [[4, 7], [10, 10]],
        }
        nodes = [
            {"path": "src/app.py", "symbol": "run", "lines": [3, 8]},
            {"path": "src/app.py", "symbol": "helper", "lines": [9, 10]},
        ]

        self.assertTrue(_source_covered(location, nodes))
        self.assertFalse(_source_covered(location, nodes[:1]))


if __name__ == "__main__":
    unittest.main()
