"""CLI entry points must start without a custom PYTHONPATH."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


class CliImportTests(unittest.TestCase):
    def test_main_entry_points_start(self):
        root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        for name in ("prepare", "predict", "judge"):
            with self.subTest(command=name):
                result = subprocess.run(
                    [sys.executable, "-m", f"scripts.main.{name}", "--help"],
                    cwd=root, env=env, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_sensitivity_entry_points_start(self):
        root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        for name in ("run.py", "summarize.py"):
            with self.subTest(script=name):
                result = subprocess.run(
                    [sys.executable, str(root / "experiments" /
                     "sensitivity" / "scripts" / name), "--help"],
                    cwd=root, env=env, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
