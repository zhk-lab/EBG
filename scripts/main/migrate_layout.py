"""Migrate stopped experiments to flat sample directories and one root summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.main.layout import migrate_experiment, migration_moves


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=Path(__file__).resolve().parents[2] / "experiments")
    parser.add_argument("--experiment-name", action="append")
    parser.add_argument("--check", action="store_true", help="Check paths and collisions without moving files")
    args = parser.parse_args()
    base = args.experiment_root.resolve()
    if args.experiment_name:
        roots = [(base / name).resolve() for name in args.experiment_name]
        for root in roots:
            root.relative_to(base)
    else:
        roots = []
        for path in sorted(base.rglob("manifest.json")):
            relative = path.relative_to(base)
            if "runs" in relative.parts or "judges" in relative.parts:
                continue
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if "selected_ids" in manifest and "arms" in manifest:
                roots.append(path.parent)
    # Check every experiment before changing any of them.
    plans = [(root, migration_moves(root)) for root in roots]
    for root, moves in plans:
        result = {"moved_samples": len(moves)} if args.check else migrate_experiment(root)
        print(json.dumps({"experiment": str(root.relative_to(base)), **result}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
