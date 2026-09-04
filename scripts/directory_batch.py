"""Build and validate every Repo Ranked Directory artifact."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.directory_pipeline import build_directory_artifact


def _build_one(task: tuple[str, str, str]) -> dict[str, Any]:
    bundle, graph, output = task
    return build_directory_artifact(bundle, graph, output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="beg-directory-batch")
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=("specgap", "silentswap"),
    )
    parser.add_argument("--evaluation-root", type=Path, default=PROJECT_ROOT / "evaluation")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")

    benchmarks = args.benchmark or ["specgap", "silentswap"]
    tasks: list[tuple[str, str, str]] = []
    for benchmark in benchmarks:
        artifact_root = args.evaluation_root / benchmark / "artifacts"
        visible_root = artifact_root / "visible_bundles"
        for bundle in sorted(path for path in visible_root.iterdir() if path.is_dir()):
            tasks.append(
                (
                    str(bundle),
                    str(artifact_root / "behavior_graphs" / bundle.name),
                    str(artifact_root / "behavior_directories" / bundle.name),
                )
            )

    completed = 0
    changed = 0
    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_tasks = {executor.submit(_build_one, task): task for task in tasks}
        for future in as_completed(future_tasks):
            task = future_tasks[future]
            try:
                result = future.result()
            except Exception as error:
                failure = {
                    "input_id": Path(task[0]).name,
                    "error": f"{type(error).__name__}: {error}",
                }
                failures.append(failure)
                print(json.dumps({"event": "failed", **failure}, ensure_ascii=False), flush=True)
                continue
            completed += 1
            changed += int(bool(result["changed"]))
            print(
                json.dumps(
                    {
                        "event": "completed",
                        "input_id": result["input_id"],
                        "changed": result["changed"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = {
        "samples": len(tasks),
        "completed": completed,
        "changed": changed,
        "failed": len(failures),
    }
    print(json.dumps({"event": "summary", **summary}, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
