"""Build graph artifacts and Repo directories with checkpointed parallelism."""

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
from scripts.graph_io import atomic_write, canonical_json_bytes
from scripts.graph_pipeline import build_graph_artifacts


def _build_one(task: tuple[str, str, str | None]) -> dict[str, Any]:
    bundle, graph_output, directory_output = task
    graph = build_graph_artifacts(bundle, graph_output)
    directory_changed: list[str] = []
    if directory_output is not None:
        directory = build_directory_artifact(bundle, graph_output, directory_output)
        directory_changed = directory["changed"]
    return {
        "input_id": graph["input_id"],
        "benchmark": graph["benchmark"],
        "graph_changed": graph["changed"],
        "directory_changed": directory_changed,
    }


def _load_checkpoint(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("invalid artifact checkpoint")
    completed = value.get("completed")
    if not isinstance(completed, list):
        raise ValueError("invalid artifact checkpoint")
    return {str(item) for item in completed}


def _save_checkpoint(path: Path | None, completed: set[str]) -> None:
    if path is None:
        return
    atomic_write(
        path,
        canonical_json_bytes(
            {"schema_version": 1, "completed": sorted(completed)}
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="beg-repo-artifact-batch")
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=("specgap", "silentswap", "feedbacktrace"),
    )
    parser.add_argument("--evaluation-root", type=Path, default=PROJECT_ROOT / "evaluation")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")

    benchmarks = args.benchmark or ["specgap", "silentswap"]
    completed = _load_checkpoint(args.checkpoint)
    tasks: list[tuple[str, str, str | None]] = []
    for benchmark in benchmarks:
        artifact_root = args.evaluation_root / benchmark / "artifacts"
        visible_root = artifact_root / "visible_bundles"
        for bundle in sorted(path for path in visible_root.iterdir() if path.is_dir()):
            key = f"{benchmark}/{bundle.name}"
            if key in completed:
                continue
            tasks.append(
                (
                    str(bundle),
                    str(artifact_root / "behavior_graphs" / bundle.name),
                    (
                        None
                        if benchmark == "feedbacktrace"
                        else str(
                            artifact_root
                            / "behavior_directories"
                            / bundle.name
                        )
                    ),
                )
            )

    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_tasks = {executor.submit(_build_one, task): task for task in tasks}
        for future in as_completed(future_tasks):
            task = future_tasks[future]
            input_id = Path(task[0]).name
            benchmark = Path(task[0]).parent.parent.parent.name
            try:
                result = future.result()
            except Exception as error:
                failure = {
                    "input_id": input_id,
                    "error": f"{type(error).__name__}: {error}",
                }
                failures.append(failure)
                print(json.dumps({"event": "failed", **failure}, ensure_ascii=False), flush=True)
                continue
            completed.add(f"{result['benchmark']}/{result['input_id']}")
            _save_checkpoint(args.checkpoint, completed)
            print(
                json.dumps(
                    {
                        "event": "completed",
                        "input_id": result["input_id"],
                        "benchmark": result["benchmark"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    print(
        json.dumps(
            {
                "event": "summary",
                "requested": len(tasks),
                "completed_total": len(completed),
                "failed": len(failures),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
