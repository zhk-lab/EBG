"""Experiment sample paths and the combined prediction/Judge summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sample_directory(root: Path, area: str, arm: str, input_id: str) -> Path:
    """Only mixed-arm experiments need an arm directory; IDs identify benchmarks."""
    manifest = _read(root / "manifest.json")
    base = root / area
    if len(manifest["arms"]) > 1:
        base /= arm
    return base / input_id


def update_summary(
    root: Path,
    *,
    prediction: dict[str, Any] | None = None,
    judge_model: str | None = None,
    judgment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = root / "summary.json"
    saved = _read(path) if path.exists() else {}
    if saved.get("schema_version") == 2 and "judges" in saved:
        combined = saved
    else:
        combined = {"schema_version": 2, "prediction": saved or None, "judges": {}}
    if prediction is not None:
        combined["prediction"] = prediction
    if judgment is not None:
        if judge_model is None:
            raise ValueError("judge_model is required for a Judge summary")
        combined["judges"][judge_model] = judgment
    _write(path, combined)
    return combined


def migration_moves(root: Path) -> list[tuple[Path, Path]]:
    """Return a checked, collision-free set of legacy directory moves."""
    root = root.resolve()
    manifest = _read(root / "manifest.json")
    moves: list[tuple[Path, Path]] = []
    judge_roots = [p for p in (root / "judges").glob("*") if p.is_dir()]
    for benchmark, ids in manifest.get("selected_ids", {}).items():
        for arm in manifest["arms"]:
            for input_id in ids:
                destination = sample_directory(root, "runs", arm, input_id)
                candidates = [
                    root / "runs" / arm / input_id,
                    root / "runs" / benchmark / arm / input_id,
                ]
                for phase in ("full", "development", "formal"):
                    candidates.extend([
                        root / "runs" / phase / benchmark / arm / input_id,
                        root / "runs" / phase / arm / input_id,
                    ])
                for source in candidates:
                    if source != destination and source.exists():
                        moves.append((source, destination))
                for judge_root in judge_roots:
                    source = judge_root / benchmark / arm / input_id
                    destination = sample_directory(root, f"judges/{judge_root.name}", arm, input_id)
                    if source != destination and source.exists():
                        moves.append((source, destination))

    destinations: set[Path] = set()
    for source, destination in moves:
        # Resolve every directory before moving it; all targets must remain here.
        source.resolve().relative_to(root)
        destination.resolve().relative_to(root)
        if destination.exists() or destination in destinations:
            raise ValueError(f"sample directory collision: {source} -> {destination}")
        destinations.add(destination)
    return moves


def migrate_experiment(root: Path) -> dict[str, int]:
    """Move legacy sample directories in place, preserving responses and retries.

    Call only when no process is writing this experiment. Moves are preflighted and
    completed moves are skipped on restart. No model calls or scoring changes occur.
    """
    root = root.resolve()
    manifest = _read(root / "manifest.json")
    moves = migration_moves(root)
    judge_roots = [p for p in (root / "judges").glob("*") if p.is_dir()]
    for source, destination in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)

    # This local metadata is checked on resume. Model messages/responses stay intact.
    updated_inputs = 0
    for judge_root in judge_roots:
        for path in judge_root.rglob("judge_input.json"):
            saved = _read(path)
            input_id = (
                saved.get("input_id")
                or saved.get("prediction", {}).get("input_id")
                or path.parent.name
            )
            arm = saved.get("arm") or manifest["arms"][0]
            prediction = sample_directory(root, "runs", arm, input_id) / "prediction.json"
            if "prediction_path" in saved and saved["prediction_path"] != str(prediction):
                saved["prediction_path"] = str(prediction)
                _write(path, saved)
                updated_inputs += 1

    summaries = 0
    for judge_root in judge_roots:
        old_summary = judge_root / "summary.json"
        if old_summary.exists():
            judgment = _read(old_summary)
            update_summary(root, judge_model=judge_root.name, judgment=judgment)
            if _read(root / "summary.json")["judges"][judge_root.name] != judgment:
                raise ValueError(f"merged Judge summary differs: {old_summary}")
            old_summary.unlink()
            summaries += 1
    if (root / "summary.json").exists():
        update_summary(root)

    # Remove only the now-empty structural directories, never sample contents.
    for area in [root / "runs", *judge_roots]:
        if not area.exists():
            continue
        structural_names = {
            "full", "development", "formal", "specgap", "silentswap", "feedbacktrace",
            "raw", "graph",
        }
        for directory in sorted(area.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if directory.is_dir() and directory.name in structural_names:
                if not any(directory.iterdir()):
                    directory.rmdir()
    return {"moved_samples": len(moves), "updated_inputs": updated_inputs, "merged_summaries": summaries}
