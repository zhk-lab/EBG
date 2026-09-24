from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = PROJECT_ROOT / ".tmp-tests"


class ProjectTemporaryDirectory:
    def __enter__(self) -> Path:
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        return Path(self._temporary.name)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._temporary.cleanup()


def make_repo_bundle(
    root: Path,
    *,
    input_id: str = "sg_test",
    benchmark: str = "specgap",
    repository_files: dict[str, str],
    document: str = "# Complete task document\n\nRequired behavior.\n",
) -> Path:
    bundle = root / input_id
    document_name = (
        "3_document_after.md" if benchmark == "specgap" else "original_document.md"
    )
    document_path = bundle / "documents" / document_name
    document_path.parent.mkdir(parents=True, exist_ok=True)
    document_path.write_text(document, encoding="utf-8", newline="")
    visible = [f"documents/{document_name}"]
    for relative, content in repository_files.items():
        path = bundle / "repository" / Path(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")
        visible.append(f"repository/{relative}")
    manifest = {
        "input_id": input_id,
        "benchmark": benchmark,
        "visible_files": sorted(visible),
    }
    (bundle / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="",
    )
    return bundle


def make_trace_bundle(
    root: Path,
    events: list[dict[str, Any]],
    *,
    input_id: str = "ft_test_long",
    cutoff: int = 100,
) -> Path:
    bundle = root / input_id
    trace_path = bundle / "trace" / "model_input.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps({"input_id": input_id, "events": events}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
        newline="",
    )
    (bundle / "input_manifest.json").write_text(
        json.dumps(
            {
                "input_id": input_id,
                "benchmark": "feedbacktrace",
                "visible_files": ["trace/model_input.json"],
                "cutoff_turn": cutoff,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="",
    )
    return bundle
