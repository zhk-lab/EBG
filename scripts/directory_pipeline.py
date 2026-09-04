"""Build and validate module-five Ranked Directory artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import tiktoken

from beg.behavior_directory import (
    DIRECTORY_ENCODING,
    build_ranked_directory,
    render_ranked_directory,
    validate_ranked_directory,
)
from beg.core.errors import DirectoryError
from beg.evidence_intake import load_visible_bundle

from .graph_io import atomic_write, canonical_json_bytes, load_json


JSON_OUTPUT = "ranked_directory.json"
TEXT_OUTPUT = "ranked_directory.txt"


def token_counter(encoding_name: str = DIRECTORY_ENCODING) -> Callable[[str], int]:
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except ValueError as error:
        raise DirectoryError(f"unknown tiktoken encoding: {encoding_name}") from error

    def count(text: str) -> int:
        return len(encoding.encode(text, disallowed_special=()))

    return count


def build_directory_artifact(
    bundle_root: str | Path,
    graph_source: str | Path,
    output_root: str | Path,
    *,
    encoding_name: str = DIRECTORY_ENCODING,
) -> dict[str, Any]:
    bundle = load_visible_bundle(bundle_root)
    graph = _load_graph(graph_source)
    directory = build_ranked_directory(
        bundle, graph, count_tokens=token_counter(encoding_name)
    )
    rendered = render_ranked_directory(directory)
    output = Path(output_root)
    changed: list[str] = []
    if atomic_write(output / JSON_OUTPUT, canonical_json_bytes(directory)):
        changed.append(JSON_OUTPUT)
    if atomic_write(output / TEXT_OUTPUT, rendered.encode("utf-8")):
        changed.append(TEXT_OUTPUT)
    validate_directory_artifact(
        bundle.root,
        graph_source,
        output,
        encoding_name=encoding_name,
    )
    return _summary(directory, output, changed)


def validate_directory_artifact(
    bundle_root: str | Path,
    graph_source: str | Path,
    output_root: str | Path,
    *,
    encoding_name: str = DIRECTORY_ENCODING,
) -> dict[str, Any]:
    bundle = load_visible_bundle(bundle_root)
    graph = _load_graph(graph_source)
    output = Path(output_root)
    json_path = output / JSON_OUTPUT
    text_path = output / TEXT_OUTPUT
    if not json_path.is_file() or not text_path.is_file():
        missing = JSON_OUTPUT if not json_path.is_file() else TEXT_OUTPUT
        raise DirectoryError(f"missing generated artifact: {missing}")
    directory = load_json(json_path)
    if not isinstance(directory, dict):
        raise DirectoryError("ranked_directory.json must contain an object")
    counter = token_counter(encoding_name)
    validate_ranked_directory(bundle, graph, directory, count_tokens=counter)
    expected_text = render_ranked_directory(directory).encode("utf-8")
    if text_path.read_bytes() != expected_text:
        raise DirectoryError("ranked_directory.txt does not match the structured index")
    return _summary(directory, output, [])


def _load_graph(source: str | Path) -> dict[str, Any]:
    path = Path(source)
    if path.is_dir():
        path = path / "behavior_graph.json"
    if not path.is_file():
        raise DirectoryError(f"missing module-four graph: {path}")
    graph = load_json(path)
    if not isinstance(graph, dict):
        raise DirectoryError("module-four graph must contain an object")
    return graph


def _summary(
    directory: dict[str, Any], output: Path, changed: list[str]
) -> dict[str, Any]:
    return {
        "input_id": directory["input_id"],
        "benchmark": directory["benchmark"],
        "directory_type": directory["directory_type"],
        "entries": len(directory["entries"]),
        "token_count": directory["token_count"],
        "output": str(output.resolve()),
        "changed": changed,
        "valid": True,
    }
