"""Command-line interface for module-five Ranked Directory construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from beg.behavior_directory import DIRECTORY_ENCODING

from .directory_pipeline import build_directory_artifact, validate_directory_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="beg-directory")
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoding", default=DIRECTORY_ENCODING)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    operation = (
        build_directory_artifact
        if args.command == "build"
        else validate_directory_artifact
    )
    result = operation(
        args.bundle,
        args.graph,
        args.output,
        encoding_name=args.encoding,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
