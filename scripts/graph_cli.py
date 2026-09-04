"""Command-line interface for building and validating BEG artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .graph_pipeline import build_graph_artifacts, validate_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="beg")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "validate"):
        subparser = subparsers.add_parser(command)
        source = subparser.add_mutually_exclusive_group(required=True)
        source.add_argument("--bundle", type=Path)
        source.add_argument("--input-root", type=Path)
        subparser.add_argument("--output", type=Path)
        subparser.add_argument("--output-root", type=Path)
        subparser.add_argument("--input-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pairs = _resolve_pairs(args)
    results = []
    for bundle, output in pairs:
        if args.command == "build":
            results.append(build_graph_artifacts(bundle, output))
        else:
            results.append(validate_output(bundle, output))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def _resolve_pairs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.bundle is not None:
        if args.output is None or args.output_root is not None or args.input_id is not None:
            raise SystemExit("--bundle requires --output and forbids --output-root/--input-id")
        return [(args.bundle, args.output)]
    if args.input_root is None or args.output_root is None or args.output is not None:
        raise SystemExit("--input-root requires --output-root and forbids --output")
    bundles = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    if args.input_id is not None:
        bundles = [path for path in bundles if path.name == args.input_id]
        if not bundles:
            raise SystemExit(f"input_id not found: {args.input_id}")
    if not bundles:
        raise SystemExit("input root contains no bundle directories")
    return [(bundle, args.output_root / bundle.name) for bundle in bundles]


if __name__ == "__main__":
    raise SystemExit(main())
