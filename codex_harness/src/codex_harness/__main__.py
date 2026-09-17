"""MCP server, event recorder and local integration setup."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .application.service import Harness
from .application.storage import HarnessError
from .integration import write_integration


def main() -> int:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='EBG disclosure harness')
    parser.add_argument('--state-dir', default='.state/runtime')
    parser.add_argument('--token-budget', type=int, default=12_000)
    parser.add_argument('--review-call-threshold', type=int, default=10)
    parser.add_argument('--review-seconds-threshold', type=float, default=300)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('serve')
    sub.add_parser('hook')
    setup = sub.add_parser('setup')
    setup.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'setup':
            write_integration(Path(args.output), Path(args.state_dir), sys.executable,
                              token_budget=args.token_budget, review_call_threshold=args.review_call_threshold,
                              review_seconds_threshold=args.review_seconds_threshold)
            print(f'Integration files written to {Path(args.output).resolve()}')
            return 0
        if args.command == 'hook' and (Path(args.state_dir) / 'hooks.disabled').is_file():
            # Already-loaded Codex hooks can outlive removal of hooks.json.
            # Do not initialize storage or record events while locally disabled.
            print('{}')
            return 0
        harness = Harness(args.state_dir, token_budget=args.token_budget,
                          review_call_threshold=args.review_call_threshold,
                          review_seconds_threshold=args.review_seconds_threshold)
        if args.command == 'serve':
            from .tools.mcp_server import create_server
            create_server(harness).run(transport='stdio')
        else:
            from .hooks.lifecycle import handle_hook
            print(json.dumps(handle_hook(harness, json.load(sys.stdin)), ensure_ascii=False))
        return 0
    except (HarnessError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f'EBG: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
