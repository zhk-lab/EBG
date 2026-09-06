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
    parser = argparse.ArgumentParser(description='BEG disclosure harness')
    parser.add_argument('--state-dir', default='.state/runtime')
    parser.add_argument('--token-budget', type=int, default=12_000)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('serve')
    sub.add_parser('hook')
    setup = sub.add_parser('setup')
    setup.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'setup':
            write_integration(Path(args.output), Path(args.state_dir), sys.executable,
                              token_budget=args.token_budget)
            print(f'Integration files written to {Path(args.output).resolve()}')
            return 0
        harness = Harness(args.state_dir, token_budget=args.token_budget)
        if args.command == 'serve':
            from .tools.mcp_server import create_server
            create_server(harness).run(transport='stdio')
        else:
            from .hooks.lifecycle import handle_hook
            print(json.dumps(handle_hook(harness, json.load(sys.stdin)), ensure_ascii=False))
        return 0
    except (HarnessError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f'BEG: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
