"""Build isolated case workspaces without modifying the source fixtures."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
WORKSPACES = ROOT / 'outputs' / 'harness' / 'workspaces'
CASES = ('01_ambiguity', '02_adjustment', '03_data_leakage', '04_api_verification', '05_search_budget')


def put(root, path, content):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content.strip() + '\n', encoding='utf-8')


def seeds():
    """Validate checked-in fixtures; do not overwrite reviewed case definitions."""
    for case in CASES:
        for path in (STUDY / 'seeds' / case / 'PLAN.md', STUDY / 'prompts' / (case + '.md')):
            if not path.is_file():
                raise FileNotFoundError(path)


def install(case, round_name, harness=True, *, workspace_root=WORKSPACES, env_file=None):
    arm = 'harness' if harness else 'plain'
    destination = (workspace_root / round_name / arm / case).resolve()
    if not destination.is_relative_to(workspace_root.resolve()) or destination.exists():
        raise ValueError(f'Expected a new experiment directory: {destination}')
    shutil.copytree(STUDY / 'seeds' / case, destination,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    if env_file is not None:
        shutil.copyfile(env_file, destination / '.env')
    put(destination, '.gitignore', '.ebg-harness/\n.agents/\n.codex/\n__pycache__/\n.env\n')
    if harness:
        from codex_harness.integration import write_integration
        write_integration(destination / '.codex', destination / '.ebg-harness', sys.executable)
        (destination / '.agents').mkdir()
        shutil.move(str(destination / '.codex/skills'), str(destination / '.agents/skills'))
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--round', default='r1')
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--plain', action='store_true')
    parser.add_argument('--workspace-root', type=Path, default=WORKSPACES)
    parser.add_argument('--env-file', type=Path, help='Optional local configuration for an API case')
    args = parser.parse_args()
    if args.install:
        for case in ([args.case] if args.case else CASES):
            print(install(case, args.round, harness=not args.plain,
                          workspace_root=args.workspace_root, env_file=args.env_file))
    else:
        seeds()
