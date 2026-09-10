"""Build isolated desktop cases, preserving existing projects and raw fixtures."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

from codex_harness.integration import write_integration

STUDY = Path(__file__).resolve().parent
ROOT = next(p for p in STUDY.parents if (p / 'codex_harness/pyproject.toml').is_file())
DESKTOP = ROOT.parent / 'BEG_autoresearch_cases'
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


def install(case, round_name, harness=True):
    # Only fresh, explicitly named experiment directories; never reset existing desktop projects.
    destination = (DESKTOP / round_name / case).resolve()
    if not destination.is_relative_to(DESKTOP.resolve()) or destination.exists():
        raise ValueError(f'Expected a new experiment directory: {destination}')
    shutil.copytree(STUDY / 'seeds' / case, destination)
    if case == '04_api_verification':
        shutil.copyfile(ROOT / '.tmp/beg-retest/seeds/case2/.env', destination / '.env')
    put(destination, '.gitignore', '.beg-harness/\n.agents/\n.codex/\n__pycache__/\n.env\n')
    if harness:
        write_integration(destination / '.codex', destination / '.beg-harness', sys.executable)
        (destination / '.agents').mkdir()
        shutil.move(str(destination / '.codex/skills'), str(destination / '.agents/skills'))
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--round', default='r1')
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--plain', action='store_true')
    args = parser.parse_args()
    if args.install:
        for case in ([args.case] if args.case else CASES):
            print(install(case, args.round, harness=not args.plain))
    else:
        seeds()
