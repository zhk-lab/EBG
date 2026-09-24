"""Install the frozen evaluation release from a local download, one sample at a time."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ebg.evidence_intake import load_visible_bundle

BENCHMARKS = ('specgap', 'silentswap', 'feedbacktrace')
SPLIT = ROOT / 'experiments/main/configs/samples.json'


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def prepare_sample(source: Path, destination: Path, benchmark: str, input_id: str,
                   *, resume: bool = False) -> str:
    source_artifacts = source / benchmark / 'artifacts'
    source_bundle = source_artifacts / 'visible_bundles' / input_id
    source_gold = source_artifacts / 'hidden_gold' / f'{input_id}.json'
    bundle = load_visible_bundle(source_bundle)
    gold = read(source_gold)
    if bundle.benchmark != benchmark or gold.get('input_id') != input_id or gold.get('benchmark') != benchmark:
        raise ValueError(f'Mismatched input and gold: {benchmark}/{input_id}')
    if benchmark == 'feedbacktrace' and (gold.get('verdict') != 'KEY' or not input_id.endswith('_long')):
        raise ValueError('The paper release contains only KEY Long FeedbackTrace inputs')
    if benchmark == 'specgap' and gold.get('conditions') and 'formal_reference' not in gold:
        raise ValueError('SpecGAP Gold lacks formal_reference; package its frozen references before release')

    artifacts = destination / benchmark / 'artifacts'
    target = artifacts / 'visible_bundles' / input_id
    target_gold = artifacts / 'hidden_gold' / f'{input_id}.json'
    marker = artifacts / 'preparation' / f'{input_id}.json'
    reference = {'source': str(source.resolve()), 'benchmark': benchmark, 'input_id': input_id}
    if source.resolve() == destination.resolve():
        return 'validated'
    if target.exists():
        if not resume:
            raise FileExistsError(f'{target} exists; use --resume for the same source release')
        load_visible_bundle(target)
        if marker.exists() and read(marker) == reference and target_gold.exists() and read(target_gold) == gold:
            return 'reused'
        # An interrupted copy may have installed the bundle before the gold/marker.
        manifest = read(source_bundle / 'input_manifest.json')
        if read(target / 'input_manifest.json') != manifest or any(
            (target / name).read_bytes() != (source_bundle / name).read_bytes()
            for name in manifest['visible_files']
        ):
            raise ValueError(f'Existing input differs from source: {input_id}; use a fresh destination')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent, prefix='.prepare-') as scratch:
            staged = Path(scratch) / input_id
            staged.mkdir()
            manifest = read(source_bundle / 'input_manifest.json')
            for name in ['input_manifest.json', *manifest['visible_files']]:
                output = staged / name
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_bundle / name, output)
            load_visible_bundle(staged)
            staged.rename(target)
    target_gold.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_gold.with_suffix('.json.tmp')
    shutil.copy2(source_gold, temporary)
    temporary.replace(target_gold)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(reference, indent=2) + '\n', encoding='utf-8')
    return 'prepared'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Unpacked frozen evaluation release')
    parser.add_argument('--destination', type=Path, default=ROOT / 'data/prepared')
    parser.add_argument('--benchmark', choices=BENCHMARKS, action='append')
    parser.add_argument('--input-id', help='Prepare one published input; requires one --benchmark')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.input_id and len(args.benchmark or []) != 1:
        parser.error('--input-id requires exactly one --benchmark')
    selected = read(SPLIT)
    for benchmark in args.benchmark or BENCHMARKS:
        ids = [args.input_id] if args.input_id else selected[benchmark]
        for input_id in ids:
            if input_id not in selected[benchmark]:
                parser.error(f'Input is not in the paper split: {input_id}')
            status = prepare_sample(args.source, args.destination, benchmark, input_id, resume=args.resume)
            print(f'{benchmark}/{input_id}: {status}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
