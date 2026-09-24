"""Bundle existing SpecGAP reference annotations into standalone hidden Gold."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.specgap.judge.judge import _formal_sample_root, load_gold_conditions


def bundle_gold(gold, references):
    if 'formal_reference' in gold:
        load_gold_conditions(gold)
        return gold
    sample = _formal_sample_root(gold, references)
    parts = json.loads((sample / '2_deleted_parts.json').read_text(encoding='utf-8'))
    mappings = json.loads((sample / '4_code_mapping.json').read_text(encoding='utf-8'))
    bundled = {**gold, 'formal_reference': {
        'deleted_parts': parts['deleted_parts'], 'code_mappings': mappings['code_mappings'],
    }}
    # The scorer verifies selected IDs and source text against the frozen Gold.
    if load_gold_conditions(bundled) != load_gold_conditions(gold, formal_data_root=references):
        raise ValueError(f'Reference annotations changed for {gold["input_id"]}')
    return bundled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Existing hidden_gold directory')
    parser.add_argument('--references', type=Path, required=True, help='Original SpecGAP reference collection')
    parser.add_argument('--output', type=Path, required=True, help='New hidden_gold directory')
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error('Choose a separate output directory to retain the source annotations')
    paths = sorted(args.source.glob('sg_*.json'))
    if not paths:
        parser.error('No SpecGAP Gold found')
    args.output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        value = bundle_gold(json.loads(path.read_text(encoding='utf-8')), args.references)
        output = args.output / path.name
        if output.exists():
            if json.loads(output.read_text(encoding='utf-8')) != value:
                raise ValueError(f'Existing output differs: {output}')
            continue
        temporary = output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(output)
    print(f'Packaged {len(paths)} standalone Gold records in {args.output}')


if __name__ == '__main__':
    main()
