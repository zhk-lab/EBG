"""Build an EBG from a synthetic repository without a model or dataset download."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ebg.evidence_intake import build_evidence, load_visible_bundle
from ebg.behavior_atomization import build_behaviors
from ebg.relation_linking import build_edges
from ebg.graph_assembly import build_graph


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/demo')
    args = parser.parse_args(argv)
    bundle_root = args.output / 'sg_demo'
    (bundle_root / 'repository').mkdir(parents=True, exist_ok=True)
    (bundle_root / 'documents').mkdir(exist_ok=True)
    (bundle_root / 'repository/scoring.py').write_text(
        'def accuracy(correct, total):\n    if total == 0:\n        return 0.0\n    return correct / total\n',
        encoding='utf-8')
    (bundle_root / 'documents/3_document_after.md').write_text(
        '# Scoring\nReturn classification accuracy for the supplied counts.\n', encoding='utf-8')
    (bundle_root / 'input_manifest.json').write_text(json.dumps({
        'input_id': 'sg_demo', 'benchmark': 'specgap',
        'visible_files': ['documents/3_document_after.md', 'repository/scoring.py'],
    }, indent=2) + '\n', encoding='utf-8')
    bundle = load_visible_bundle(bundle_root)
    evidence = build_evidence(bundle)
    behaviors = build_behaviors(bundle, evidence)
    edges = build_edges(bundle, evidence, behaviors)
    graph = build_graph(bundle, evidence, behaviors, edges)
    output = args.output / 'behavior_graph.json'
    output.write_text(json.dumps(graph, indent=2) + '\n', encoding='utf-8')
    print(f'Graph saved to {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
