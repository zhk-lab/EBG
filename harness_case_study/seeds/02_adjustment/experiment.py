import argparse
import json
from pathlib import Path
from statistics import mean

import backends
from jobs import make_jobs


def save_result(path, rows, method, backend, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {'method': method, 'backend': backend, 'config': config,
              'mean_objective': mean(row['objective'] for row in rows), 'rows': rows}
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['random', 'refine'], default='refine')
    parser.add_argument('--backend', choices=['process', 'serial'], default='process')
    parser.add_argument('--output', type=Path, default=Path('results/candidate.json'))
    args = parser.parse_args()
    config = json.loads(Path('config.json').read_text(encoding='utf-8'))
    rows = getattr(backends, args.backend)(make_jobs(config), args.method, config)
    save_result(args.output, rows, args.method, args.backend, config)
