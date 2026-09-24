import argparse
import json
from pathlib import Path
from metrics import measure

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    result = dict(threshold=args.threshold, **measure(json.loads(Path('samples.json').read_text()), args.threshold))
    Path('runs').mkdir(exist_ok=True)
    Path('runs', args.name + '.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
