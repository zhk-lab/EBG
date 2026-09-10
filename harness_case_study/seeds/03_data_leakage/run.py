"""Execute one configured experiment and persist its actual inputs."""

import argparse
import json
from pathlib import Path

from dataset import training_rows, validation_rows
from model import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    reference = training_rows(config)
    validation = validation_rows()
    correct = sum(predict(reference, row['features'], config) == row['label'] for row in validation)
    result = {'config': config, 'config_path': args.config, 'correct': correct,
              'total': len(validation), 'accuracy': correct / len(validation),
              'training_row_ids': [row['id'] for row in reference],
              'evaluation_row_ids': [row['id'] for row in validation]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'accuracy': result['accuracy'], 'correct': correct,
                      'total': len(validation), 'record': args.output}))


if __name__ == '__main__':
    main()
