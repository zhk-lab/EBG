"""CSV adapters for the fixed feature representation."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_rows(name):
    with (ROOT / 'data' / name).open(newline='', encoding='utf-8') as stream:
        return [{'id': row['sample_id'], 'label': int(row['label']),
                 'features': [float(row[f'x{i}']) for i in range(4)]}
                for row in csv.DictReader(stream)]


def training_rows(config):
    reference = read_rows('train.csv')
    if config.get('use_extra', False):
        reference += read_rows('extra.csv')
    return reference


def validation_rows():
    return read_rows('validation.csv')
