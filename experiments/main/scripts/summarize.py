"""Export saved judge means without rerunning prediction or judging."""
import argparse
import csv
import json
from pathlib import Path

METHODS = {'graph': 'EBG', 'raw': 'Base', 'repograph': 'RepoGraph'}


def rows_from_summary(model, path):
    document = json.loads(path.read_text(encoding='utf-8'))
    if 'workflow' in document and 'score_means' in document:
        if document.get('status') != 'complete' or not document['score_means']:
            raise ValueError(f'Incomplete source-review summary: {path}')
        groups = {document['judge_model']: {'silentswap/graph': document}}
    else:
        judgments = document.get('judges')
        if judgments is None and 'judge_model' in document:
            judgments = {document['judge_model']: document}
        if not judgments:
            raise ValueError(f'No saved judgments in {path}')
        groups = {judge: record['groups'] for judge, record in judgments.items()}
    for judge, results in groups.items():
        for group, record in results.items():
            benchmark, arm = group.split('/')
            for metric, value in record['score_means'].items():
                yield {
                    'model': model, 'judge': judge, 'benchmark': benchmark,
                    'method': METHODS[arm], 'metric': metric, 'value': value,
                    'selected_samples': record['selected_samples'],
                    'completed_samples': record['completed_samples'],
                    'source': path.as_posix(),
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', action='append', required=True, metavar='MODEL=PATH')
    parser.add_argument('--output', type=Path, default=Path('outputs/analysis/main/scores.csv'))
    args = parser.parse_args()
    rows, identities = [], set()
    for item in args.summary:
        model, separator, filename = item.partition('=')
        if not model or not separator or not filename:
            parser.error('--summary must be MODEL=PATH')
        for row in rows_from_summary(model, Path(filename)):
            identity = tuple(row[k] for k in ('model', 'judge', 'benchmark', 'method', 'metric'))
            if identity in identities:
                parser.error(f'Duplicate metric {identity}; select one source per method')
            identities.add(identity)
            rows.append(row)
    if not rows:
        parser.error('Selected summaries contain no scores')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Exported {len(rows)} scores to {args.output}')


if __name__ == '__main__':
    main()
