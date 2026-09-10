"""Run the fixed retrieval evaluation."""
import json
from pathlib import Path
from runner import run_queries


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def metrics(queries, rankings):
    hits = sum(row['items'][0]['id'] == query['relevant'] for query, row in zip(queries, rankings))
    return {'hits_at_1': hits, 'query_count': len(queries), 'precision_at_1': hits / len(queries)}


def run():
    queries, settings = read('queries.json'), read('settings.json')
    rankings = run_queries(queries, settings)
    result = {'settings': settings, 'rankings': rankings, 'metrics': metrics(queries, rankings)}
    Path('runs').mkdir(exist_ok=True)
    Path('runs/evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Evaluation complete:', json.dumps(result['metrics']))


if __name__ == '__main__':
    run()
