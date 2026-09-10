"""Query job construction and execution."""
from model import PairModel


def prepare(query, top_k):
    return {'query_id': query['id'], 'top_k': top_k,
            'items': [{'id': row['id'], 'score': row['retrieval_score'],
                       'pair_features': row['pair_features']} for row in query['candidates']]}


def finish(job):
    items = sorted(job['items'], key=lambda row: row['score'], reverse=True)[:job['top_k']]
    return {'query_id': job['query_id'],
            'items': [{'id': row['id'], 'score': row['score']} for row in items]}


def execute(job):
    model = PairModel()
    scored = {**job, 'items': [{**row, 'score': model.score(row)} for row in job['items']]}
    return finish(scored)
