"""Check saved retrieval output consistency."""
from experiment import metrics, read


def verify():
    queries = read('queries.json')
    result = read('runs/evaluation.json')
    assert result['settings'] == read('settings.json')
    rankings = result['rankings']
    assert [row['query_id'] for row in rankings] == [query['id'] for query in queries]
    for query, row in zip(queries, rankings):
        ids = [item['id'] for item in row['items']]
        assert len(ids) == min(result['settings']['top_k'], len(query['candidates']))
        assert len(set(ids)) == len(ids)
        assert set(ids) <= {item['id'] for item in query['candidates']}
        scores = [item['score'] for item in row['items']]
        assert scores == sorted(scores, reverse=True)
    assert result['metrics'] == metrics(queries, rankings)
    print('Verification passed')


if __name__ == '__main__':
    verify()
