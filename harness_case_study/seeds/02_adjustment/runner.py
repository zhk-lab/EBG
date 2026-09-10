"""Dispatch query jobs using the configured local execution resources."""
from concurrent.futures import ThreadPoolExecutor
from jobs import execute, finish, prepare


def run_queries(queries, settings):
    jobs = [prepare(query, settings['top_k']) for query in queries]
    if settings['workers'] == 1:
        return [finish(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=settings['workers']) as pool:
        return list(pool.map(execute, jobs))
