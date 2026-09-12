from concurrent.futures import ProcessPoolExecutor
import random

from jobs import run_one
from solver import solve


def process(jobs, method, config):
    with ProcessPoolExecutor(max_workers=2) as pool:
        return list(pool.map(lambda job: run_one(job, method), jobs))


def serial(jobs, method, config):
    rng = random.Random(config['seed'])
    return [solve(job, rng, method) for job in jobs]
