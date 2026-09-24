import random

from solver import solve


def make_jobs(config):
    return [{'id': index, 'seed': config['seed'] + index,
             'shift': [0.3 * math_value for math_value in
                       ((index + axis) % 5 - 2 for axis in range(config['dimensions']))],
             'evaluations': config['evaluations']}
            for index in range(config['problems'])]


def run_one(job, method):
    return solve(job, random.Random(job['seed']), method)
