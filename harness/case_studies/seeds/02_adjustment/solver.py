"""Budgeted search on shifted Rastrigin objectives."""
import math


def objective(point, shift):
    return sum((x - s) ** 2 + 10 * (1 - math.cos(2 * math.pi * (x - s)))
               for x, s in zip(point, shift))


def solve(job, rng, method):
    best = None
    best_value = float('inf')
    for step in range(job['evaluations']):
        if method == 'refine' and step % 4 == 3:
            point = best.copy()
            axis = rng.randrange(len(point))
            point[axis] = max(-5.12, min(5.12, point[axis] + rng.uniform(-0.2, 0.2)))
        else:
            point = [rng.uniform(-5.12, 5.12) for _ in job['shift']]
        value = objective(point, job['shift'])
        if value < best_value:
            best, best_value = point, value
    return {'id': job['id'], 'objective': best_value,
            'evaluations': job['evaluations']}
