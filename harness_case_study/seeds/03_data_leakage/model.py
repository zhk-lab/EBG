"""Distance-weighted reference classifier."""

import math


def predict(reference, features, config):
    weights = config.get('feature_weights', [1.0] * len(features))
    neighbors = sorted(
        (sum(w * (a - b) ** 2 for w, a, b in zip(weights, row['features'], features)), row['label'])
        for row in reference
    )[:config['neighbors']]
    if config.get('distance_weighted', True):
        votes = [0.0, 0.0]
        for distance, label in neighbors:
            votes[label] += 1.0 / (math.sqrt(distance) + 1e-9)
    else:
        votes = [sum(label == value for _, label in neighbors) for value in (0, 1)]
    return int(votes[1] > votes[0])
