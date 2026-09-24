"""Fixed binary classification protocol, trained on CPU."""

import numpy as np

EPOCHS = 24
SEED = 2026


def dataset():
    rng = np.random.default_rng(81)
    features = rng.normal(size=(960, 12))
    coefficients = np.array([1.8, -1.4, .9, -.7, .4, .3, -.2, .2, .1, -.1, .1, .2])
    latent = features @ coefficients + .9 * rng.normal(size=960)
    labels = (latent > 0).astype(float)
    return features[:720], labels[:720], features[720:], labels[720:]


def fit(config):
    train_x, train_y, valid_x, valid_y = dataset()
    rng = np.random.default_rng(SEED)
    weights = rng.normal(scale=.08, size=train_x.shape[1])
    velocity = np.zeros_like(weights)
    bias = 0.
    bias_velocity = 0.
    for _ in range(EPOCHS):
        order = rng.permutation(len(train_y))
        for start in range(0, len(order), 48):
            rows = order[start:start + 48]
            x, y = train_x[rows], train_y[rows]
            error = 1 / (1 + np.exp(-np.clip(x @ weights + bias, -40, 40))) - y
            gradient = x.T @ error / len(rows) + config['decay'] * weights
            bias_gradient = error.mean()
            momentum = config['momentum'] if config['optimizer'] == 'momentum' else 0.
            velocity = momentum * velocity + gradient
            bias_velocity = momentum * bias_velocity + bias_gradient
            weights -= config['rate'] * velocity
            bias -= config['rate'] * bias_velocity
    accuracy = float(np.mean((valid_x @ weights + bias >= 0) == valid_y))
    return dict(accuracy=accuracy, parameters=config, epochs=EPOCHS, seed=SEED,
                training_rows=720, evaluation_rows=240, metric='validation_accuracy')
