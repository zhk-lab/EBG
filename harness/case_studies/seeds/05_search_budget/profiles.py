"""Expand experiment execution profiles."""

from itertools import product


def configurations(profile):
    for rate, decay, momentum in product(profile['rates'], profile['decays'], profile['momenta']):
        yield dict(optimizer=profile['optimizer'], rate=rate, decay=decay, momentum=momentum)


def select_result(records):
    return max(records, key=lambda record: record['accuracy'])
