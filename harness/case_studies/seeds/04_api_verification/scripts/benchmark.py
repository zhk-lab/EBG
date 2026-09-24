"""Fixed replay workload; cost units are simulated scheduling costs."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import execute


class BenchmarkClient:
    def __init__(self):
        self.calls = 0
        self.cost = 0

    def complete(self, prompt):
        self.calls += 1
        self.cost += 11
        return prompt.upper()

    def complete_batch(self, prompts):
        self.calls += 1
        self.cost += 10 + len(prompts)
        return [prompt.upper() for prompt in prompts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    output = Path('runs') / (args.name + '.json')
    if output.exists():
        print(output.read_text())
        return
    requests = [f'item {i % 8}' for i in range(48)]
    client = BenchmarkClient()
    first = execute(requests, client, completed={})
    completed = dict(zip(requests[:24], first[:24]))
    second = execute(requests, client, completed=completed)
    assert first == second == [item.upper() for item in requests]
    result = {'name': args.name, 'workload': 'replay', 'items': 96,
              'calls': client.calls, 'cost_units': client.cost,
              'ordering_valid': True}
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
