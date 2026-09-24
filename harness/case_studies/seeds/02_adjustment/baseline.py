import json
from pathlib import Path

from experiment import save_result
from jobs import make_jobs, run_one


if __name__ == '__main__':
    config = json.loads(Path('config.json').read_text(encoding='utf-8'))
    rows = [run_one(job, 'random') for job in make_jobs(config)]
    save_result(Path('results/baseline.json'), rows, 'random', 'reference', config)
