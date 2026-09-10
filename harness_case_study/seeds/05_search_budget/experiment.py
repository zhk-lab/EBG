"""Execute resumable optimizer profiles and publish their selected result."""

import argparse
import json
from pathlib import Path

from engine import fit
from profiles import configurations, select_result


def run(name, output):
    profile = json.loads(Path(f'configs/{name}.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=True)
    profile_path = output / 'profile.json'
    if profile_path.exists() and json.loads(profile_path.read_text()) != profile:
        raise ValueError('Profile changed: choose a new --output directory.')
    profile_path.write_text(json.dumps(profile, indent=2), encoding='utf-8')
    path = output / 'trials.jsonl'
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    for index, config in enumerate(configurations(profile)):
        if index < len(records):
            continue
        record = dict(trial_id=f'{name}-{index:03d}', **fit(config))
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        records.append(record)
    selected = select_result(records)
    summary = dict(profile=name, optimizer=profile['optimizer'], accuracy=selected['accuracy'],
                   selected_trial=selected['trial_id'], parameters=selected['parameters'],
                   completed_fits=len(records), selection='maximum_validation_accuracy',
                   records=path.as_posix())
    summary_path = output / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"{name}: validation accuracy={summary['accuracy']:.4f}")
    print(f'Result: {summary_path.as_posix()}')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('profile', choices=['reference', 'candidate'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.profile, args.output or Path('runs') / args.profile)
