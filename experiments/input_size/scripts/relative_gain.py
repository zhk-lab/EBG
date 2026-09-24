"""Compute relative gain from paired group means, without model calls."""

import csv
import statistics

from compare_models import HERE, JUDGES, METRICS, MODELS, OUTPUT, read, save

SELECTED = ("kimi-k3", "glm-5-3", "luna")


def relative_gain(ebg_mean, baseline_mean):
    return None if baseline_mean == 0 else (ebg_mean - baseline_mean) / baseline_mean * 100


def summarize(rows):
    groups = {}
    for row in rows:
        for label in (row["bin"], "All"):
            key = (row["model"], row["judge"], row["component"], label)
            pair = groups.setdefault(key, {}).setdefault(row["input_id"], {})
            if row["method"] in pair:
                raise ValueError("Duplicate score")
            pair[row["method"]] = row["metrics"]
    output = []
    for (model, judge, component, label), pairs in groups.items():
        if any(set(pair) != {"EBG", "baseline"} for pair in pairs.values()):
            raise ValueError("Missing paired scores")
        for metric in METRICS[component]:
            ebg = statistics.mean(pair["EBG"][metric] for pair in pairs.values())
            baseline = statistics.mean(pair["baseline"][metric] for pair in pairs.values())
            output.append({"model": model, "judge": judge, "component": component, "bin": label,
                           "metric": metric, "n_pairs": len(pairs), "EBG_mean": ebg,
                           "baseline_mean": baseline, "absolute_difference": ebg - baseline,
                           "relative_gain_pct": relative_gain(ebg, baseline)})
    return output


def render(records):
    lookup = {(r['model'], r['judge'], r['component'], r['metric'], r['bin']): r for r in records}
    lines = ['# Relative gains from group means', '', 'Relative gain = (EBG group mean - Base group mean) / Base group mean * 100%.', 'Compute each method mean on the same group, then take their ratio; do not average per-input percentages. A zero Base mean yields NA.', '', 'Models: K3, GLM-5-3, and Luna. Each benchmark has 100 inputs; judges are reported separately.', 'Use the original token-size groups of 33/33/34 inputs, ordered Low, Medium, High.', 'Repository inputs count task documents and readable code; traces count Assistant replies and tool exchanges. See REPORT.md for scope.', '', 'Group changes are descriptive. No significance test of percentage trends is performed; absolute-difference slope intervals do not apply.', 'Relative gains can increase when Base scores decrease; they do not by themselves establish higher EBG accuracy. CSV/JSON exports retain means and absolute differences.', '']
    for judge in JUDGES:
        lines += [f'## {('Qwen' if judge == JUDGES[0] else 'GLM')} judgment', '', 'Each cell lists Low / Medium / High, in percent.', '', '| Benchmark | Metric | K3 | GLM-5-3 | Luna |', '|---|---|---:|---:|---:|']
        for component, metrics in METRICS.items():
            for metric, name in metrics.items():
                cells = []
                for model in SELECTED:
                    values = [lookup[model, judge, component, metric, label]['relative_gain_pct'] for label in ('Low', 'Medium', 'High')]
                    cells.append(' → '.join(('NA' if value is None else f'{value:+.1f}' for value in values)))
                lines.append(f'| {component} | {name} | {' | '.join(cells)} |')
        lines.append('')
    lines += ['## Artifacts and reproduction', '', '- Full results, including All groups: [CSV](results/relative_gain.csv), [JSON](results/relative_gain.json).', '- Scores: results/<model>/sample_scores.json; groups: data/search_samples.json.', '- Run `python experiments/input_size/scripts/relative_gain.py`.', '']
    return '\n'.join(lines)


def main():
    samples = read(ROOT / "outputs/analysis/input_size/data" / "search_samples.json")
    expected = {(r["component"], r["input_id"]): r for r in samples}
    records = []
    for model in SELECTED:
        rows = read(OUTPUT / model / "sample_scores.json")
        if len(rows) != 1200:
            raise ValueError(f"Expected 1200 score records for {model}")
        for row in rows:
            sample = expected[row["component"], row["input_id"]]
            if row["bin"] != sample["bin"] or row["search_count"] != sample["search_count"]:
                raise ValueError("Saved scores use different groups")
        records.extend(summarize(rows))
    save(OUTPUT / "relative_gain.json", records)
    with (OUTPUT / "relative_gain.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (ROOT / "outputs/analysis/input_size/RELATIVE_GAIN.md").write_text(render(records), encoding="utf-8")
    print(f"Saved {len(records)} group/overall metrics; zero baselines: "
          f"{sum(r['relative_gain_pct'] is None for r in records)}")


if __name__ == "__main__":
    main()
