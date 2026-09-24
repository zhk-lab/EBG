"""Collect Qwen sensitivity scores and update the report tables."""
import json
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
JUDGE = "qwen3.7-max-2026-06-08"
METRICS = {"specgap": ("f1", "question_quality", "location_f1"), "silentswap": ("localization_score", "location_correct", "code_change_correct")}


def collect(model, benchmark, arm, hops, rounds):
    default = 8 if benchmark == "specgap" else 6
    reused = hops == 1 and rounds == default
    if reused:
        name = "gpt5.6" if model == "luna" else model
        root = ROOT / "outputs/main" / ("EBG" if arm == "graph" else "baseline") / benchmark / name
        first = root / "stage1" if benchmark == "silentswap" and arm == "graph" else root
    else:
        root = ROOT / "outputs/analysis/sensitivity/results/conditions" / model / benchmark / f"{arm}_h{hops}_r{rounds}"
        first = root / "stage1"
    stages = [first]
    if benchmark == "silentswap" and arm == "graph":
        stages.append(root / "stage2")
    samples = json.loads((BASE / "configs/samples.json").read_text(encoding="utf-8"))[benchmark]
    rows = []
    assigned = []
    for sample in samples:
        records = []
        for stage in stages:
            path = stage / "judges" / JUDGE / sample / "status.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["status"] == "complete", str(path)
            assert data["input_id"] == sample
            if data.get("score_source") == "user_assigned_zero":
                assigned.append(sample)
            records.append(data)
        metrics = {key: records[0]["metrics"][key] for key in METRICS[benchmark]}
        if len(records) == 2:
            metrics["code_change_correct"] = records[1]["metrics"]["code_change_correct"]
        rows.append({"input_id": sample, "metrics": metrics})
    assert len(rows) == 100
    return {"model": model, "benchmark": benchmark, "arm": arm, "hops": hops, "rounds": rounds,
            "source": str(root.relative_to(ROOT)), "reused_main": reused, "samples": rows,
            "assigned_zero_ids": assigned, "means": {key: statistics.mean(r["metrics"][key] for r in rows) for key in METRICS[benchmark]}}


def main():
    records = {}

    def get(model, bench, arm, hops, rounds):
        key = (model, bench, arm, hops, rounds)
        if key not in records:
            records[key] = collect(*key)
        return records[key]
    lines = ['## Qwen judge results', '', 'Judge: qwen3.7-max-2026-06-08, enable_thinking=false. Each setting averages 100 inputs, displayed to four decimals. Default-turn Base and one-hop EBG reuse main runs.', '', 'There are 3,200 input/setting/stage judgments: 3,198 scored and two assigned zero by author instruction. Both are GLM-5-3 / SilentSwap stage 2: three-hop ss_039 and five-turn EBG ss_089. Code Correct is zero; localization retains valid stage-1 scores. These two zeros were not scored by Qwen.', '']
    for experiment in ('hops', 'rounds'):
        lines += ['### ' + ('Expansion hops' if experiment == 'hops' else 'Maximum interaction turns'), '']
        for bench in ('specgap', 'silentswap'):
            default = 8 if bench == 'specgap' else 6
            for model in ('luna', 'glm-5-3'):
                title = ('Luna' if model == 'luna' else 'GLM-5-3') + ' · ' + ('SpecGap' if bench == 'specgap' else 'SilentSwap')
                headers = ['Macro F1', 'Question Quality', 'Location F1'] if bench == 'specgap' else ['Localization Score', 'Location Correct', 'Code Correct']
                lines += ['#### ' + title, '', '| Setting | ' + ' | '.join(headers) + ' |', '|---|---:|---:|---:|']
                settings = [('Base (main)', 'raw', 1, default), ('One-hop EBG (main)', 'graph', 1, default), ('Two-hop EBG', 'graph', 2, default), ('Three-hop EBG', 'graph', 3, default)] if experiment == 'hops' else [(f'{n} turns {label}' + (' (main)' if n == default else ''), arm, 1, n) for n in ([4, 6, 8] if bench == 'specgap' else [4, 5, 6]) for label, arm in (('baseline', 'raw'), ('EBG', 'graph'))]
                for label, arm, hops, rounds in settings:
                    result = get(model, bench, arm, hops, rounds)
                    lines.append('| ' + label + ' | ' + ' | '.join((f'{result['means'][key]:.4f}' for key in METRICS[bench])) + ' |')
                lines.append('')
    lines += ['### Observations', '', '- Both models exceed Base on mean SpecGAP Macro F1 and SilentSwap Localization Score across tested settings. These are descriptive differences without significance claims.', '- More hops do not provide uniform gains. Qwen-scored SpecGAP F1 peaks at three hops; SilentSwap Localization Score peaks at three for Luna and two for GLM-5-3. Other metrics have different optima.', '- The main-metric advantages persist at 4/6/8 turns for SpecGAP and 4/5/6 for SilentSwap. EBG scores do not increase monotonically with turns.', '- Code Correct differs: Luna remains below Base in all tested Qwen-scored settings. GLM-5-3 is below Base at two hops, tied at four turns, and slightly below at five turns. This does not establish superiority on every metric.', '', 'Scores, setting means, main-run sources, and author-assigned zeros are in [qwen_results.json](data/qwen_results.json). Regenerate tables with `python experiments/sensitivity/scripts/report_qwen.py`.', '']
    data = ROOT / 'outputs/analysis/sensitivity/data/qwen_results.json'
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(json.dumps({'judge': JUDGE, 'conditions': list(records.values())}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report = ROOT / "outputs/analysis/sensitivity/REPORT.md"
    original = (report.read_text(encoding='utf-8') if report.exists() else '# Analysis report\n').split('## Qwen judge results')[0].rstrip()
    marker = '## Shared evaluation and presentation'
    if '## GLM-5.2 judge results' not in original:
        prefix, old = original.split(marker, 1)
        old = old.replace('## ', '### ')
        original = prefix.rstrip() + '\n\n' + marker + '\n\nThese results cover Luna and GLM-5-3. K3 is outside this supplementary Qwen evaluation. Means are descriptive; no significance claim is made. Existing GLM scores are retained; Qwen tables follow.\n\n## GLM-5.2 judge results\n' + old
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(original.rstrip() + '\n\n' + '\n'.join(lines), encoding='utf-8')
    print(json.dumps([{k: v for k, v in r.items() if k != 'samples'} for r in records.values()], ensure_ascii=False))


if __name__ == "__main__":
    main()
