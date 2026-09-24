"""Classify SilentSwap stage-1 localization failures by expansion depth."""
from collections import Counter

from attribute import CATEGORIES, VERSION, audit_run, classify, write
from summarize import HERE, ROOT, read, screen


MODELS = ("luna", "glm-5-3")
LABELS = {"luna": "Luna", "glm-5-3": "GLM-5-3"}


def main():
    baseline = [
        row
        for row in read(ROOT / "outputs/analysis/failure_analysis/data/attributed_keys.json")
        if row["benchmark"] == "silentswap"
        and row["judge"] == "glm-5-2"
        and row["model"] in MODELS
    ]
    records = [dict(row, hops=1) for row in baseline]

    for model in MODELS:
        for hops in (2, 3):
            stage = (
                ROOT
                / "outputs/analysis/sensitivity/results/conditions"
                / model
                / "silentswap"
                / f"graph_h{hops}_r6"
                / "stage1"
            )
            paths = sorted((stage / "judges/glm-5-2").glob("*/result.json"))
            assert len(paths) == 100, (model, hops, len(paths))
            for path in paths:
                sample = path.parent.name
                gold = read(
                    ROOT
                    / "data/prepared/silentswap/artifacts/hidden_gold"
                    / f"{sample}.json"
                )
                failures = [
                    item
                    for item in screen("silentswap", gold, read(path))
                    if item["outcome"] != "full"
                ]
                if not failures:
                    continue

                run = stage / "runs" / sample
                prediction = read(run / "prediction.json")
                assert read(path.with_name("judge_input.json"))["prediction"] == prediction
                base = dict(
                    benchmark="silentswap",
                    model=model,
                    sample_id=sample,
                    run_path=run.relative_to(ROOT).as_posix(),
                    prediction=prediction,
                )
                cache = (
                    ROOT / "outputs/analysis/failure_analysis/data/silentswap_hop_audits"
                    / model
                    / f"h{hops}"
                    / f"{sample}.json"
                )
                audit = read(cache) if cache.exists() else None
                if audit is None or audit.get("version") != VERSION:
                    audit = audit_run(base)
                    audit["version"] = VERSION
                    write(cache, audit)

                for item in failures:
                    row = dict(base, **item)
                    category, reason, evidence = classify(row, audit)
                    records.append(
                        dict(
                            model=model,
                            sample_id=sample,
                            key_id=item["key_id"],
                            hops=hops,
                            outcome=item["outcome"],
                            category=category,
                            reason=reason,
                            evidence=evidence,
                            result_path=path.relative_to(ROOT).as_posix(),
                            request_audit_path=cache.relative_to(ROOT).as_posix(),
                        )
                    )
            print(f"Completed {model} h{hops}", flush=True)

    summary = []
    for model in MODELS:
        for hops in (1, 2, 3):
            rows = [
                row
                for row in records
                if row["model"] == model and row["hops"] == hops
            ]
            counts = Counter(row["category"] for row in rows)
            summary.append(
                dict(
                    model=model,
                    hops=hops,
                    failures=len(rows),
                    pending=counts[None],
                    **{category: counts[category] for category in CATEGORIES},
                )
            )

    write(ROOT / "outputs/analysis/failure_analysis/data/silentswap_hop_failure_keys.json", records)
    write(ROOT / "outputs/analysis/failure_analysis/data/silentswap_hop_failure_summary.json", summary)
    update_report(summary)
    for row in summary:
        print(row)


def update_report(summary):
    title = '## SilentSwap hops and failure categories'
    text = '\n\nSilentSwap first-stage localization failures, scored by GLM-5.2. Each model/setting has 100 inputs and 500 Gold swaps. Percentages use attributed failures.\n\n| Model | Hops | Failures | File selection | Evidence presentation | Model judgment | Unresolved |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for row in summary:
        attributed = row['failures'] - row['pending']
        cells = [f'{row[category]}（{row[category] / attributed:.1%}）' if attributed else str(row[category]) for category in CATEGORIES]
        text += f'| {LABELS[row['model']]} | {row['hops']} | {row['failures']} | ' + ' | '.join(cells) + f' | {row['pending']} |\n'
    text += '\nFor Luna, evidence-presentation omissions decrease from 32 at one hop to 31 at two and 26 at three; file-selection omissions increase from 19 to 28. GLM-5-3 has 27, 26, and 30 presentation omissions, with no consistent decline. This is not a shared trend across models.\n\nK3 has no formal two/three-hop SilentSwap results here. Settings are independent runs, so distribution changes are not fixed-case recovery rates. Luna uses different read budgets at two and three hops; those differences cannot be attributed solely to hops. GLM-5-3 uses the same budget for both.\n\nRun `python experiments/failure_analysis/scripts/silentswap_hop_failure_analysis.py`. Outputs: `data/silentswap_hop_failure_keys.json` and `data/silentswap_hop_failure_summary.json`.\n'
    report = ROOT / "outputs/analysis/failure_analysis/REPORT.md"
    content = (report.read_text(encoding='utf-8') if report.exists() else '# Analysis report\n').split(title)[0].rstrip()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(content + '\n\n' + title + text, encoding='utf-8')


if __name__ == "__main__":
    main()
