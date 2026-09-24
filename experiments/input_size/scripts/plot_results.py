"""Plot the current evidence-search summary without rerunning any models."""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_models import JUDGES, METRICS, MODELS, OUTPUT, read


def main():
    records = read(OUTPUT / 'group_medians.json')
    bins = read(OUTPUT / 'bin_summary.json')
    models = list(dict.fromkeys((r['model'] for r in records)))
    selected = [r for r in records if r['judge'] == JUDGES[0] and r['model'] in models]
    lookup = {(r['model'], r['component'], r['metric'], r['bin']): r for r in selected}
    plot_rows, captions = ([], [])
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    for component, metrics in METRICS.items():
        groups = [g for g in bins if g['component'] == component]
        fig, axes = plt.subplots(len(models), 3, figsize=(12, 3.7 * len(models)), squeeze=False, layout='constrained')
        for row, model in enumerate(models):
            for col, (metric, title) in enumerate(metrics.items()):
                ax = axes[row, col]
                for method, color, offset in (('EBG', '#156e94', -0.04), ('baseline', '#bb6738', 0.04)):
                    values = [lookup[model, component, metric, g['bin']] for g in groups]
                    median = np.array([v[f'{method}_median'] for v in values])
                    ci = np.array([v[f'{method}_ci95'] for v in values])
                    ax.errorbar(np.arange(3) + offset, median, yerr=np.maximum(0, np.vstack((median - ci[:, 0], ci[:, 1] - median))), color=color, marker='o', markersize=5, capsize=3, label=method)
                    for g, v in zip(groups, values):
                        plot_rows.append({'model': model, 'judge': JUDGES[0], 'component': component, 'metric': metric, 'bin': g['bin_label'], 'n': g['n'], 'method': method, 'median': v[f'{method}_median'], 'ci_low': v[f'{method}_ci95'][0], 'ci_high': v[f'{method}_ci95'][1]})
                ax.set_xticks(range(3), [f'{g['bin_label']}\n(n={g['n']})' for g in groups])
                ax.set_ylim(-0.04, 1.04)
                ax.set_title(title)
                ax.set_xlabel('Visible input size (tokens)')
                ax.grid(axis='y', alpha=0.2)
                if col == 0:
                    ax.set_ylabel(f'{MODELS[model]}\nMedian score')
                if row == 0 and col == 2:
                    ax.legend(loc='upper right', frameon=False)
        fig.suptitle(f'{component} | Visible input size | Qwen judge', fontsize=14)
        path = OUTPUT / f'fig4b_{component}_evidence_search.png'
        fig.savefig(path, dpi=180)
        plt.close(fig)
        captions += [f'## {path.name}', '', 'Points show group score medians; error bars are 95% intervals from 2,000 paired sample bootstrap replicates. Connecting lines illustrate group differences rather than regressions. See group_medians.json for paired mean differences and intervals.', '']
    with (OUTPUT / 'plot_data.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plot_rows[0]))
        writer.writeheader()
        writer.writerows(plot_rows)
    (OUTPUT / 'figure_captions.md').write_text('\n'.join(captions), encoding='utf-8')


if __name__ == "__main__":
    main()
