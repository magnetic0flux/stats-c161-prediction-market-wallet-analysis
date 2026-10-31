"""Regenerate the edited four-model HTML summary from saved run outputs."""
from pathlib import Path
import base64
import html
import json
import re

import numpy as np
import pandas as pd

MODELS = [
    ('logistic__baseline', 'Logistic: baseline', '#2563eb', '-'),
    ('logistic__fixed24h_5', 'Logistic: with history', '#a855f7', '--'),
    ('random_forest__baseline', 'Random forest: baseline', '#d97706', '-'),
    ('random_forest__fixed24h_5', 'Random forest: with history', '#0d9488', '--'),
]


def make_performance_figures(run_dir, folder):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_curve, roc_auc_score
    run_dir, folder = Path(run_dir), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(run_dir / 'exploratory_test_predictions.csv')
    y = data.conditional_profit_label.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6))
    for column, label, color, style in MODELS:
        p = data[column].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=color, linestyle=style, linewidth=2,
                     label=f'{label} (AUC {roc_auc_score(y, p):.4f})')
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy='quantile')
        axes[1].plot(predicted, observed, color=color, linestyle=style, marker='o',
                     markersize=5, linewidth=2, label=label)
    assert len(axes[0].lines) == len(axes[1].lines) == 4
    axes[0].set(title='ROC curve', xlabel='False positive rate', ylabel='True positive rate',
                xlim=(0, 1), ylim=(0, 1.015))
    axes[1].plot([0, 1], [0, 1], color='#94a3b8', linestyle=':', linewidth=1.5, zorder=0)
    axes[1].set(title='Calibration', xlabel='Mean predicted probability',
                ylabel='Observed profitable fraction', xlim=(0, 1), ylim=(0, 1.015))
    for ax in axes:
        ax.grid(alpha=.13)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='lower right' if ax is axes[0] else 'upper left', fontsize=8.5,
                  framealpha=.95, edgecolor='#dbe3ec')
    fig.suptitle('Logistic regression and random forest: baseline vs history', fontsize=15,
                 fontweight='bold', y=.98)
    fig.text(.06, .025, f'History = five prior-24h features. All curves use the same {len(y):,} test observations.\n'
                       'Some curves nearly overlap. The dotted gray line in calibration indicates perfect calibration.',
             fontsize=9, color='#475569', linespacing=1.5)
    fig.subplots_adjust(left=.06, right=.985, top=.87, bottom=.20, wspace=.25)
    fig.savefig(folder / 'roc_calibration_four_models.png', dpi=180, facecolor='white')
    plt.close(fig)
    return folder / 'roc_calibration_four_models.png'


def metric_table(frame):
    frame = frame.copy()
    labels = {'baseline': 'Baseline', 'expanding_3': '+3 cumulative history',
              'fixed24h_3': '+3 prior-24h history', 'expanding_5': '+5 cumulative history',
              'fixed24h_5': '+5 prior-24h history'}
    frame['Model'] = frame.model.map({'logistic': 'Logistic', 'random_forest': 'Random forest', 'reference': 'Reference'})
    frame['Feature set'] = frame.variant.map(labels).fillna(frame.variant)
    frame['Accuracy (%)'] = 100 * frame.accuracy_threshold_0_5
    frame = frame.rename(columns={'roc_auc': 'ROC-AUC', 'log_loss': 'Log loss',
                         'brier': 'Brier score', 'pr_auc_average_precision': 'Average precision'})
    return frame[['Model', 'Feature set', 'Accuracy (%)', 'ROC-AUC', 'Log loss',
                  'Brier score', 'Average precision']].to_html(index=False, float_format=lambda x: f'{x:.4f}')


def write_summary(run_dir, folder):
    """Replace every result table and figure in the supplied edited HTML template."""
    run_dir, folder = Path(run_dir), Path(folder)
    text = Path(__file__).with_name('RF_logistic_resultssummary.template.html').read_text(encoding='utf-8')
    def update_section(title, replacements):
        nonlocal text
        start = text.index('<h2>' + title + '</h2>')
        end = text.find('<h2>', start + 5)
        if end < 0: end = text.index('</body>', start)
        section = text[start:end]
        matches = list(re.finditer(r'<table\b.*?</table>', section, flags=re.S))
        for index in sorted(replacements, reverse=True):
            match = matches[index]
            section = section[:match.start()] + replacements[index] + section[match.end():]
        text = text[:start] + section + text[end:]

    cohorts = pd.read_csv(run_dir / 'cohort_summary.csv')
    features = pd.read_csv(run_dir / 'feature_sets.csv')
    folds = pd.read_csv(run_dir / 'forward_cv_audit.csv')
    validation = pd.read_csv(run_dir / 'validation_metrics.csv')
    test = pd.read_csv(run_dir / 'exploratory_test_metrics.csv')
    selected = json.loads((run_dir / 'selected_parameters.json').read_text())
    manifest = json.loads((run_dir / 'model_manifest.json').read_text())
    config = manifest['config']
    candidate_count = len(json.loads((run_dir / 'candidate_grid.json').read_text()))
    update_section('Notebook 01: data preparation outputs', {
        0: cohorts[['cohort', 'rows', 'wallets', 'markets', 'event_groups', 'positive_rate']].to_html(index=False),
        1: features.to_html(index=False),
    })
    first = next(iter(selected.values()))
    same = all(params == first for params in selected.values())
    display_selected = lambda key: html.escape(str(first[key])) if same else 'Varies by feature set; see below'
    tuning = '<table><thead><tr><th>Parameter</th><th>Values considered</th><th>Selected value</th></tr></thead><tbody>'
    for key, values in [('max_depth', '6, 12, None'), ('min_samples_leaf', '10, 25, 50'),
                        ('max_features', 'sqrt, 0.7, 1.0')]:
        tuning += f'<tr><td><code>{key}</code></td><td>{values}</td><td>{display_selected(key)}</td></tr>'
    tuning += '</tbody></table>'
    if not same:
        tuning += pd.DataFrame(selected).T.to_html()
    update_section('Notebook 02: tuning and outer-validation outputs', {
        0: folds[['fold', 'validation_start', 'validation_end', 'training_rows', 'validation_rows',
                  'training_events', 'validation_events']].to_html(index=False),
        1: tuning,
        2: metric_table(validation),
    })
    update_section('Notebook 03: held-forward test outputs', {
        0: metric_table(test[test.model.isin(['logistic', 'random_forest'])]),
        1: metric_table(test[test.model.eq('reference')]),
    })
    text = text.replace('8 parameter combinations', f'{candidate_count} parameter combinations')
    text = text.replace('same eight candidates', f'same {candidate_count} candidates')
    text = text.replace('The outer validation set has 6,956 rows.',
                        f'The outer validation set has {int(cohorts.set_index("cohort").loc["validation", "rows"]):,} rows.')
    if not same:
        text = text.replace('All five feature sets selected the same parameter combination.',
                            'Selected settings differ by feature set, as listed above.')
    params = next(fit['parameters'] for fit in manifest['fits'] if fit['model'] == 'random_forest')
    keys = ['n_estimators', 'max_depth', 'min_samples_leaf', 'max_features', 'criterion',
            'bootstrap', 'class_weight', 'random_state']
    expression = 'RandomForestClassifier(\n' + ',\n'.join(f'    {key}={params[key]!r}' for key in keys) + '\n)'
    text = re.sub(r'<pre><code>RandomForestClassifier\(.*?</code></pre>',
                  lambda _: '<pre><code>' + html.escape(expression) + '</code></pre>', text, flags=re.S)
    text = text.replace('200 trees were fixed, not tuned', f'{params["n_estimators"]} trees were fixed, not tuned')
    text = text.replace('All forests used 200 trees;', f'All forests used {params["n_estimators"]} trees;')
    # Template counts are updated for validation runs as well as the actual cohort.
    counts = cohorts.set_index('cohort').rows
    for original, cohort in [('26,507', 'test'), ('52,888', 'refit'), ('39,659', 'train')]:
        text = text.replace(original, f'{int(counts[cohort]):,}')
    positive = test.loc[test.model.eq('logistic') & test.variant.eq('baseline')].iloc[0].positive_rate
    text = text.replace('59.11%', f'{100 * positive:.2f}%')
    image_paths = [folder / 'roc_calibration_four_models.png', folder / 'variable_importance_comparison.png',
                   folder / 'validation_feature_importance.png']
    matches = list(re.finditer(r'<img\b[^>]*>', text, flags=re.I))
    assert len(matches) == len(image_paths)
    for match, path in reversed(list(zip(matches, image_paths))):
        image = '<img alt="' + html.escape(path.stem) + '" src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '">'
        text = text[:match.start()] + image + text[match.end():]
    destination = folder / 'RF_logistic_resultssummary.html'
    destination.write_text(text, encoding='utf-8')
    return destination
