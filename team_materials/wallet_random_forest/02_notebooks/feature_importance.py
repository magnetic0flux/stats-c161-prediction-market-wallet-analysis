"""Extract individual-feature importance from the existing fitted wallet models.

No fitting, tuning, or test scoring occurs. Requires the models/, feature_sets.csv,
and rf_impurity_importance.csv from the completed full-data workflow.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import html
import json
import shutil

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VARIANTS = ['baseline', 'expanding_3', 'fixed24h_3', 'expanding_5', 'fixed24h_5']
TITLES = {'baseline': 'Baseline', 'expanding_3': 'Baseline + 3 cumulative-history features',
          'fixed24h_3': 'Baseline + 3 prior-24h history features',
          'expanding_5': 'Baseline + 5 cumulative-history features',
          'fixed24h_5': 'Baseline + 5 prior-24h history features'}
LABELS = {
    'entry_log_buy': 'Entry buy dollars (log)',
    'entry_price': 'Entry price', 'entry_price_squared': 'Entry price²',
    'entry_price_cubed': 'Entry price³',
    'entry_concentration': 'Entry one-sidedness',
    'entry_log_fill_rows': 'Entry fill count (log)',
    'market_log_dollars': 'Prior market dollars (log)',
    'market_log_wallets': 'Prior market wallets (log)',
    'market_log_observed_age': 'Observed market age (log)',
    'clock_sin': 'Hour of day: sine', 'clock_cos': 'Hour of day: cosine',
    'window_exposure_days': 'Elapsed study days',
    'entry_price__missing': 'Entry price missing',
    'history__log_fills': 'History: fill count (log)',
    'history__log_markets': 'History: market count (log)',
    'history__log_buy_dollars': 'History: buy dollars (log)',
    'history__buy_hhi': 'History: buy concentration (HHI)',
    'history__log_avg_buy_ticket': 'History: average buy ticket (log)',
    'history__buy_hhi__missing': 'History: buy concentration missing',
    'history__log_avg_buy_ticket__missing': 'History: average buy ticket missing',
}
PRICE = {'entry_price', 'entry_price_squared', 'entry_price_cubed'}


def extract(run_dir):
    """Preserve the ColumnTransformer's output order when naming coefficients."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'model_manifest.json').read_text())
    fits = {(f['model'], f['variant']): f for f in manifest['fits']}
    feature_sets = pd.read_csv(run_dir / 'feature_sets.csv').set_index('variant')
    existing_rf = pd.read_csv(run_dir / 'rf_impurity_importance.csv')
    rows, identities = [], []
    for variant in VARIANTS:
        columns = feature_sets.loc[variant, 'feature_names'].split(', ')
        for kind in ['logistic', 'random_forest']:
            path = run_dir / 'models' / f'{kind}_{variant}.joblib'
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != manifest['model_files'][path.name]:
                raise ValueError('Saved model hash changed: ' + path.name)
            fit = fits[(kind, variant)]
            assert list(fit['feature_names']) == columns
            model = joblib.load(path)
            assert list(model.named_steps['impute'].feature_names_in_) == columns
            identities.append({'variant': variant, 'model': kind, 'filename': path.name,
                'sha256': digest, 'refit_rows': int(fit['refit_rows']),
                'development_rows': int(fit['development_rows'])})
            estimator = model.named_steps['model']
            if kind == 'logistic':
                ordered, scaled = [], []
                for name, transformer, indices in model.named_steps['scale'].transformers_:
                    if isinstance(transformer, str) and transformer == 'drop':
                        continue
                    ordered.extend(columns[int(i)] for i in indices)
                    scaled.extend([name == 'continuous'] * len(indices))
                values = estimator.coef_.ravel()
                assert len(values) == len(ordered) == len(columns)
                assert set(ordered) == set(columns)
                for feature, beta, standard in zip(ordered, values, scaled):
                    rows.append(dict(model=kind, variant=variant, feature=feature,
                        label=LABELS[feature], is_missing_indicator=feature.endswith('__missing'),
                        coefficient=float(beta), importance=float(abs(beta)),
                        measure='absolute_standardized_coefficient' if standard else 'absolute_unscaled_indicator_coefficient',
                        coefficient_units='log odds per refit SD' if standard else 'log odds for missing vs observed',
                        refit_rows=int(fit['refit_rows'])))
            else:
                assert estimator.criterion == 'gini'
                values = estimator.feature_importances_
                assert len(values) == len(columns)
                assert np.isclose(values.sum(), 1.)
                old = existing_rf[existing_rf.variant.eq(variant)].set_index('feature')
                assert np.allclose(values, old.loc[columns, 'impurity_importance'])
                for feature, value in zip(columns, values):
                    rows.append(dict(model=kind, variant=variant, feature=feature,
                        label=LABELS[feature], is_missing_indicator=feature.endswith('__missing'),
                        coefficient=np.nan, importance=float(value),
                        measure='gini_mean_decrease_in_impurity', coefficient_units='', refit_rows=int(fit['refit_rows'])))
    result = pd.DataFrame(rows)
    assert not result.duplicated(['model', 'variant', 'feature']).any()
    assert np.isfinite(result.importance).all()
    result['rank_among_nonmissing_features'] = np.nan
    for (_, _), indices in result[~result.is_missing_indicator].groupby(['model', 'variant']).groups.items():
        result.loc[indices, 'rank_among_nonmissing_features'] = result.loc[indices, 'importance'].rank(ascending=False, method='min')
    return result, identities


def export_tables(run_dir, out=None):
    """Refresh only importance tables from existing models, including evaluated runs."""
    run_dir = Path(run_dir)
    out = Path(out) if out is not None else run_dir
    out.mkdir(parents=True, exist_ok=True)
    result, identities = extract(run_dir)
    result.to_csv(out / 'variable_importance_all_models.csv', index=False)
    result[result.model.eq('logistic')].to_csv(out / 'logistic_coefficient_importance.csv', index=False)
    result[result.model.eq('random_forest')].to_csv(out / 'random_forest_gini_importance.csv', index=False)
    return result, identities


def panel(ax, data, kind, detail=False):
    subset = data[~data.is_missing_indicator].copy()
    if detail:
        subset = subset[~subset.feature.isin(PRICE)]
    subset = subset.sort_values(['importance', 'feature'])
    colors = ['#e68b2c' if f in PRICE else '#0d9488' if f.startswith('history__') else '#4974a5'
              for f in subset.feature]
    positions = np.arange(len(subset))
    ax.barh(positions, subset.importance, color=colors, height=.72)
    ax.set_yticks(positions, subset.label, fontsize=9)
    maximum = max(float(subset.importance.max()), .001)
    for y, row in zip(positions, subset.itertuples()):
        label = f'{row.importance:.3f}' if row.importance == 0 or row.importance >= .001 else f'{row.importance:.1e}'
        if kind == 'logistic' and row.coefficient != 0:
            label += ' (+)' if row.coefficient > 0 else ' (−)'
        ax.text(row.importance + maximum * .015, y, label, va='center', fontsize=8, color='#243349')
    ax.set_xlim(0, maximum * (1.34 if kind == 'logistic' else 1.23))
    ax.set_xlabel('Absolute standardized coefficient |β| (log-odds units)' if kind == 'logistic'
                  else 'Gini importance (fraction of total impurity reduction)', fontsize=9, labelpad=8)
    ax.xaxis.grid(True, alpha=.18)
    ax.set_axisbelow(True)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(axis='y', length=0)


def plot_comparison(data, destination, variants, detail=False):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': '#172a42',
                         'axes.labelcolor': '#334155', 'xtick.color': '#475569', 'ytick.color': '#243349'})
    height = 7.1 * len(variants) if not detail else 6.1 * len(variants)
    fig, axes = plt.subplots(len(variants), 2, figsize=(17.5, height), squeeze=False)
    for row, variant in enumerate(variants):
        for col, kind in enumerate(['logistic', 'random_forest']):
            panel(axes[row, col], data[data.variant.eq(variant) & data.model.eq(kind)], kind, detail)
            axes[row, col].set_title(('Logistic regression' if kind == 'logistic' else 'Random forest')
                                    + '\n' + TITLES[variant], loc='left', fontsize=13, fontweight='bold', pad=15)
    heading = 'Individual feature importance' if not detail else 'Smaller-feature detail: price terms omitted from display'
    fig.suptitle(heading, x=.035, y=.992, ha='left', fontsize=20, fontweight='bold')
    counts = ', '.join(f'{int(n):,}' for n in sorted(data.refit_rows.unique()))
    note = (f'Same saved final models • {counts} refit observations • each panel has its own scale\n'
            'Orange: price terms   Teal: history   Blue: other baseline features   (+/−): logistic coefficient sign\n'
            'Missingness indicators are listed separately in the report and full CSV. Importance is not statistical significance.')
    if detail:
        note = 'Models still include price, price² and price³; no refitting or renormalization.\n' + note
    fig.text(.035, .012, note, fontsize=9.5, va='bottom', linespacing=1.5)
    fig.subplots_adjust(left=.19, right=.975, bottom=.10 if len(variants) > 1 else .18,
                        top=.93 if len(variants) > 1 else .88, wspace=.66, hspace=.48)
    fig.savefig(destination, dpi=170, facecolor='white')
    fig.savefig(destination.with_suffix('.svg'), facecolor='white')
    plt.close(fig)


def image_tag(path):
    encoded = base64.b64encode(path.read_bytes()).decode()
    return '<img alt="' + html.escape(path.stem) + '" src="data:image/png;base64,' + encoded + '">'


def build_report(data, out):
    sections = []
    for variant in VARIANTS:
        table = data[data.variant.eq(variant)].pivot(index=['feature', 'label', 'is_missing_indicator'],
                     columns='model', values='importance').reset_index()
        table = table.sort_values(['is_missing_indicator', 'random_forest'], ascending=[True, False])
        table.columns.name = None
        sections.append('<details><summary>' + TITLES[variant] + ': all individual columns</summary>'
                        + table.to_html(index=False, float_format=lambda x: f'{x:.6f}') + '</details>')
    flags = data[data.is_missing_indicator & data.variant.isin(['baseline', 'fixed24h_5'])][
        ['variant', 'model', 'label', 'importance', 'measure']]
    body = '''<!DOCTYPE html><html><head><meta charset="utf-8"><title>Wallet variable importance comparison</title>
    <style>body{font:16px/1.55 system-ui,sans-serif;color:#172a42;max-width:1250px;margin:35px auto;padding:0 24px}
    h1{font-size:32px;line-height:1.2}h2{margin-top:35px}p{max-width:1040px}img{width:100%;height:auto}
    table{border-collapse:collapse;width:100%;font-size:13px;margin:15px 0}th,td{padding:8px;border-bottom:1px solid #dbe3ec;text-align:left}
    th{background:#edf3f8}summary{cursor:pointer;font-weight:650;padding:12px;background:#f1f5f9}details{margin:15px 0}
    .note{padding:18px;background:#eff6ff;border-left:4px solid #4974a5}code{font-size:.9em}</style></head><body>
    <h1>Wallet variable importance: logistic regression vs random forest</h1>
    <p>Updated individual-column rankings from the existing full-data models. The main comparison shows
    the baseline and baseline plus five prior-24-hour history features. All five feature-set variants are included in the tables and downloadable package.</p>
    <div class="note"><b>What changed:</b> logistic importance is the absolute coefficient after continuous-predictor standardization;
    random forest importance is mean decrease in Gini impurity. Individual entry-price terms are displayed separately.
    The saved models, tuning, data split and predictions have not changed.</div>
    <h2>Main comparison</h2>'''
    body += image_tag(out / 'variable_importance_comparison.png')
    rf = data[data.model.eq('random_forest') & data.feature.isin(PRICE)].groupby('variant').importance.sum()
    body += (f'<p><b>Result:</b> the three price terms account for {rf["baseline"]:.1%} of total Gini importance in the baseline forest '
             f'and {rf["fixed24h_5"]:.1%} in the forest with five prior-24h history features. This is the share of training impurity reduction, '
             'not a share of accuracy or explained profit. The same underlying price information is split across three correlated columns.</p>')
    body += '''<h2>How to read these scores</h2><ul>
    <li><b>Logistic:</b> |β| is coefficient magnitude per one standard deviation of the transformed continuous predictor.
    The sign is shown beside the bar, but each polynomial-price coefficient should not be interpreted in isolation.
    The fitted logistic models use L2 regularization (C=1).</li>
    <li><b>Random forest:</b> each score is the normalized, sample-weighted total Gini impurity reduction attributed to
    that column across trees. Scores across all columns sum to one for each model.
    Impurity scores describe training splits and may favor variables with many distinct values.</li>
    <li>Compare rankings within each model. Logistic coefficients and Gini scores have different units;
    neither score is a p-value, a percent accuracy improvement, or proof of causality.</li>
    <li>Entry price, price² and price³ are strongly correlated. Their separate scores can split or redistribute
    the same underlying signal. The earlier grouped permutation plot remains the supporting check for price overall.</li></ul>
    <h2>Missing-value indicators</h2><p>These columns remain unscaled in the logistic pipeline, so their coefficients represent
    the change from observed to missing (0→1), rather than a one-SD change. They are shown here instead of being mixed into the continuous-predictor ranking.
    Random forest importance is on the same Gini scale as its other columns. All indicators remain in the models and CSV.</p>'''
    body += flags.to_html(index=False, float_format=lambda x: f'{x:.6f}')
    body += '<h2>Full numeric results</h2>' + ''.join(sections)
    body += '''<h2>Supporting check: grouped validation permutation importance</h2>
    <p>This retained chart uses the earlier development models and outer-validation rows, with sampling and shuffle counts
    recorded in the supporting CSV and original run configuration. Scores are increases in validation log loss.
    It is a different measure from the refitted-model scores above;
    error bars show shuffle standard deviation, not confidence intervals.</p>'''
    prior = out / 'supporting_grouped_permutation_importance.png'
    if prior.exists():
        body += image_tag(prior)
    body += '''<h2>Definitions and provenance</h2><p>The individual scores were extracted from the ten saved final pipelines,
    fitted on the refit cohort recorded in the model manifest and numeric results. No model was retrained and no test labels were read for this importance update.
    Scalers and imputers were fitted within the prior workflow; coefficient names are mapped through the actual ColumnTransformer output order.
    Exported RF scores were checked against the previous rf_impurity_importance.csv.</p>
    <p>Entry price is buy dollars divided by bought shares at the wallet-market pair's earliest observed timestamp in the supplied week.
    Market age is time since the first observed market trade, not verified opening time. Elapsed study days is time since September 29 UTC,
    not weekday. History is either cumulative observed-week history or strictly prior 24-hour history as labeled.</p>
    <p>Reproduce with the accompanying notebook or Python script and the saved full-data workflow folder.
    See metadata.json for model hashes and checks. Existing test-performance results are exploratory because the test set has already been inspected.</p>
    </body></html>'''
    (out / 'variable_importance_report.html').write_text(body, encoding='utf-8')


def build(run_dir, out):
    run_dir, out = Path(run_dir).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    result, identities = export_tables(run_dir, out)
    selected = result[result.variant.isin(['baseline', 'fixed24h_5'])]
    plot_comparison(result, out / 'variable_importance_comparison.png', ['baseline', 'fixed24h_5'])
    for variant in VARIANTS:
        plot_comparison(result, out / f'importance_{variant}.png', [variant])
    source = run_dir / 'figures' / 'validation_feature_importance.png'
    if source.exists():
        shutil.copyfile(source, out / 'supporting_grouped_permutation_importance.png')
    shutil.copyfile(run_dir / 'validation_grouped_permutation_importance.csv', out / 'supporting_grouped_permutation_importance.csv')
    metadata = {'model_refit_rows': sorted(int(n) for n in result.refit_rows.unique()), 'models_retrained': False, 'test_labels_read': False,
        'variants': VARIANTS, 'models': identities,
        'checks': ['Saved-model hashes match model_manifest.json',
                   'Coefficient names match actual transformer output order',
                   'RF criterion verified as gini for all five models',
                   'All five RF importance vectors match prior export',
                   'Each RF vector sums to one', 'All importance values finite',
                   'All model/variant/feature identifiers unique'],
        'missing_indicators': 'Unscaled in logistic; omitted from main continuous-feature bars, retained in tables and CSV',
        'supporting_permutation': 'Prior development-model grouped validation importance; not recalculated or numerically comparable'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    build_report(result, out)
    top = selected[~selected.is_missing_indicator].sort_values('importance', ascending=False).groupby(['variant', 'model']).head(5)
    print(top[['variant', 'model', 'label', 'importance']].to_string(index=False))
    print(f'Wrote {len(result)} individual-feature rows from 10 saved models to {out}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, help='Folder containing models/ and feature_sets.csv')
    parser.add_argument('--output-dir', default='variable_importance_results')
    args = parser.parse_args()
    build(args.run_dir, args.output_dir)
