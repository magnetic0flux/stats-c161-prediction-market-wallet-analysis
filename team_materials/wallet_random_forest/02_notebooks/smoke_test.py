"""Exercise all four notebooks on explicitly synthetic data, never real inputs.

Run: python smoke_test.py
Synthetic files and executed copies are created in a temporary directory.
This validates code execution, not scientific performance on actual wallets.
"""
from pathlib import Path
import ast
import argparse
import contextlib
import json
import os
import tempfile

import joblib
import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd

import rf_common as rf


def synthetic_project(root):
    root = Path(root)
    data = root / 'data'
    bundle = root / '04_results/03_history_validation'
    data.mkdir(parents=True)
    bundle.mkdir(parents=True)
    rng = np.random.default_rng(2481)
    rows = []
    for day in range(7):
        for i in range(210):
            t = pd.Timestamp('2026-09-29', tz='UTC') + pd.Timedelta(days=day, minutes=30 + i * 6)
            price = float(rng.uniform(.05, .95))
            history = int(rng.integers(0, 60))
            probability = 1 / (1 + np.exp(-(-.2 + 2.2 * (price - .5) + .25 * np.sin(history))))
            event = f'day{day}_event{i // 7}'
            # Earlier members of a test event must be excluded from train/refit.
            if day in [1, 5] and i < 7:
                event = 'cross_day_test_event'
            # Earlier members of an inner validation event must be purged.
            if day in [1, 2] and 7 <= i < 14:
                event = 'cross_day_cv_event'
            resolved = t + pd.Timedelta(minutes=25 + int(rng.integers(0, 90)))
            if i % 37 == 0:
                resolved += pd.Timedelta(days=1)
            p = np.nan if i % 89 == 0 else price
            r = dict(wallet=f'wallet{i % 85}', market=f'market_day{day}_{i // 7}',
                prediction_time=t, resolved_at=resolved, event_group=event,
                conditional_profit_label=int(rng.random() < probability),
                confirmed_net_profit_label=np.nan,
                feature_entry_log_buy_dollars=float(np.log1p(rng.uniform(1, 300))),
                feature_entry_buy_vwap=p,
                feature_entry_direction_concentration=float(rng.uniform(0, 1)),
                feature_entry_fill_rows=int(rng.integers(1, 5)),
                feature_strict_prior_market_log_dollars=float(rng.uniform(0, 10)),
                feature_strict_prior_market_log_wallets=float(rng.uniform(0, 6)),
                feature_observed_market_age_hours=float(rng.uniform(0, 48)),
                feature_entry_utc_hour=t.hour,
                feature_week_history_latest_time=t - pd.Timedelta(minutes=1),
                audit_only_fixed24h_left_truncated=int(day == 0))
            for period, count in [('week_history', history + day * 3), ('fixed24h', history)]:
                r['feature_' + period + '_log_fill_rows'] = float(np.log1p(count))
                r['feature_' + period + '_log_markets'] = float(np.log1p(min(count, 10)))
                r['feature_' + period + '_log_buy_dollars'] = float(np.log1p(count * rng.uniform(0, 10)))
                r['feature_' + period + '_buy_market_hhi'] = np.nan if count == 0 else float(rng.uniform(.1, 1))
                r['feature_' + period + '_log_avg_buy_ticket'] = np.nan if count == 0 else float(np.log1p(rng.uniform(0, 10)))
            rows.append(r)
    d = pd.DataFrame(rows)
    t0, t1 = pd.Timestamp(rf.DEFAULTS['validation_start']), pd.Timestamp(rf.DEFAULTS['test_start'])
    test_events = set(d.loc[d.prediction_time.ge(t1), 'event_group'])
    val_events = set(d.loc[d.prediction_time.ge(t0) & d.prediction_time.lt(t1) &
                           ~d.event_group.isin(test_events), 'event_group'])
    d['assigned_partition'] = 'development_train'
    d.loc[d.event_group.isin(val_events), 'assigned_partition'] = 'validation'
    d.loc[d.event_group.isin(test_events), 'assigned_partition'] = 'test'
    d['eligible_conditional'] = d.prediction_time.lt(d.resolved_at)
    d['split'] = 'excluded'
    d.loc[d.assigned_partition.eq('development_train') & d.prediction_time.lt(t0) & d.resolved_at.lt(t0), 'split'] = 'train'
    d.loc[d.assigned_partition.eq('validation') & d.prediction_time.ge(t0) & d.prediction_time.lt(t1) & d.resolved_at.lt(t1), 'split'] = 'validation'
    d.loc[d.assigned_partition.eq('test') & d.prediction_time.ge(t1), 'split'] = 'test'
    d['final_refit'] = ~d.event_group.isin(test_events) & d.prediction_time.lt(t1) & d.resolved_at.lt(t1)
    audit_cols = rf.KEYS + ['prediction_time', 'resolved_at', 'event_group', 'assigned_partition',
                            'eligible_conditional', 'split', 'final_refit']
    d[audit_cols].to_csv(bundle / 'row_split_audit.csv', index=False)
    mapping = d[['market', 'event_group']].drop_duplicates()
    mapping['official_event_ids'] = mapping.event_group.map(lambda g: json.dumps(['official_' + g]))
    mapping.to_csv(bundle / 'official_event_connected_mapping.csv', index=False)
    hist_cols = rf.KEYS + ['prediction_time', 'audit_only_fixed24h_left_truncated'] + [
        c for c in d if c.startswith('feature_fixed24h_') or c in [
            'feature_week_history_buy_fill_rows', 'feature_week_history_avg_buy_ticket',
            'feature_week_history_log_avg_buy_ticket']]
    d[hist_cols].to_parquet(bundle / 'fixed24h_history_features.parquet', index=False)
    derived_only = ['resolved_at', 'event_group', 'assigned_partition', 'eligible_conditional', 'split', 'final_refit']
    candidate = d.drop(columns=derived_only + [c for c in hist_cols if c not in rf.KEYS + ['prediction_time']])
    candidate.to_parquet(data / 'wallet_market_candidates_full_week.parquet', index=False)
    return d


def compare_original_design(package, frame):
    original = json.loads((package / 'reference/original_03_final_validation.ipynb').read_text())
    ns = {'pd': pd, 'np': np, 'EXPERIENCE': ['log_fills', 'log_markets', 'log_buy_dollars'],
          're': __import__('re')}
    for cell_index, wanted in [(12, {'raw_design', 'TrainPreprocessor'}),
                                (14, {'experience_design'}), (16, {'prediction_design'})]:
        tree = ast.parse(''.join(original['cells'][cell_index]['source']))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<original-function>', 'exec'), ns)
    for variant in rf.VARIANTS:
        actual = rf.prediction_design(frame, variant)
        expected = ns['prediction_design'](frame, variant)
        pd.testing.assert_frame_equal(actual, expected)
        # Confirm the logistic comparator reproduces original imputation/scaling.
        pre = ns['TrainPreprocessor']().fit(expected)
        original_x = pre.transform(expected)
        from sklearn.linear_model import LogisticRegression
        direct = LogisticRegression(C=1., solver='lbfgs', max_iter=500, random_state=rf.DEFAULTS['seed'])
        direct.fit(original_x, frame.conditional_profit_label.astype(int))
        pipeline = rf.make_pipeline('logistic', actual, rf.DEFAULTS)
        rf.fit_checked(pipeline, actual, frame.conditional_profit_label)
        np.testing.assert_allclose(direct.predict_proba(original_x)[:, 1],
                                   rf.positive_probability(pipeline, actual), atol=2e-6, rtol=2e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', action='store_true',
                        help='Execute through real Jupyter kernels instead of direct code-cell execution.')
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='wallet_rf_smoke_') as temp:
        temp = Path(temp)
        root, output = temp / 'original_project', temp / 'new_outputs'
        synthetic_project(root)
        env = dict(WALLET_PROJECT_ROOT=str(root), WALLET_RF_OUTPUT_BASE=str(output),
                   WALLET_RF_SMOKE='1', WALLET_RF_RUN_NAME='synthetic_smoke')
        old = {key: os.environ.get(key) for key in env}
        try:
            os.environ.update(env)
            for name in ['01_rf_data_preparation.ipynb', '02_rf_models.ipynb', '03_rf_validation.ipynb',
                         '04_feature_importance_only.ipynb']:
                if name.startswith('04_'):
                    existing_run = output / 'synthetic_smoke'
                    saved_hashes = {p.name: rf.sha(p) for p in (existing_run / 'models').glob('*.joblib')}
                    prediction_hash = rf.sha(existing_run / 'exploratory_test_predictions.csv')
                path = package / '02_notebooks' / name
                nb = nbformat.read(path, as_version=4)
                nbformat.validate(nb)
                print('Executing', name, flush=True)
                if args.kernel:
                    NotebookClient(nb, timeout=600, kernel_name='python3',
                        resources={'metadata': {'path': str(path.parent)}}).execute()
                    nbformat.write(nb, temp / ('executed_' + name))
                else:
                    # Same notebook source, separate namespaces, no network/socket requirement.
                    # Deliberately no fake notebook outputs are written by this route.
                    namespace = {'__name__': '__wallet_smoke_' + name[:2] + '__'}
                    previous_cwd = Path.cwd()
                    try:
                        os.chdir(path.parent)
                        with (temp / (name + '.log')).open('w') as log, contextlib.redirect_stdout(log):
                            for index, cell in enumerate(nb.cells):
                                if cell.cell_type == 'code':
                                    exec(compile(cell.source, f'{name}:cell{index}', 'exec'), namespace)
                    finally:
                        os.chdir(previous_cwd)
                if name.startswith('04_'):
                    assert saved_hashes == {p.name: rf.sha(p) for p in (existing_run / 'models').glob('*.joblib')}
                    assert prediction_hash == rf.sha(existing_run / 'exploratory_test_predictions.csv')
            run = output / 'synthetic_smoke'
            frames = rf.load_frames(run)
            compare_original_design(package, frames['train'])
            all_keys = pd.read_parquet(run / 'all_entry_keys.parquet')
            folds, audit = rf.forward_event_folds(frames['train'], all_keys, {
                **rf.DEFAULTS, 'min_cv_train': 20, 'min_cv_validation': 20})
            assert len(folds) == 2
            assert not any(f.event_group.eq('cross_day_test_event').any() for f in
                           [frames['train'], frames['refit']])
            assert 'cross_day_cv_event' not in set(frames['train'].iloc[folds[0][0]].event_group)
            metrics = pd.read_csv(run / 'exploratory_test_metrics.csv')
            assert len(metrics) == 12 and metrics.rows.nunique() == 1
            assert metrics[['roc_auc', 'log_loss', 'brier']].notna().all().all()
            assert len(list((run / 'models').glob('*.joblib'))) == 10
            required_figures = ['roc_calibration_four_models.png', 'validation_feature_importance.png',
                                'variable_importance_comparison.png']
            assert all((run / 'figures' / name).is_file() for name in required_figures)
            assert (run / 'figures' / 'RF_logistic_resultssummary.html').is_file()
            scores = pd.read_csv(run / 'variable_importance_all_models.csv')
            assert len(scores) == 170 and scores.variant.nunique() == 5
            assert scores.refit_rows.eq(len(frames['refit'])).all()
            assert scores.loc[scores.model.eq('random_forest')].groupby('variant').importance.sum().apply(
                lambda total: np.isclose(total, 1.)).all()
            # Reconstruct logistic predictions using EXPORTED feature names and signs.
            # This detects mislabeling caused by moving missing indicators to the end.
            for variant in rf.VARIANTS:
                model = joblib.load(run / 'models' / f'logistic_{variant}.joblib')
                X = rf.prediction_design(frames['refit'].head(12), variant)
                imputed = model.named_steps['impute'].transform(X)
                scaler = model.named_steps['scale'].named_transformers_['continuous']
                numeric = [i for i, c in enumerate(X) if not c.endswith('__missing')]
                transformed = imputed.copy()
                transformed[:, numeric] = (imputed[:, numeric] - scaler.mean_) / scaler.scale_
                coefficients = scores.loc[scores.model.eq('logistic') & scores.variant.eq(variant)].set_index('feature')
                reconstructed = transformed @ coefficients.loc[list(X), 'coefficient'].to_numpy() + model.named_steps['model'].intercept_[0]
                assert np.allclose(reconstructed, model.decision_function(X), atol=1e-10)
            importance = pd.read_csv(run / 'validation_grouped_permutation_importance.csv')
            assert importance.variant.nunique() == 5
            assert importance.feature_group.str.startswith('entry_price (').sum() == 5
            assert importance.feature_group.eq('ALL_HISTORY_TOGETHER').sum() == 4
            # Deliberate corruptions must fail, rather than silently changing cohorts.
            bad = {name: f.copy() for name, f in frames.items()}
            bad['test'].loc[0, 'event_group'] = bad['train'].event_group.iloc[0]
            try:
                rf.validate_splits(bad, rf.DEFAULTS)
            except ValueError:
                pass
            else:
                raise AssertionError('Failed to reject event leakage.')
            bad = {name: f.copy() for name, f in frames.items()}
            bad['train'].loc[0, 'resolved_at'] = pd.Timestamp(rf.DEFAULTS['validation_start'])
            try:
                rf.validate_splits(bad, rf.DEFAULTS)
            except ValueError:
                pass
            else:
                raise AssertionError('Failed to reject immature training outcome.')
            summary = dict(status='passed', dataset='synthetic only; no actual wallet model fitted',
                execution_mode='jupyter_kernel' if args.kernel else 'notebook_code_cells_in_separate_namespaces',
                executed_notebooks=4, fitted_saved_models=10, evaluated_model_and_reference_rows=12,
                forward_folds=2, original_feature_design_match='all five variants',
                original_logistic_probability_match='within 2e-6 on synthetic fixture',
                individual_importance_rows=170,
                coefficient_mapping='exported named coefficients reconstruct all five logistic decision functions',
                importance_only_update='saved model and test prediction hashes unchanged',
                guards=['event overlap rejected', 'immature training label rejected',
                        'original test groups purged', 'inner CV groups purged'],
                versions=rf.versions())
            print(json.dumps(summary, indent=2))
            report_path = os.environ.get('WALLET_RF_CHECK_REPORT')
            if report_path:
                rf.dump_json(report_path, summary)
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == '__main__':
    main()
