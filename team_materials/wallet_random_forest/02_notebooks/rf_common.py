"""Shared, inspectable helpers for the wallet random-forest notebooks.

The notebooks expose configuration and every analysis stage. This file keeps
identical feature/split definitions in one place; it performs no work on import.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, log_loss, roc_auc_score)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

VARIANTS = ['baseline', 'expanding_3', 'fixed24h_3', 'expanding_5', 'fixed24h_5']
KEYS = ['wallet', 'market']
DEFAULTS = dict(
    seed=20261019, validation_start='2026-10-03T00:00:00Z',
    test_start='2026-10-04T00:00:00Z', full_window_start='2026-09-30T00:00:00Z',
    week_start='2026-09-29T00:00:00Z', n_candidates=8, n_estimators=200,
    n_jobs=min(4, os.cpu_count() or 1), min_cv_train=100, min_cv_validation=100,
    cv_boundaries=['2026-10-01T00:00:00Z', '2026-10-02T00:00:00Z',
                   '2026-10-03T00:00:00Z'],
    permutation_repeats=5, permutation_max_rows=5000, bootstrap_replicates=300,
    threshold=0.5, evaluation_status='Previously inspected test period; exploratory reuse',
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def versions():
    result = {'python': platform.python_version()}
    for package in ['numpy', 'pandas', 'scikit-learn', 'pyarrow', 'joblib']:
        result[package] = importlib.metadata.version(package)
    return result


def row_fingerprint(frame):
    # Exactly the original notebook's ordered key fingerprint.
    return hashlib.sha256(frame[KEYS].to_csv(index=False).encode()).hexdigest()


def key_set_fingerprint(frame):
    return row_fingerprint(frame.sort_values(KEYS, kind='stable'))


def read_bool(series):
    values = series.astype(str).str.lower().str.strip()
    if not values.isin(['true', 'false', '1', '0', '1.0', '0.0']).all():
        raise ValueError('Unrecognized boolean values in ' + str(series.name))
    return values.isin(['true', '1', '1.0'])


def input_file(root, name, explicit=None, required=True):
    """Search only the supplied project, reject genuinely different duplicates."""
    root = Path(root).expanduser().resolve()
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(str(p))
        return p
    search = [root / name, root / 'data' / name,
              root / '02_notebooks' / 'data' / name]
    for folder in ['04_results', 'results', 'server_run']:
        if (root / folder).is_dir():
            search.extend(sorted((root / folder).rglob(name)))
    hits = sorted({p.resolve() for p in search if p.is_file()})
    if len(hits) > 1 and len({sha(p) for p in hits}) > 1:
        raise ValueError(f'Multiple different copies of {name}; set its explicit path:\n' +
                         '\n'.join(map(str, hits)))
    if hits:
        return hits[0]
    if required:
        raise FileNotFoundError(
            f'Missing {name} under {root}. The supplied code-only ZIP omits data. '
            'Use the complete original project; see README.md for required files.')
    return None


def validation_bundle(root, explicit=None):
    """Keep history, split audit and event mapping from one original run."""
    if explicit:
        folders = [Path(explicit).expanduser().resolve()]
    else:
        histories = []
        for folder in ['04_results', 'results', 'server_run']:
            if (Path(root) / folder).is_dir():
                histories += list((Path(root) / folder).rglob('fixed24h_history_features.parquet'))
        folders = sorted({p.parent.resolve() for p in histories})
    required = ['fixed24h_history_features.parquet', 'row_split_audit.csv',
                'official_event_connected_mapping.csv']
    valid = [p for p in folders if all((p / name).is_file() for name in required)]
    if len(valid) > 1:
        fingerprints = {tuple(sha(p / name) for name in required) for p in valid}
        if len(fingerprints) > 1:
            raise ValueError('Multiple validation bundles; set VALIDATION_DIR explicitly: ' + str(valid))
    return valid[0] if valid else None


def rebuild_original_preparation(root, run_dir, original_notebook):
    """Run ONLY original data/evidence/split cells, without logistic fitting.

    The original notebook is copied unchanged in reference/. Large inputs are
    linked into a new staging directory; originals are never edited. Requires
    the candidate table, metadata, and all seven original weekly trade files.
    """
    root, run_dir = Path(root), Path(run_dir)
    names = ['wallet_market_candidates_full_week.parquet', 'weekly_market_metadata_complete.csv']
    inputs = {name: input_file(root, name) for name in names}
    for name in ['settlement_mapping_audit.csv', 'public_mapping_evidence.csv',
                 'settlement_mapping_summary.json', 'pilot_market_metadata.csv']:
        p = input_file(root, name, required=False)
        if p:
            inputs[name] = p
    trade_candidates = [root / '02_notebooks/data/weekly_trades', root / 'data/weekly_trades']
    for folder in ['04_results', 'results']:
        if (root / folder).is_dir():
            trade_candidates += list((root / folder).rglob('weekly_trades'))
    trade_dirs = {p.resolve() for p in trade_candidates
                  if p.is_dir() and len(list(p.glob('trades-*.parquet'))) == 7}
    if len(trade_dirs) != 1:
        raise FileNotFoundError('Need exactly one weekly_trades folder containing seven '
                                'trades-*.parquet files to rebuild histories. Found: ' + str(trade_dirs))
    stage = run_dir / 'original_preparation_stage'
    data = stage / 'data'
    data.mkdir(parents=True, exist_ok=True)
    for name, source in inputs.items():
        target = data / name
        if not target.exists():
            target.symlink_to(source)
    (data / 'weekly_trades').symlink_to(next(iter(trade_dirs)), target_is_directory=True)
    notebook = json.loads(Path(original_notebook).read_text(encoding='utf-8'))
    checks = {4: 'input_path=find_input', 6: 'def stream_fixed_histories',
              8: 'pilot_path=ROOT', 10: 'def event_proxy'}
    ns = {'__name__': '__wallet_original_preparation__'}
    env = {'WALLET_VALIDATION_ROOT': str(stage.resolve()),
           'WALLET_VALIDATION_OUTPUT': str((stage / 'derived').resolve()),
           'WALLET_VALIDATION_MODE': 'production'}
    previous = {key: os.environ.get(key) for key in env}
    try:
        os.environ.update(env)
        for index, marker in checks.items():
            source = ''.join(notebook['cells'][index]['source'])
            if marker not in source:
                raise ValueError('Original preparation notebook differs from the inspected source.')
            print('Executing original preparation cell', index, flush=True)
            exec(compile(source, f'original_03_cell_{index}', 'exec'), ns)
    finally:
        if ns.get('con') is not None:
            ns['con'].close()
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return stage / 'derived'


def prepare_data(root, run_dir, bundle_dir=None, candidate_path=None):
    """Join existing derived features to the ORIGINAL membership audit."""
    run_dir = Path(run_dir)
    candidate = input_file(root, 'wallet_market_candidates_full_week.parquet', candidate_path)
    bundle = validation_bundle(root, bundle_dir)
    if bundle is None:
        raise FileNotFoundError('No complete original validation bundle. Either run original notebook 03 '
                                'or set REBUILD_FROM_RAW=True in notebook 01.')
    paths = {'candidate': candidate, **{name: bundle / name for name in
             ['fixed24h_history_features.parquet', 'row_split_audit.csv',
              'official_event_connected_mapping.csv']}}
    d = pd.read_parquet(candidate)
    hist = pd.read_parquet(paths['fixed24h_history_features.parquet'])
    audit = pd.read_csv(paths['row_split_audit.csv'])
    mapping = pd.read_csv(paths['official_event_connected_mapping.csv'])
    for frame, name in [(d, 'candidate'), (hist, 'histories'), (audit, 'split audit')]:
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError('Missing/duplicate wallet-market keys in ' + name)
        frame['prediction_time'] = pd.to_datetime(frame.prediction_time, utc=True, format='mixed')
    if len(d) != len(audit) or len(d) != len(hist):
        raise ValueError('Candidate/history/audit row counts differ; use a matched original bundle.')
    # Verify timestamps, rather than merging on them and silently dropping mismatches.
    for other, name in [(hist, 'history'), (audit, 'audit')]:
        check = d[KEYS + ['prediction_time']].merge(
            other[KEYS + ['prediction_time']], on=KEYS, validate='one_to_one', suffixes=('', '_other'))
        if len(check) != len(d) or not check.prediction_time.eq(check.prediction_time_other).all():
            raise ValueError('Key/timestamp mismatch with ' + name)
    history_cols = [c for c in hist if c not in KEYS + ['prediction_time']]
    for c in history_cols:
        if c in d:
            pair = d[KEYS + [c]].merge(hist[KEYS + [c]], on=KEYS, suffixes=('_a', '_b'))
            if not np.allclose(pair[c + '_a'], pair[c + '_b'], equal_nan=True):
                raise ValueError('Different pre-existing history feature: ' + c)
    d = d.drop(columns=[c for c in history_cols if c in d]).merge(
        hist[KEYS + history_cols], on=KEYS, how='left', validate='one_to_one', sort=False)
    audit_cols = ['resolved_at', 'event_group', 'assigned_partition', 'eligible_conditional',
                  'split', 'final_refit']
    d = d.drop(columns=[c for c in audit_cols if c in d]).merge(
        audit[KEYS + audit_cols], on=KEYS, how='left', validate='one_to_one', sort=False)
    d['resolved_at'] = pd.to_datetime(d.resolved_at, utc=True, format='mixed', errors='coerce')
    for c in ['eligible_conditional', 'final_refit']:
        d[c] = read_bool(d[c])
    if mapping.market.duplicated().any():
        raise ValueError('Duplicate markets in original event mapping.')
    d = d.merge(mapping[['market', 'event_group']].rename(columns={'event_group': '_mapped_event'}),
                on='market', how='left', validate='many_to_one', sort=False)
    if not d.event_group.eq(d._mapped_event).all():
        raise ValueError('The event mapping and row audit come from different original runs.')
    d = d.drop(columns='_mapped_event')
    if 'feature_week_history_latest_time' in d:
        last = pd.to_datetime(d.feature_week_history_latest_time, utc=True, format='mixed')
        if not (last.isna() | last.lt(d.prediction_time)).all():
            raise ValueError('Wallet history includes the opening timestamp or a future trade.')
    full = pd.to_numeric(d.audit_only_fixed24h_left_truncated, errors='raise').eq(0)
    frames = {name: d.loc[mask & full].copy().reset_index(drop=True) for name, mask in {
        'train': d.split.eq('train'), 'validation': d.split.eq('validation'),
        'test': d.split.eq('test'), 'refit': d.final_refit}.items()}
    validate_splits(frames, DEFAULTS, mapping)
    for variant in VARIANTS:
        for frame in frames.values():
            prediction_design(frame, variant)  # fail early on a missing source feature
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_parquet(run_dir / f'{name}.parquet', index=False)
    # CV event allocation uses all original candidate entry times, before target filtering.
    d[['wallet', 'market', 'prediction_time', 'event_group', 'assigned_partition']].to_parquet(
        run_dir / 'all_entry_keys.parquet', index=False)
    mapping.to_csv(run_dir / 'event_mapping.csv', index=False)
    summary = cohort_summary(frames)
    summary.to_csv(run_dir / 'cohort_summary.csv', index=False)
    prepared_names = [f'{name}.parquet' for name in frames] + ['all_entry_keys.parquet', 'event_mapping.csv']
    manifest = {'inputs': {k: {'path': str(v), 'sha256': sha(v)} for k, v in paths.items()},
                'prepared_files': {name: sha(run_dir / name) for name in prepared_names},
                'source_bundle': str(bundle), 'candidate_rows': len(d),
                'candidate_known_conditional_labels': int(d.conditional_profit_label.notna().sum()),
                'complete_24h_only': True, 'versions': versions(),
                'cohorts': summary.to_dict('records'),
                'limitations': ['Conditional gross-profit labels, not verified net-profit labels.',
                    'Recorded resolution timestamps are not verified result-publication timestamps.',
                    'Event groups use original conservative unions and matched official IDs.',
                    'Repeated wallets may occur in different event groups.',
                    'The final test dates have already been inspected in the original project.']}
    dump_json(run_dir / 'data_manifest.json', manifest)
    return frames, manifest


def load_frames(run_dir):
    return {name: pd.read_parquet(Path(run_dir) / f'{name}.parquet')
            for name in ['train', 'validation', 'test', 'refit']}


def verify_prepared_files(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'data_manifest.json').read_text())
    for name, expected in manifest['prepared_files'].items():
        if sha(run_dir / name) != expected:
            raise ValueError('Prepared cohort file changed after data preparation: ' + name)


def cohort_summary(frames):
    return pd.DataFrame([dict(cohort=name, rows=len(f), wallets=f.wallet.nunique(),
        markets=f.market.nunique(), event_groups=f.event_group.nunique(),
        positive_rate=f.conditional_profit_label.mean(), first_entry=f.prediction_time.min(),
        last_entry=f.prediction_time.max(), ordered_keys_sha256=row_fingerprint(f),
        sorted_keys_sha256=key_set_fingerprint(f)) for name, f in frames.items()])


def validate_splits(frames, config, mapping=None):
    t0, t1 = pd.Timestamp(config['validation_start']), pd.Timestamp(config['test_start'])
    for name, f in frames.items():
        if f.empty or f.duplicated(KEYS).any():
            raise ValueError('Empty/duplicate cohort: ' + name)
        if not f.conditional_profit_label.isin([0, 1]).all():
            raise ValueError('Nonbinary or missing outcome in ' + name)
        if not f.prediction_time.lt(f.resolved_at).all():
            raise ValueError('Entry at/after recorded resolution in ' + name)
        if not pd.to_numeric(f.audit_only_fixed24h_left_truncated).eq(0).all():
            raise ValueError('Incomplete 24-hour history in ' + name)
        if not f.prediction_time.ge(pd.Timestamp(config['full_window_start'])).all():
            raise ValueError('Entry before full 24-hour coverage in ' + name)
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test'), ('refit', 'test')]:
        for col in ['market', 'event_group']:
            if set(frames[a][col]) & set(frames[b][col]):
                raise ValueError(f'{col} overlap: {a}/{b}')
    train, val, test, refit = (frames[n] for n in ['train', 'validation', 'test', 'refit'])
    if not (train.prediction_time.lt(t0).all() and train.resolved_at.lt(t0).all()
            and val.prediction_time.ge(t0).all() and val.prediction_time.lt(t1).all()
            and val.resolved_at.lt(t1).all() and test.prediction_time.ge(t1).all()
            and refit.prediction_time.lt(t1).all() and refit.resolved_at.lt(t1).all()):
        raise ValueError('Original chronological/label-maturity boundaries failed.')
    if mapping is not None and 'official_event_ids' in mapping:
        def ids(f):
            relevant = mapping[mapping.market.isin(f.market)]
            return {str(x) for raw in relevant.official_event_ids.dropna()
                    for x in json.loads(raw)}
        for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test'), ('refit', 'test')]:
            if ids(frames[a]) & ids(frames[b]):
                raise ValueError('Known official event crosses a split: ' + a + '/' + b)


def prediction_design(frame, variant='baseline'):
    """Exact prediction-design formulas in original notebook 03, cells 12/16.

    Explicit allowlist: wallet/market identifiers, outcomes, payouts, resolutions,
    audits and final/full-week trading summaries cannot enter the design.
    """
    if variant not in VARIANTS:
        raise ValueError(variant)
    X = pd.DataFrame(index=frame.index)
    X['entry_log_buy'] = frame.feature_entry_log_buy_dollars
    p = frame.feature_entry_buy_vwap
    X['entry_price'] = p
    X['entry_price_squared'] = p ** 2
    X['entry_price_cubed'] = p ** 3
    X['entry_concentration'] = frame.feature_entry_direction_concentration
    X['entry_log_fill_rows'] = np.log1p(frame.feature_entry_fill_rows)
    X['market_log_dollars'] = frame.feature_strict_prior_market_log_dollars
    X['market_log_wallets'] = frame.feature_strict_prior_market_log_wallets
    X['market_log_observed_age'] = np.log1p(frame.feature_observed_market_age_hours.clip(lower=0))
    X['clock_sin'] = np.sin(2 * np.pi * frame.feature_entry_utc_hour / 24)
    X['clock_cos'] = np.cos(2 * np.pi * frame.feature_entry_utc_hour / 24)
    X['window_exposure_days'] = (frame.prediction_time - pd.Timestamp('2026-09-29', tz='UTC')).dt.total_seconds() / 86400
    X['entry_price__missing'] = X.entry_price.isna().astype(float)
    if variant != 'baseline':
        period, size = variant.split('_')
        source = 'feature_week_history_' if period == 'expanding' else 'feature_fixed24h_'
        for target, suffix in [('log_fills', 'log_fill_rows'), ('log_markets', 'log_markets'),
                               ('log_buy_dollars', 'log_buy_dollars')]:
            X['history__' + target] = frame[source + suffix]
        if size == '5':
            X['history__buy_hhi'] = frame[source + 'buy_market_hhi']
            X['history__log_avg_buy_ticket'] = frame[source + 'log_avg_buy_ticket']
            for name in ['history__buy_hhi', 'history__log_avg_buy_ticket']:
                X[name + '__missing'] = X[name].isna().astype(float)
    X = X.astype(float)
    if np.isinf(X.to_numpy()).any():
        raise ValueError('Infinite feature value; inspect source data instead of silently replacing it.')
    return X


def make_pipeline(kind, X, config, params=None):
    # Fit imputation separately in every CV fold. keep_empty_features reproduces
    # the original all-missing-column fallback of 0 and preserves column order.
    steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if kind == 'logistic':
        numeric = [i for i, c in enumerate(X) if not c.endswith('__missing')]
        flags = [i for i, c in enumerate(X) if c.endswith('__missing')]
        steps.append(('scale', ColumnTransformer([
            ('continuous', StandardScaler(), numeric), ('missing_flags', 'passthrough', flags)])))
        estimator = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500,
                                       random_state=config['seed'])
    elif kind == 'random_forest':
        estimator = RandomForestClassifier(n_estimators=config['n_estimators'],
            random_state=config['seed'], n_jobs=config['n_jobs'], class_weight=None,
            bootstrap=True, criterion='gini', **(params or {}))
    else:
        raise ValueError(kind)
    return Pipeline(steps + [('model', estimator)])


def fit_checked(model, X, y):
    if pd.Series(y).nunique() != 2:
        raise ValueError('Training cohort must contain both outcome classes.')
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(X, np.asarray(y, dtype=int))
    convergence = [w for w in recorded if issubclass(w.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError('Logistic reference did not converge. Increase max_iter before evaluation: ' +
                           '; '.join(str(w.message) for w in convergence))
    return model


def positive_probability(model, X):
    classes = list(model.named_steps['model'].classes_)
    return model.predict_proba(X)[:, classes.index(1)]


def probability_metrics(y, p, threshold=0.5):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    if len(p) != len(y) or not np.isfinite(p).all():
        raise ValueError('Invalid probability vector.')
    return dict(rows=len(y), positive_rate=float(y.mean()), log_loss=log_loss(y, p, labels=[0, 1]),
        brier=brier_score_loss(y, p), roc_auc=roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        pr_auc_average_precision=average_precision_score(y, p) if y.sum() else np.nan,
        accuracy_threshold_0_5=accuracy_score(y, p >= threshold),
        mean_prediction=float(p.mean()), calibration_mean_error=float(p.mean() - y.mean()))


def forward_event_folds(train, all_keys, config):
    """Two daily forward folds inside development train (not the October 3 check).

    Assign validation event groups using ALL entry keys before looking at labels.
    Purge their earlier members from training, then enforce recorded resolution
    strictly before the validation start (and before fold end for validation).
    """
    allowed = set(train.event_group)
    universe = all_keys[all_keys.event_group.isin(allowed)]
    boundaries = [pd.Timestamp(t) for t in config['cv_boundaries']]
    if boundaries != sorted(set(boundaries)) or boundaries[-1] > pd.Timestamp(config['validation_start']):
        raise ValueError('CV boundaries must increase and finish by the outer validation start.')
    folds, rows = [], []
    for number, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
        validation_events = set(universe.loc[universe.prediction_time.ge(start) &
                                             universe.prediction_time.lt(end), 'event_group'])
        a = train.prediction_time.lt(start) & train.resolved_at.lt(start) & ~train.event_group.isin(validation_events)
        b = train.prediction_time.ge(start) & train.prediction_time.lt(end) & train.resolved_at.lt(end)
        ai, bi = np.flatnonzero(a), np.flatnonzero(b)
        accepted = (len(ai) >= config['min_cv_train'] and len(bi) >= config['min_cv_validation']
                    and train.iloc[ai].conditional_profit_label.nunique() == 2)
        rows.append(dict(fold=number, validation_start=start, validation_end=end,
                         training_rows=len(ai), validation_rows=len(bi), accepted=accepted,
                         training_events=train.iloc[ai].event_group.nunique(),
                         validation_events=train.iloc[bi].event_group.nunique(),
                         training_keys_sha256=row_fingerprint(train.iloc[ai]),
                         validation_keys_sha256=row_fingerprint(train.iloc[bi])))
        if accepted:
            if set(train.iloc[ai].event_group) & set(train.iloc[bi].event_group):
                raise AssertionError('CV event overlap.')
            if not train.iloc[ai].resolved_at.lt(start).all():
                raise AssertionError('CV training outcome not mature at fold start.')
            folds.append((ai, bi))
    audit = pd.DataFrame(rows)
    if len(folds) < 2:
        raise ValueError('Fewer than two usable forward folds. Obtain a longer period or deliberately '
                         'configure fixed RF settings and a separate validation design. '
                         'Do not fall back silently to shuffled CV.\n' + audit.to_string(index=False))
    return folds, audit


def tune_variant(train, variant, folds, candidates, config):
    X, y = prediction_design(train, variant), train.conditional_profit_label.astype(int)
    records = []
    for candidate_id, params in enumerate(candidates):
        print(f'{variant}: candidate {candidate_id + 1}/{len(candidates)}', flush=True)
        for fold_id, (a, b) in enumerate(folds, 1):
            model = make_pipeline('random_forest', X, config, params)
            fit_checked(model, X.iloc[a], y.iloc[a])
            metrics = probability_metrics(y.iloc[b], positive_probability(model, X.iloc[b]))
            records.append(dict(variant=variant, candidate_id=candidate_id, fold=fold_id,
                                params_json=json.dumps(params, sort_keys=True), **metrics))
    details = pd.DataFrame(records)
    # Row-weighted held-forward log loss, fixed before any final test evaluation.
    summary = details.groupby(['candidate_id', 'params_json'], as_index=False).apply(
        lambda f: pd.Series(dict(cv_log_loss=np.average(f.log_loss, weights=f.rows),
                                 cv_brier=np.average(f.brier, weights=f.rows),
                                 cv_mean_roc_auc=f.roc_auc.mean())), include_groups=False)
    summary = summary.sort_values(['cv_log_loss', 'candidate_id'], kind='stable').reset_index(drop=True)
    best_id = int(summary.iloc[0].candidate_id)
    return candidates[best_id], details, summary


def importance_groups(columns):
    """Jointly permute derived columns of one underlying feature."""
    groups, used = {}, set()
    for name, members in {
        'entry_price (p, p2, p3, missing)': ['entry_price', 'entry_price_squared', 'entry_price_cubed', 'entry_price__missing'],
        'hour_of_day (sin, cos)': ['clock_sin', 'clock_cos'],
        'history_concentration (incl. missing)': ['history__buy_hhi', 'history__buy_hhi__missing'],
        'history_avg_buy_size (incl. missing)': ['history__log_avg_buy_ticket', 'history__log_avg_buy_ticket__missing'],
    }.items():
        present = [c for c in members if c in columns]
        if present:
            groups[name] = present
            used.update(present)
    for c in columns:
        if c not in used:
            groups[c] = [c]
    return groups


def grouped_permutation(model, frame, variant, config):
    # Descriptive validation-set model reliance, not an inferential p-value.
    X = prediction_design(frame, variant)
    if len(X) > config['permutation_max_rows']:
        idx = np.sort(np.random.default_rng(config['seed']).choice(
            len(X), config['permutation_max_rows'], replace=False))
        X = X.iloc[idx].copy()
        y = frame.conditional_profit_label.iloc[idx].to_numpy(dtype=int)
    else:
        X = X.copy()
        y = frame.conditional_profit_label.to_numpy(dtype=int)
    baseline = probability_metrics(y, positive_probability(model, X))['log_loss']
    groups = importance_groups(X.columns)
    history = [c for c in X if c.startswith('history__')]
    if history:
        groups['ALL_HISTORY_TOGETHER'] = history
    rng = np.random.default_rng(config['seed'])
    permutations = [rng.permutation(len(X)) for _ in range(config['permutation_repeats'])]
    result = []
    for name, members in groups.items():
        increases = []
        for index in permutations:
            shuffled = X.copy()
            shuffled.loc[:, members] = X.iloc[index][members].to_numpy()
            loss = probability_metrics(y, positive_probability(model, shuffled))['log_loss']
            increases.append(loss - baseline)
        result.append(dict(variant=variant, feature_group=name, columns_json=json.dumps(members),
                           mean_log_loss_increase=float(np.mean(increases)),
                           shuffle_sd=float(np.std(increases, ddof=1)) if len(increases) > 1 else 0.0,
                           evaluated_rows=len(X), repeats=len(increases),
                           interpretation='Positive means worse validation loss after shuffling; SD is not a CI.'))
    return pd.DataFrame(result).sort_values('mean_log_loss_increase', ascending=False)


def fit_models(run_dir, config):
    """Tune on inner forward folds, inspect outer validation, refit before test."""
    run_dir = Path(run_dir)
    if (run_dir / 'evaluation_complete.json').exists():
        raise RuntimeError('This run has already evaluated test outcomes. Start a new exploratory run '
                           'instead of overwriting model choices after seeing test results.')
    verify_prepared_files(run_dir)
    frames = load_frames(run_dir)
    validate_splits(frames, config, pd.read_csv(run_dir / 'event_mapping.csv'))
    train, val, refit = (frames[name] for name in ['train', 'validation', 'refit'])
    all_keys = pd.read_parquet(run_dir / 'all_entry_keys.parquet')
    folds, fold_audit = forward_event_folds(train, all_keys, config)
    fold_audit.to_csv(run_dir / 'forward_cv_audit.csv', index=False)
    candidates = list(ParameterSampler(dict(max_depth=[6, 12, None], min_samples_leaf=[10, 25, 50],
                                             max_features=['sqrt', 0.7, 1.0]),
                                       n_iter=config['n_candidates'], random_state=config['seed']))
    dump_json(run_dir / 'candidate_grid.json', candidates)
    model_dir = run_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    selections, validation_rows, tuning_rows, importance_rows, mdi_rows, fit_rows = {}, [], [], [], [], []
    for variant in VARIANTS:
        best, details, summary = tune_variant(train, variant, folds, candidates, config)
        details.to_csv(run_dir / f'cv_folds_{variant}.csv', index=False)
        summary.to_csv(run_dir / f'cv_candidates_{variant}.csv', index=False)
        tuning_rows.extend(details.to_dict('records'))
        selections[variant] = best
        for kind in ['random_forest', 'logistic']:
            X = prediction_design(train, variant)
            dev = make_pipeline(kind, X, config, best if kind == 'random_forest' else None)
            fit_checked(dev, X, train.conditional_profit_label)
            pv = positive_probability(dev, prediction_design(val, variant))
            validation_rows.append(dict(model=kind, variant=variant, evaluation='outer_validation_reused',
                                       **probability_metrics(val.conditional_profit_label, pv)))
            if kind == 'random_forest':
                imp = grouped_permutation(dev, val, variant, config)
                importance_rows.extend(imp.to_dict('records'))
            final = make_pipeline(kind, prediction_design(refit, variant), config,
                                  best if kind == 'random_forest' else None)
            fit_checked(final, prediction_design(refit, variant), refit.conditional_profit_label)
            joblib.dump(final, model_dir / f'{kind}_{variant}.joblib', compress=3)
            fit_rows.append(dict(model=kind, variant=variant, development_rows=len(train),
                refit_rows=len(refit), refit_keys_sha256=row_fingerprint(refit),
                feature_names=list(prediction_design(refit, variant)),
                parameters=final.named_steps['model'].get_params(),
                preprocessing='training-only median imputation; logistic continuous terms scaled'))
            if kind == 'random_forest':
                mdi_rows.extend(dict(variant=variant, feature=c, impurity_importance=float(v))
                    for c, v in zip(prediction_design(refit, variant), final.named_steps['model'].feature_importances_))
        print('Completed', variant, flush=True)
    pd.DataFrame(tuning_rows).to_csv(run_dir / 'all_cv_results.csv', index=False)
    pd.DataFrame(validation_rows).to_csv(run_dir / 'validation_metrics.csv', index=False)
    pd.DataFrame(importance_rows).to_csv(run_dir / 'validation_grouped_permutation_importance.csv', index=False)
    pd.DataFrame(mdi_rows).to_csv(run_dir / 'rf_impurity_importance.csv', index=False)
    dump_json(run_dir / 'selected_parameters.json', selections)
    dump_json(run_dir / 'model_manifest.json', dict(config=config, versions=versions(), fits=fit_rows,
        selection_rule='Lowest row-weighted forward-CV log loss per variant; deterministic candidate-id tie break.',
        source_data_manifest_sha256=sha(run_dir / 'data_manifest.json'),
        model_files={p.name: sha(p) for p in sorted(model_dir.glob('*.joblib'))}))
    # New primary importance tables use final saved models, not test outcomes.
    export_individual_importance(run_dir)
    return pd.DataFrame(validation_rows), selections


def export_individual_importance(run_dir):
    """Export standardized logistic coefficients and individual RF Gini scores."""
    from feature_importance import export_tables
    scores, _ = export_tables(Path(run_dir))
    return scores


def make_importance_figures(run_dir):
    """Generate importance comparisons without fitting or loading test outcomes.

    Safe to use for existing, already evaluated runs: only descriptive importance
    exports change. Saved models, selections, predictions and metrics are untouched.
    """
    from feature_importance import build
    run_dir = Path(run_dir)
    export_individual_importance(run_dir)
    build(run_dir, run_dir / 'figures')
    return run_dir / 'figures'


def paired_event_delta(frame, reference, alternative, config):
    y = frame.conditional_profit_label.to_numpy(dtype=float)
    r, a = np.clip(reference, 1e-12, 1 - 1e-12), np.clip(alternative, 1e-12, 1 - 1e-12)
    grouped = pd.DataFrame(dict(event=frame.event_group.to_numpy(), n=1,
        delta=-(y * np.log(a) + (1-y) * np.log(1-a)) + y * np.log(r) + (1-y) * np.log(1-r))).groupby('event').sum()
    rng = np.random.default_rng(config['seed'])
    samples = []
    for _ in range(config['bootstrap_replicates']):
        draw = grouped.iloc[rng.integers(0, len(grouped), len(grouped))]
        samples.append(draw.delta.sum() / draw.n.sum())
    shares = grouped.n / grouped.n.sum()
    return dict(alternative_minus_reference_log_loss=float(grouped.delta.sum() / grouped.n.sum()),
        descriptive_low=float(np.quantile(samples, .025)), descriptive_high=float(np.quantile(samples, .975)),
        event_groups=len(grouped), effective_event_groups=float(1 / (shares ** 2).sum()),
        largest_event_share=float(shares.max()),
        interpretation='Negative favors alternative. Exploratory event-only bootstrap; repeated-wallet dependence not resampled.')


def evaluate_test(run_dir, config):
    """Evaluate frozen saved models; never tune or choose features here."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'model_manifest.json').read_text())
    if manifest['config'] != config:
        raise ValueError('Evaluation config differs from the frozen model manifest.')
    for name, expected in manifest['model_files'].items():
        if sha(run_dir / 'models' / name) != expected:
            raise ValueError('A saved model changed after selection: ' + name)
    if sha(run_dir / 'data_manifest.json') != manifest['source_data_manifest_sha256']:
        raise ValueError('Data manifest changed after model fitting.')
    verify_prepared_files(run_dir)
    frames = load_frames(run_dir)
    validate_splits(frames, config, pd.read_csv(run_dir / 'event_mapping.csv'))
    test, refit = frames['test'], frames['refit']
    predictions = test[KEYS + ['prediction_time', 'event_group', 'conditional_profit_label']].copy()
    metrics = []
    for kind in ['logistic', 'random_forest']:
        for variant in VARIANTS:
            # Load only models created locally by notebook 02, never untrusted pickle files.
            model = joblib.load(run_dir / 'models' / f'{kind}_{variant}.joblib')
            p = positive_probability(model, prediction_design(test, variant))
            predictions[kind + '__' + variant] = p
            metrics.append(dict(model=kind, variant=variant, evaluation=config['evaluation_status'],
                                **probability_metrics(test.conditional_profit_label, p)))
    for name, p in [('training_prevalence', np.repeat(refit.conditional_profit_label.mean(), len(test))),
                    ('always_profitable', np.ones(len(test)))]:
        predictions[name] = p
        metrics.append(dict(model='reference', variant=name, evaluation=config['evaluation_status'],
                            **probability_metrics(test.conditional_profit_label, p)))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(run_dir / 'exploratory_test_metrics.csv', index=False)
    predictions.to_csv(run_dir / 'exploratory_test_predictions.csv', index=False)
    comparisons = [('random_forest__baseline', 'random_forest__' + v) for v in VARIANTS[1:]]
    comparisons += [('logistic__' + v, 'random_forest__' + v) for v in VARIANTS]
    deltas = pd.DataFrame([dict(reference=r, alternative=a,
        **paired_event_delta(test, predictions[r].to_numpy(), predictions[a].to_numpy(), config))
        for r, a in comparisons])
    deltas.to_csv(run_dir / 'paired_event_log_loss_differences.csv', index=False)
    dump_json(run_dir / 'evaluation_complete.json', dict(
        status=config['evaluation_status'], threshold=config['threshold'],
        test_rows=len(test), test_keys_sha256=row_fingerprint(test),
        test_wallets_previously_seen_fraction=float(test.wallet.isin(refit.wallet).mean()),
        fixed_model_manifest_sha256=sha(run_dir / 'model_manifest.json'),
        metrics_sha256=sha(run_dir / 'exploratory_test_metrics.csv'),
        test_was_used_for_tuning=False,
        note='A new run on these dates remains exploratory; rerunning cannot make this test fresh.'))
    return metrics, predictions, deltas


def make_figures(run_dir):
    import matplotlib.pyplot as plt
    from comparison_summary import make_performance_figures, write_summary
    run_dir = Path(run_dir)
    folder = run_dir / 'figures'
    folder.mkdir(exist_ok=True)
    make_performance_figures(run_dir, folder)
    importance = pd.read_csv(run_dir / 'validation_grouped_permutation_importance.csv')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, variant in zip(axes, ['baseline', 'fixed24h_5']):
        f = importance[importance.variant.eq(variant)].sort_values('mean_log_loss_increase').tail(15)
        ax.barh(f.feature_group, f.mean_log_loss_increase, xerr=f.shuffle_sd, color='#3268a8')
        ax.axvline(0, color='gray', linewidth=1)
        ax.set(title=variant, xlabel='Increase in validation log loss after joint shuffling')
    fig.suptitle('Validation feature importance; error bars = shuffle SD, not confidence intervals', fontsize=10)
    fig.tight_layout()
    fig.savefig(folder / 'validation_feature_importance.png', dpi=160)
    plt.close(fig)
    make_importance_figures(run_dir)
    write_summary(run_dir, folder)
    return folder
