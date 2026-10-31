"""Extract all supplied market results and reconstruct the original model inputs.

Run locally from this folder with the seven daily Parquets under data/weekly_trades:
  python prepare_full_model_inputs.py --metadata /path/to/market_results.csv

The matching settlement export is already broad: no new database query is needed
to reproduce its snapshot. Unrecorded results stay unknown. Conditional gross
labels are not verified net profit. All time boundaries are UTC.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import time
import gc
import shutil

import duckdb
import numpy as np
import pandas as pd

from original_feature_algorithms import build_weekly_entry_history, stream_fixed_histories, fixed_history_fixture, event_proxy

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['wallet', 'market']
START = pd.Timestamp('2026-09-29', tz='UTC')
END = pd.Timestamp('2026-10-06', tz='UTC')
T0, T1 = pd.Timestamp('2026-10-03', tz='UTC'), pd.Timestamp('2026-10-04', tz='UTC')


def stamp(values):
    return pd.to_datetime(values, utc=True, format='mixed', errors='coerce')


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_outcomes(value):
    try:
        a = json.loads(value)
    except (TypeError, ValueError):
        return []
    return a if isinstance(a, list) and all(isinstance(x, str) for x in a) else []


def extract_all_results(metadata_path, root=ROOT, require_original_counts=False):
    root = Path(root)
    folder = root / '04_results/market_results'
    folder.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(metadata_path)
    s = pd.read_csv(root / 'data/market_trade_summary.csv')
    if m.market.isna().any() or m.market.duplicated().any():
        raise ValueError('Missing or duplicate settlement market keys.')
    if set(m.market) != set(s.market):
        raise ValueError('Settlement export does not cover the exact observed market-slug list.')
    if len(m) != 78543 or s.n_trades.sum() != 6815814:
        raise ValueError('This reconstruction expects the supplied full seven-day universe.')
    m['resolved_at'] = stamp(m.resolved_at)
    m['start_date'] = stamp(m.start_date)
    labels = m.outcomes.map(read_outcomes)
    binary = labels.map(lambda a: len(a) == 2 and len(set(a)) == 2)
    known = m.resolved_yes.notna() & m.resolved_at.notna() & binary & m.price_at_close.isin([0, 1])
    # Check the database export's own ordered-outcome mapping rather than guessing
    # from trade prices. This is a consistency check, not independent on-chain proof.
    flags = m.resolved_yes.astype('string').str.strip().str.lower()
    if not flags[m.resolved_yes.notna()].isin(['true', 'false']).all():
        raise ValueError('Unexpected resolved_yes encoding.')
    payout = flags.map({'true': 1.0, 'false': 0.0})
    if not np.allclose(m.loc[known, 'price_at_close'], payout[known]):
        raise ValueError('resolved_yes and price_at_close disagree.')
    reconstructed = pd.Series(pd.NA, index=m.index, dtype='string')
    for i in m.index[known]:
        reconstructed[i] = labels[i][0 if payout[i] == 1 else 1]
    comparison = m.winning_outcome.astype('string').str.strip().str.lower()
    if not reconstructed[known].str.strip().str.lower().eq(comparison[known]).all():
        raise ValueError('Supplied winning_outcome contradicts ordered outcomes.')
    m['result_available_for_accounting'] = known
    m['result_availability_reason'] = np.where(known, 'recorded_binary_settlement',
        np.where(m.metadata_found.eq(False), 'market_metadata_not_found', 'no_recorded_usable_settlement'))
    d = s.merge(m, on='market', validate='one_to_one')
    inside = (d.result_available_for_accounting & d.start_date.ge(START) & d.start_date.lt(END)
              & d.resolved_at.ge(d.start_date) & d.resolved_at.lt(END))
    # These date/observed-resolution filters recover the source's exact universe.
    candidate = inside & stamp(d.last_observed_trade_utc).le(d.resolved_at)
    selected = d[candidate].copy()
    obs = selected.observed_outcomes_json.map(read_outcomes)
    matches = [set(x.strip().lower() for x in a).issubset(set(x.strip().lower() for x in b))
               for a, b in zip(obs, selected.outcomes.map(read_outcomes))]
    selected['trade_outcome_names_match'] = matches
    original_counts_match = (len(selected) == 2661 and selected.n_trades.sum() == 239434
                             and selected.n_wallets.sum() == 96184)
    if require_original_counts:
        if not all(matches) or not selected.invalid_trade_rows.eq(0).all():
            raise ValueError('Candidate outcome labels or trade validity need investigation.')
        if not original_counts_match:
            raise ValueError('The refreshed candidate universe differs from the original study. '
                             'Full results can be exported, but deliberately revise the modeling cohort before refitting.')
    m.to_csv(root / 'data/weekly_market_metadata_complete.csv', index=False)
    selected.to_csv(root / 'data/candidate_markets_preliminary.csv', index=False)
    selected[selected.n_wallets.ge(100)].to_csv(root / 'data/candidate_markets_100plus_wallets.csv', index=False)
    m[known].to_csv(folder / 'recorded_market_results.csv', index=False)
    m.loc[~known, ['market', 'metadata_found', 'has_resolution', 'result_availability_reason']].to_csv(
        folder / 'markets_without_recorded_results.csv', index=False)
    coverage = m[['market', 'metadata_found', 'has_resolution', 'winning_outcome', 'resolved_at',
                  'result_available_for_accounting', 'result_availability_reason']].copy()
    coverage['in_original_candidate_universe'] = coverage.market.isin(selected.market)
    coverage.to_csv(folder / 'all_market_result_coverage.csv', index=False)
    summary = dict(markets_requested=len(s), exported_unique_markets=len(m), missing_market_rows=0,
        metadata_found=int(m.metadata_found.sum()), metadata_missing=int((~m.metadata_found).sum()),
        recorded_resolutions=int(m.has_resolution.sum()), usable_binary_results=int(known.sum()),
        no_usable_recorded_result=int((~known).sum()), original_candidate_markets=len(selected),
        original_candidate_trades=int(selected.n_trades.sum()), candidate_wallet_market_pairs=int(selected.n_wallets.sum()),
        reproduces_original_candidate_counts=bool(original_counts_match),
        candidate_outcome_name_mismatches=int((~selected.trade_outcome_names_match).sum()),
        candidates_with_100plus_wallets=int(selected.n_wallets.ge(100).sum()),
        excluded_observed_trades_after_resolution=int((inside & ~candidate).sum()),
        extracted_at_min=str(pd.to_datetime(m.extracted_at, utc=True).min()),
        extracted_at_max=str(pd.to_datetime(m.extracted_at, utc=True).max()),
        source_metadata_sha256=sha(metadata_path),
        result_mapping_check='Database resolved_yes, price_at_close, ordered outcomes and winner agree; not independent on-chain verification.',
        retained_after_week_resolutions=True, missing_results_recoded_as_losses=False)
    dump(folder / 'coverage_summary.json', summary)
    print('Market coverage:', json.dumps(summary, indent=2), flush=True)
    return m, selected


def build_inputs(metadata_path, root=ROOT):
    began = time.perf_counter()
    root = Path(root)
    data = root / 'data'
    base_out, history_out = root / '04_results/01_data_preparation', root / '04_results/03_history_validation'
    base_out.mkdir(parents=True, exist_ok=True)
    history_out.mkdir(parents=True, exist_ok=True)
    m, candidates = extract_all_results(metadata_path, root, require_original_counts=True)
    raw_paths = sorted((data / 'weekly_trades').glob('trades-*.parquet'))
    if len(raw_paths) != 7:
        raise FileNotFoundError('Need all seven daily trade parquets under data/weekly_trades.')
    spill = base_out / 'duckdb_spill'
    spill.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute('SET threads=2')
    con.execute("SET memory_limit='2GB'")
    con.execute("SET TimeZone='UTC'")
    con.execute("SET temp_directory='" + str(spill).replace("'", "''") + "'")
    con.from_parquet([str(p) for p in raw_paths]).create_view('raw_week')
    con.register('candidate_keys', candidates[['market']])
    ct = con.execute('SELECT t.* FROM raw_week t JOIN candidate_keys USING(market)').df()
    ct['time'] = stamp(ct.time)
    assert len(ct) == 239434
    mapping = []
    for r in candidates.itertuples(index=False):
        for i, outcome in enumerate(read_outcomes(r.outcomes)):
            mapping.append(dict(market=r.market, outcome=outcome, outcome_index=i,
                                payout=float(r.price_at_close if i == 0 else 1-r.price_at_close)))
    ct = ct.merge(pd.DataFrame(mapping), on=['market', 'outcome'], how='left', validate='many_to_one')
    if ct.outcome_index.isna().any():
        raise ValueError('Exact candidate trade/metadata outcome names differ.')
    ct['signed_shares'] = np.where(ct.side.eq('BUY'), ct['size'], -ct['size'])
    ct['buy_dollars'] = np.where(ct.side.eq('BUY'), ct.dollars, 0.)
    ct['sell_dollars'] = np.where(ct.side.eq('SELL'), ct.dollars, 0.)
    ct['cashflow'] = ct.sell_dollars - ct.buy_dollars
    ct['contribution'] = ct.cashflow + ct.signed_shares * ct.payout
    d = ct.groupby(KEYS).agg(prediction_time=('time', 'min'),
        label_assumed_zero_start_gross_pnl=('contribution', 'sum'),
        audit_only_final_fill_rows=('time', 'size')).reset_index()
    balances = ct.groupby(KEYS + ['outcome', 'time']).signed_shares.sum().reset_index().sort_values('time')
    balances['balance'] = balances.groupby(KEYS + ['outcome']).signed_shares.cumsum()
    minimum = balances.groupby(KEYS).balance.min().clip(upper=0).rename('audit_only_min_observed_inventory_zero_start')
    d = d.merge(minimum, on=KEYS, validate='one_to_one')
    d['audit_only_negative_observed_inventory'] = d.audit_only_min_observed_inventory_zero_start.lt(-1e-6).astype(int)
    d['conditional_profit_label'] = d.label_assumed_zero_start_gross_pnl.gt(1e-6).astype('Int64').mask(
        d.audit_only_negative_observed_inventory.eq(1))
    d['confirmed_net_profit_label'] = pd.Series(pd.NA, index=d.index, dtype='Int64')
    entry = ct.merge(d[KEYS + ['prediction_time']], on=KEYS, validate='many_to_one')
    entry = entry[entry.time.eq(entry.prediction_time)].copy()
    entry['buy_shares'] = np.where(entry.side.eq('BUY'), entry['size'], 0.)
    ef = entry.groupby(KEYS).agg(feature_entry_buy_dollars=('buy_dollars', 'sum'),
        feature_entry_buy_shares=('buy_shares', 'sum'), feature_entry_fill_rows=('time', 'size'))
    ef['feature_entry_buy_vwap'] = ef.feature_entry_buy_dollars / ef.feature_entry_buy_shares.replace(0, np.nan)
    ef['feature_entry_log_buy_dollars'] = np.log1p(ef.feature_entry_buy_dollars)
    ob = entry.groupby(KEYS + ['outcome_index']).buy_dollars.sum().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
    share = ob[0] / ef.feature_entry_buy_dollars.replace(0, np.nan)
    ef['feature_entry_direction_concentration'] = abs(2 * share - 1)
    d = d.merge(ef, on=KEYS, validate='one_to_one')
    mt = ct.groupby(['market', 'time']).dollars.sum().rename('dollars').reset_index()
    firsts = ct.groupby(['market', 'wallet']).time.min().reset_index()
    new_wallets = firsts.groupby(['market', 'time']).size().rename('new_wallets').reset_index()
    mt = mt.merge(new_wallets, on=['market', 'time'], how='left').sort_values(['market', 'time'])
    mt['new_wallets'] = mt.new_wallets.fillna(0)
    mt['feature_strict_prior_market_log_dollars'] = np.log1p(mt.groupby('market').dollars.cumsum() - mt.dollars)
    mt['feature_strict_prior_market_log_wallets'] = np.log1p(mt.groupby('market').new_wallets.cumsum() - mt.new_wallets)
    mt['feature_observed_market_age_hours'] = (mt.time - mt.groupby('market').time.transform('min')).dt.total_seconds() / 3600
    d = d.merge(mt[['market', 'time', 'feature_strict_prior_market_log_dollars',
                    'feature_strict_prior_market_log_wallets', 'feature_observed_market_age_hours']],
        left_on=['market', 'prediction_time'], right_on=['market', 'time'], validate='many_to_one').drop(columns='time')
    d['feature_entry_utc_hour'] = d.prediction_time.dt.hour
    assert len(d) == 96184 and d.conditional_profit_label.notna().sum() == 89736
    assert d.audit_only_negative_observed_inventory.sum() == 6448
    print('Candidate accounting reproduced: 96,184 pairs / 89,736 conditional labels.', flush=True)
    # Full-week prefix histories use all original markets, not just candidates.
    print('Building all-market strict-prior supplied-week history...', flush=True)
    build_weekly_entry_history(con, source_view='raw_week', output_table='week_history')
    con.register('entry_keys', d[KEYS + ['prediction_time']])
    hist = con.execute('''SELECT e.wallet,e.market,h.* EXCLUDE(wallet,time)
        FROM entry_keys e JOIN week_history h ON e.wallet=h.wallet AND e.prediction_time=h.time''').df()
    assert len(hist) == len(d)
    for c in ['feature_week_history_first_time', 'feature_week_history_latest_time']:
        hist[c] = stamp(hist[c])
    d = d.merge(hist, on=KEYS, validate='one_to_one')
    con.execute('DROP TABLE week_history')
    d.to_parquet(base_out / 'wallet_market_candidates_full_week.parquet', index=False)
    # Fixed-window function is copied unchanged from the original validation notebook.
    fixed_history_fixture()
    con.register('candidate_wallets', pd.DataFrame({'wallet': d.wallet.unique()}))
    con.execute('''CREATE TABLE history_batches AS
        SELECT wallet,market,epoch_ms(time)::BIGINT t,count(*)::BIGINT n,
          count(*) FILTER(WHERE side='BUY')::BIGINT nb,
          sum(CASE WHEN side='BUY' THEN dollars ELSE 0 END)::DOUBLE b
        FROM raw_week JOIN candidate_wallets USING(wallet) GROUP BY wallet,market,time''')
    entries = {}
    for i, r in d[['wallet', 'prediction_time']].iterrows():
        entries.setdefault(r.wallet, {}).setdefault(int(r.prediction_time.value // 1000000), []).append(i)
    def records():
        # DB-API fetchmany is stable across the old and current DuckDB releases.
        cursor = con.execute('SELECT wallet,market,t,n,nb,b FROM history_batches ORDER BY wallet,t,market')
        while True:
            batch = cursor.fetchmany(65536)
            if not batch:
                break
            yield from batch
    print('Building full 24-hour histories...', flush=True)
    rows, diagnostics = stream_fixed_histories(records(), entries, int(START.value // 1000000))
    w = pd.DataFrame(rows).set_index('pair_id').sort_index()
    assert list(w.index) == list(d.index)
    for a, b in [('proof_week_fill_rows', 'feature_week_history_fill_rows'),
                 ('proof_week_markets', 'feature_week_history_markets'),
                 ('proof_week_buy_dollars', 'feature_week_history_buy_dollars')]:
        assert np.allclose(w[a], d[b], atol=1e-6, rtol=1e-9)
    d['audit_only_fixed24h_left_truncated'] = 1 - w.full24h_observed_span
    for col in w:
        if col.startswith('fixed24h_'):
            d['feature_' + col] = w[col]
    d['feature_week_history_buy_fill_rows'] = w.week_buy_fill_rows
    d['feature_week_history_avg_buy_ticket'] = w.week_avg_buy_ticket
    for period in ['fixed24h', 'week_history']:
        for name in ['fill_rows', 'markets', 'buy_dollars', 'avg_buy_ticket']:
            d['feature_' + period + '_log_' + name] = np.log1p(d['feature_' + period + '_' + name])
    history_cols = [c for c in d if c in KEYS + ['prediction_time', 'audit_only_fixed24h_left_truncated']
        or c.startswith('feature_fixed24h_') or c in ['feature_week_history_buy_fill_rows',
            'feature_week_history_avg_buy_ticket', 'feature_week_history_log_avg_buy_ticket']]
    d[history_cols].to_parquet(history_out / 'fixed24h_history_features.parquet', index=False)
    # Independent explicit target exclusion/time filtering on deterministic cutoffs.
    sample = d.assign(_hash=(d.wallet + '|' + d.market).map(lambda x: hashlib.sha256(x.encode()).hexdigest())).sort_values('_hash').head(128)
    con.register('verify_keys', sample[KEYS + ['prediction_time']])
    checks = con.execute('''SELECT e.wallet,e.market,count(t.wallet) n,
        count(DISTINCT t.market) markets,coalesce(sum(CASE WHEN t.side='BUY' THEN t.dollars ELSE 0 END),0) buys
        FROM verify_keys e LEFT JOIN raw_week t ON t.wallet=e.wallet AND t.market<>e.market
          AND t.time<e.prediction_time AND t.time>=e.prediction_time-INTERVAL '24 HOURS'
        GROUP BY e.wallet,e.market''').df()
    checks = sample.merge(checks, on=KEYS, validate='one_to_one')
    checks = checks[checks.audit_only_fixed24h_left_truncated.eq(0)]
    assert np.allclose(checks.n, checks.feature_fixed24h_fill_rows)
    assert np.allclose(checks.markets, checks.feature_fixed24h_markets)
    assert np.allclose(checks.buys, checks.feature_fixed24h_buy_dollars, atol=1e-6, rtol=1e-8)
    print('Independent fixed-window raw-trade checks passed:', len(checks), flush=True)
    con.close()
    shutil.rmtree(spill, ignore_errors=True)
    # Conservative proxy families, further merged if the course database links
    # markets to the same event_slug. These are database IDs, not on-chain proof.
    event_rows = []
    for r in candidates.itertuples(index=False):
        group, rule = event_proxy(r.market)
        ids = [] if pd.isna(r.event_slug) else ['course_db_event_slug:' + r.event_slug]
        event_rows.append(dict(market=r.market, event_group=group, event_mapping_rule=rule,
                              official_event_ids=json.dumps(ids), event_identity_source='course database event_slug'))
    events = pd.DataFrame(event_rows)
    parent = {g: g for g in events.event_group}
    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    owners = {}
    for r in events.itertuples(index=False):
        for identity in json.loads(r.official_event_ids):
            if identity in owners:
                a, b = find(r.event_group), find(owners[identity])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                owners[identity] = r.event_group
    events['event_group'] = events.event_group.map(find)
    events.to_csv(history_out / 'official_event_connected_mapping.csv', index=False)
    d = d.merge(events, on='market', validate='many_to_one').merge(m[['market', 'resolved_at']], on='market', validate='many_to_one')
    test_groups = set(d.loc[d.prediction_time.ge(T1), 'event_group'])
    val_groups = set(d.loc[d.prediction_time.ge(T0) & d.prediction_time.lt(T1)
                           & ~d.event_group.isin(test_groups), 'event_group'])
    d['assigned_partition'] = 'development_train'
    d.loc[d.event_group.isin(val_groups), 'assigned_partition'] = 'validation'
    d.loc[d.event_group.isin(test_groups), 'assigned_partition'] = 'test'
    d['eligible_conditional'] = d.conditional_profit_label.notna() & d.resolved_at.notna() & d.prediction_time.lt(d.resolved_at)
    d['split'] = 'excluded'
    d.loc[d.eligible_conditional & d.assigned_partition.eq('development_train') & d.prediction_time.lt(T0) & d.resolved_at.lt(T0), 'split'] = 'train'
    d.loc[d.eligible_conditional & d.assigned_partition.eq('validation') & d.prediction_time.ge(T0) & d.prediction_time.lt(T1) & d.resolved_at.lt(T1), 'split'] = 'validation'
    d.loc[d.eligible_conditional & d.assigned_partition.eq('test') & d.prediction_time.ge(T1), 'split'] = 'test'
    d['final_refit'] = d.eligible_conditional & ~d.event_group.isin(test_groups) & d.prediction_time.lt(T1) & d.resolved_at.lt(T1)
    d[KEYS + ['prediction_time', 'resolved_at', 'event_group', 'assigned_partition',
              'eligible_conditional', 'split', 'final_refit']].to_csv(history_out / 'row_split_audit.csv', index=False)
    dump(history_out / 'preparation_checks.json', dict(
        reconstructed_candidate_rows=len(d), known_conditional_labels=int(d.conditional_profit_label.notna().sum()),
        source_raw_rows=6815814, independent_fixed24h_checked_rows=len(checks), window_diagnostics=diagnostics,
        event_group_source='Original conservative proxies unioned by exact course-database event_slug; no independent on-chain verification rerun.',
        elapsed_seconds=time.perf_counter() - began,
        raw_file_hashes={p.name: sha(p) for p in raw_paths}))
    print('All full-model inputs are ready:', root, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--results-only', action='store_true')
    args = parser.parse_args()
    if args.results_only:
        extract_all_results(args.metadata, args.root)
    else:
        build_inputs(args.metadata, args.root)
