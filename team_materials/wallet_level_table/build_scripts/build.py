# Wallet-level Polymarket table, v0.2: profit from every market with a recorded outcome.
# usage: python3 build.py <out_dir> <path/to/wallet_profit_research/04_results/>
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cat import category
from pairs import market_outcomes, pair_accounting

OUT = sys.argv[1]; R = os.path.join(sys.argv[2], '')
WEEK_END = pd.Timestamp('2026-10-05 23:59:59', tz='UTC')

# ---------- inputs ----------
t = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(R + 'reproduction_inputs/weekly_trades/*.parquet'))],
              ignore_index=True)
meta = pd.read_csv(R + 'reproduction_inputs/weekly_market_metadata_complete.csv')
verified = set(pd.read_csv(R + '03_history_validation/official_event_connected_mapping.csv').market)
events = pd.read_csv(R + '03_history_validation/official_event_connected_mapping.csv', usecols=['market', 'event_group'])
mk = pd.Series(t.market.unique()); cat_of = dict(zip(mk, mk.map(category)))

# ---------- 1. wallet activity (all markets, all trades) ----------
def wallet_activity(t):
    t = t.copy()
    t['is_buy'] = t.side == 'BUY'
    t['buy_d'] = np.where(t.is_buy, t.dollars, 0.0); t['sell_d'] = np.where(t.is_buy, 0.0, t.dollars)
    t['buy_q'] = np.where(t.is_buy, t['size'], 0.0)
    t['buy_pxd'] = np.where(t.is_buy, t.price * t.dollars, 0.0)
    t['longshot'] = np.where(t.is_buy & (t.price < 0.20), t.dollars, 0.0)
    t['favorite'] = np.where(t.is_buy & (t.price > 0.80), t.dollars, 0.0)
    t['date'] = t.time.dt.date
    t['cat'] = t.market.map(cat_of)
    t['dup'] = t.duplicated(subset=['wallet', 'market', 'time', 'side', 'outcome', 'price', 'size', 'dollars'])
    t['weekend'] = t.time.dt.weekday >= 5
    t['night'] = t.time.dt.hour.between(5, 11)
    g = t.groupby('wallet', sort=True)
    w = pd.DataFrame({
        'n_trades': g.size(), 'n_buy_trades': g.is_buy.sum(), 'n_markets': g.market.nunique(),
        'n_active_days': g.date.nunique(), 'n_distinct_timestamps': g.time.nunique(),
        'first_trade_time': g.time.min(), 'last_trade_time': g.time.max(),
        'total_volume_dollars': g.dollars.sum(), 'buy_dollars': g.buy_d.sum(), 'sell_dollars': g.sell_d.sum(),
        'mean_trade_dollars': g.dollars.mean(), 'median_trade_dollars': g.dollars.median(),
        'max_trade_dollars': g.dollars.max(), 'buy_shares': g.buy_q.sum(),
        '_bpxd': g.buy_pxd.sum(), '_long': g.longshot.sum(), '_fav': g.favorite.sum(),
        'n_duplicate_rows': g.dup.sum(), 'share_trades_weekend': g.weekend.mean(),
        'share_trades_us_night': g.night.mean()})
    w['n_sell_trades'] = w.n_trades - w.n_buy_trades
    w['sell_trade_share'] = w.n_sell_trades / w.n_trades
    w['net_cashflow_dollars'] = w.sell_dollars - w.buy_dollars
    w['active_span_hours'] = (w.last_trade_time - w.first_trade_time).dt.total_seconds() / 3600
    w['trades_per_active_day'] = w.n_trades / w.n_active_days
    w['trades_per_market'] = w.n_trades / w.n_markets
    bd = w.buy_dollars.replace(0, np.nan)
    w['buy_vwap_price'] = w.buy_dollars / w.buy_shares.replace(0, np.nan)
    w['buy_dollar_weighted_price'] = w._bpxd / bd
    w['longshot_buy_share'] = w._long / bd
    w['favorite_buy_share'] = w._fav / bd
    md = t.groupby(['wallet', 'market']).dollars.sum()
    sh = md / md.groupby(level=0).transform('sum')
    w['market_hhi'] = (sh ** 2).groupby(level=0).sum()
    w['top_market_dollar_share'] = sh.groupby(level=0).max()
    cd = t.pivot_table(index='wallet', columns='cat', values='dollars', aggfunc='sum', fill_value=0)
    cd = cd.div(cd.sum(axis=1), axis=0)
    for c, name in [('crypto_updown', 'crypto_updown'), ('crypto_price', 'crypto_price'),
                    ('sports_esports', 'sports_esports'), ('weather', 'weather'), ('other', 'politics_other')]:
        w[f'share_dollars_{name}'] = cd.get(c, 0.0)
    ts = t[['wallet', 'time']].drop_duplicates().sort_values(['wallet', 'time'])
    ts['gap'] = ts.groupby('wallet').time.diff().dt.total_seconds() / 60
    w['median_gap_minutes'] = ts.groupby('wallet').gap.median()
    return w.drop(columns=[c for c in w.columns if c.startswith('_')])

# ---------- 2. pair accounting on every resolved market ----------
oc = market_outcomes(meta)
p = pair_accounting(t[t.market.isin(set(oc.market))], oc)
p = p.merge(oc[['market', 'resolved_at']], on='market', how='left').merge(events, on='market', how='left')
p['verified_market'] = p.market.isin(verified).astype(int)
p['resolved_after_week'] = (p.resolved_at > WEEK_END).astype(int)
p['market_category'] = p.market.map(cat_of).replace({'other': 'politics_other'})

# ---------- 3. pairs -> wallet outcomes ----------
def wallet_outcomes(p, suffix=''):
    lab = p[p.profit_label.notna()]
    gc, gl = p.groupby('wallet'), lab.groupby('wallet')
    o = pd.DataFrame({
        'n_pairs_resolved': gc.size(),
        'n_pairs_negative_inventory': gc.negative_inventory.sum(),
        'n_pairs_labelled': gl.size(),
        'n_pairs_profitable': gl.profit_label.sum(),
        'labelled_buy_dollars': gl.buy_dollars.sum(),
        'labelled_sell_dollars': gl.sell_dollars.sum(),
        'gross_pnl_dollars': gl.pnl.sum(),
        'median_pair_pnl_dollars': gl.pnl.median(),
        'max_pair_pnl_dollars': gl.pnl.max(),
        'min_pair_pnl_dollars': gl.pnl.min(),
        'mean_entry_buy_price_labelled': gl.entry_buy_vwap.mean(),
        'n_pairs_resolved_after_week': gl.resolved_after_week.sum(),
    })
    o['pair_win_rate'] = o.n_pairs_profitable / o.n_pairs_labelled
    o['gross_roi'] = o.gross_pnl_dollars / o.labelled_buy_dollars.replace(0, np.nan)
    o['profitable_wallet'] = np.where(o.n_pairs_labelled > 0, (o.gross_pnl_dollars > 1e-6).astype(float), np.nan)
    return o.add_suffix(suffix)

o_all = wallet_outcomes(p)
o_ver = wallet_outcomes(p[p.verified_market == 1])[['n_pairs_labelled', 'gross_pnl_dollars', 'pair_win_rate',
                                                     'profitable_wallet']].add_suffix('_verified')
o_ver['n_events_labelled_verified'] = p[(p.verified_market == 1) & p.profit_label.notna()].groupby('wallet').event_group.nunique()

df = wallet_activity(t).join(o_all).join(o_ver)
df['in_profit_sample'] = df.n_pairs_resolved.notna().astype(int)
for c in ['n_pairs_resolved', 'n_pairs_negative_inventory', 'n_pairs_labelled', 'n_pairs_profitable',
          'n_pairs_resolved_after_week', 'n_pairs_labelled_verified', 'n_events_labelled_verified']:
    df[c] = df[c].fillna(0).astype(int)
df = df.reset_index().rename(columns={'index': 'wallet'})

ORDER = ['wallet', 'in_profit_sample',
  'profitable_wallet', 'gross_pnl_dollars', 'gross_roi', 'pair_win_rate',
  'n_pairs_resolved', 'n_pairs_labelled', 'n_pairs_profitable', 'n_pairs_negative_inventory',
  'n_pairs_resolved_after_week', 'labelled_buy_dollars', 'labelled_sell_dollars',
  'median_pair_pnl_dollars', 'max_pair_pnl_dollars', 'min_pair_pnl_dollars', 'mean_entry_buy_price_labelled',
  'profitable_wallet_verified', 'gross_pnl_dollars_verified', 'pair_win_rate_verified',
  'n_pairs_labelled_verified', 'n_events_labelled_verified',
  'n_trades', 'n_buy_trades', 'n_sell_trades', 'sell_trade_share', 'n_markets', 'n_active_days',
  'n_distinct_timestamps', 'first_trade_time', 'last_trade_time', 'active_span_hours',
  'trades_per_active_day', 'trades_per_market', 'median_gap_minutes',
  'total_volume_dollars', 'buy_dollars', 'sell_dollars', 'net_cashflow_dollars',
  'mean_trade_dollars', 'median_trade_dollars', 'max_trade_dollars', 'buy_shares',
  'buy_vwap_price', 'buy_dollar_weighted_price', 'longshot_buy_share', 'favorite_buy_share',
  'market_hhi', 'top_market_dollar_share',
  'share_dollars_crypto_updown', 'share_dollars_crypto_price', 'share_dollars_sports_esports',
  'share_dollars_weather', 'share_dollars_politics_other',
  'share_trades_weekend', 'share_trades_us_night', 'n_duplicate_rows']
assert set(ORDER) == set(df.columns), set(ORDER) ^ set(df.columns)
df = df[ORDER]
for c in ['first_trade_time', 'last_trade_time']:
    df[c] = df[c].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
os.makedirs(OUT, exist_ok=True)
df.to_csv(os.path.join(OUT, 'polymarket_wallets.csv'), index=False, float_format='%.10g', na_rep='NA')

# pair-level file (for timing / per-market work later)
pp = p[['wallet', 'market', 'market_category', 'verified_market', 'event_group', 'first_time', 'resolved_at',
        'resolved_after_week', 'fills', 'buy_dollars', 'sell_dollars', 'buy_shares', 'q0', 'q1', 'payout0',
        'entry_buy_vwap', 'negative_inventory', 'pnl', 'profit_label']].rename(columns={
        'first_time': 'first_trade_time', 'q0': 'net_shares_outcome0', 'q1': 'net_shares_outcome1',
        'payout0': 'payout_outcome0', 'pnl': 'gross_pnl_dollars'})
for c in ['first_trade_time', 'resolved_at']:
    pp[c] = pp[c].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
pp.to_csv(os.path.join(OUT, 'polymarket_wallet_market_pairs.csv.gz'), index=False, float_format='%.10g',
          na_rep='NA', compression='gzip')

# ---------- checks ----------
print('wallets', len(df), 'unique', df.wallet.is_unique, 'trades', df.n_trades.sum())
print('profit sample', df.in_profit_sample.sum(), 'labelled', df.profitable_wallet.notna().sum(),
      'verified labelled', df.profitable_wallet_verified.notna().sum())
print('pairs', len(p), 'labelled', p.profit_label.notna().sum(), 'verified pairs', p.verified_market.sum(),
      'markets', p.market.nunique(), 'resolved after week pairs', p.resolved_after_week.sum())
print(df.profitable_wallet.value_counts().to_dict(), df.profitable_wallet_verified.value_counts().to_dict())
print(p.groupby('market_category').agg(markets=('market', 'nunique'), pairs=('wallet', 'size')).to_string())
