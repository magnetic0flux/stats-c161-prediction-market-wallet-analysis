"""Pair-level (wallet x market) conditional gross accounting, same rules as the original study.
pnl = sell$ - buy$ + q0*p0 + q1*(1-p0); q = net shares bought during the week (zero opening holdings);
pair is unlabelled if the running inventory of either outcome ever goes below zero."""
import json, numpy as np, pandas as pd

def market_outcomes(meta):
    r = meta[meta.resolution_found == True].copy()
    o = r.outcomes.map(json.loads)
    r['outcome0'] = o.str[0]; r['outcome1'] = o.str[1]
    r['payout0'] = r.resolved_yes.astype(bool).astype(float)
    r['resolved_at'] = pd.to_datetime(r.resolved_at, utc=True, format='ISO8601')
    return r[['market', 'outcome0', 'outcome1', 'payout0', 'resolved_at']]

def pair_accounting(trades, outcomes, tie_order='buys_first', eps_inv=1e-9, eps_profit=1e-6):
    t = trades.merge(outcomes[['market', 'outcome0', 'outcome1', 'payout0']], on='market', how='inner')
    norm = lambda s: s.str.replace(r"['’`]", '', regex=True).str.strip()  # trades drop apostrophes
    o, o0, o1 = norm(t.outcome), norm(t.outcome0), norm(t.outcome1)
    t['k'] = np.where(o == o0, 0, np.where(o == o1, 1, -1))
    assert (t.k >= 0).all(), t.loc[t.k < 0, ['market', 'outcome']].drop_duplicates().head()
    buy = t.side == 'BUY'
    t['dq'] = np.where(buy, t['size'], -t['size'])
    t['buy_d'] = np.where(buy, t.dollars, 0.0); t['sell_d'] = np.where(buy, 0.0, t.dollars)
    t['buy_q'] = np.where(buy, t['size'], 0.0)
    t['ord'] = np.arange(len(t))
    if tie_order == 'buys_first':
        t['ord2'] = (~buy).astype(int)
        t = t.sort_values(['wallet', 'market', 'k', 'time', 'ord2', 'ord'], kind='mergesort')
    else:
        t = t.sort_values(['wallet', 'market', 'k', 'time', 'ord'], kind='mergesort')
    t['inv'] = t.groupby(['wallet', 'market', 'k']).dq.cumsum()
    g = t.groupby(['wallet', 'market'])
    p = pd.DataFrame({
        'fills': g.size(), 'buy_dollars': g.buy_d.sum(), 'sell_dollars': g.sell_d.sum(),
        'buy_shares': g.buy_q.sum(), 'min_inventory': g.inv.min(),
        'first_time': g.time.min(), 'payout0': g.payout0.first()})
    q = t.groupby(['wallet', 'market', 'k']).dq.sum().unstack(fill_value=0.0)
    p['q0'] = q.get(0, 0.0); p['q1'] = q.get(1, 0.0)
    p['pnl'] = p.sell_dollars - p.buy_dollars + p.q0 * p.payout0 + p.q1 * (1 - p.payout0)
    p['negative_inventory'] = (p.min_inventory < -eps_inv).astype(int)
    p['profit_label'] = np.where(p.negative_inventory == 1, np.nan, (p.pnl > eps_profit).astype(float))
    # entry batch (first timestamp) buy VWAP
    fb = t[t.time == t.groupby(['wallet', 'market']).time.transform('min')]
    fg = fb.groupby(['wallet', 'market'])
    p['entry_buy_vwap'] = fg.buy_d.sum() / fg.buy_q.sum().replace(0, np.nan)
    return p.reset_index()
