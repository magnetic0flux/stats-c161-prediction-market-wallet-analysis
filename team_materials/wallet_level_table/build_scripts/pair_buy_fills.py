# Buy-fill counts per wallet x resolved market: the one column the pair table lacks for the
# profitability study's "average buy size" (buy dollars / buy fills).
# usage: python3 pair_buy_fills.py <wallet_level_table dir> <path/to/wallet_profit_research/04_results/>
# Writes polymarket_wallet_market_pair_buy_fills.csv.gz (wallet, market, buy_fills) with the same rows as
# polymarket_wallet_market_pairs.csv.gz, and checks fills and buy dollars against that table.
import os, sys, glob, numpy as np, pandas as pd

OUT = sys.argv[1]; R = os.path.join(sys.argv[2], '')
pairs = pd.read_csv(os.path.join(OUT, 'polymarket_wallet_market_pairs.csv.gz'),
                    usecols=['wallet', 'market', 'fills', 'buy_dollars'])
keys = set(pairs.market)
t = pd.concat([pd.read_parquet(f, columns=['wallet', 'market', 'side', 'dollars'])
               for f in sorted(glob.glob(R + 'reproduction_inputs/weekly_trades/*.parquet'))], ignore_index=True)
t = t[t.market.isin(keys)]
t['is_buy'] = t.side == 'BUY'
t['buy_d'] = np.where(t.is_buy, t.dollars, 0.0)
g = t.groupby(['wallet', 'market'])
b = pd.DataFrame({'buy_fills': g.is_buy.sum(), '_fills': g.size(), '_buy_d': g.buy_d.sum()}).reset_index()

m = pairs.merge(b, on=['wallet', 'market'], how='left', validate='one_to_one')
assert m.buy_fills.notna().all() and len(m) == len(pairs)
assert (m._fills == m.fills).all()                                  # same trades as the pair table
assert np.allclose(m._buy_d, m.buy_dollars, rtol=1e-8, atol=1e-6)    # the CSV keeps ~10 significant digits
out = m[['wallet', 'market', 'buy_fills']].astype({'buy_fills': int})
out.to_csv(os.path.join(OUT, 'polymarket_wallet_market_pair_buy_fills.csv.gz'), index=False)
print(f'{len(out):,} pairs; {int(out.buy_fills.sum()):,} buy fills; fills and buy dollars match the pair table')
