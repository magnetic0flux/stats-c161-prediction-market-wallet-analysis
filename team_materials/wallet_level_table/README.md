# Wallet-level Polymarket table (v0.2)

One row per wallet, week 2026-09-29 to 2026-10-05 (UTC). Every number is computed from the supplied raw inputs; nothing is copied from the original study's result tables.

## Where the data comes from

| Input (in `wallet_profit_research/04_results/reproduction_inputs`) | Used for |
|---|---|
| `weekly_trades/trades-*.parquet`: 6,815,814 raw fills (wallet, market, time, side, outcome, price, size, dollars) | All activity columns and the trade-level profit accounting |
| `weekly_market_metadata_complete.csv`: market list with final outcomes | Which outcome won in 13,794 resolved markets, and when they resolved |
| `../03_history_validation/official_event_connected_mapping.csv` (original study output) | Which 2,661 markets are blockchain-confirmed, and their event groups |

Market categories (crypto, sports, weather, politics/other) come from keyword rules on the market name. The profit calculation reproduces the original study's results exactly on all 96,184 of its wallet-market pairs.

| Rows | Count |
|---|---|
| All wallets | 97,613 |
| Traded >= 1 market with a recorded outcome (`in_profit_sample == 1`) | 45,696 |
| With a target (`!is.na(profitable_wallet)`) | 43,472 (24,294 profitable) |
| With a blockchain-confirmed target (`!is.na(profitable_wallet_verified)`) | 16,511 (the original study's wallets) |

## Files
- `polymarket_wallets.csv`: the wallet table. Read with `read.csv(..., colClasses = c(wallet = "character"))`
- `polymarket_wallets_dictionary.csv`: column definitions
- `polymarket_wallets_insider.csv`: the 31 insider-screening columns, 43,472 labelled wallets (`select_insider_columns.R` rebuilds it)
- `polymarket_wallet_market_pairs.csv.gz` + dictionary: one row per wallet x resolved market (351,650 rows), the basis of the profit columns
- `polywallet_0.2.0.tar.gz`: R package. Install with `install.packages("polywallet_0.2.0.tar.gz", repos = NULL, type = "source")`. `polywallet/` is its source
- `example_analysis.R`, `build_scripts/` (Python that rebuilds everything from the raw inputs)

## Caveats
- Profit is conditional gross profit: no fees, and it assumes no shares were held before the week. Wallets that sold shares they hadn't been seen buying are left unlabelled for that market.
- Outcomes outside the 2,661 confirmed markets come from the market file. It agreed with the blockchain on all 2,661 confirmed markets. Use the `_verified` columns as a robustness check.
- 31,788 pairs are in markets that resolved after Oct 5 (`n_pairs_resolved_after_week`). Trades made after Oct 5 are missing for those.
- Activity and profit come from the same week, so models on this table are descriptive, not forecasts.

## Columns


### id

| column | type | units | definition |
|---|---|---|---|
| `wallet` | character | - | Polymarket wallet address (one row per wallet). Unique key. |
| `in_profit_sample` | integer 0/1 | - | 1 if the wallet traded at least one of the 13,794 markets with a recorded final outcome (45,696 wallets). 0 = activity only; profit columns are NA. |

### outcome

| column | type | units | definition |
|---|---|---|---|
| `profitable_wallet` | integer 0/1 | - | MAIN TARGET. 1 if gross_pnl_dollars > $0.000001, else 0. Uses all resolved markets. NA when the wallet has no labelled pairs. |
| `gross_pnl_dollars` | numeric | USD | Sum over labelled pairs of conditional gross profit = sell proceeds - buy spending + settlement value of shares still held (winning share = $1, losing = $0). Assumes no shares held before the week, complete trade records and no fees: gross, not true net profit. A 'pair' is one wallet in one market. |
| `gross_roi` | numeric | ratio | gross_pnl_dollars / labelled_buy_dollars. -1 = lost everything spent; 0.10 = 10% gross return. |
| `pair_win_rate` | numeric | proportion | n_pairs_profitable / n_pairs_labelled: share of the wallet's labelled markets that were profitable. |
| `n_pairs_resolved` | integer | count | Number of the wallet's markets that have a recorded final outcome (labelled + unlabelled). |
| `n_pairs_labelled` | integer | count | Resolved pairs with a usable profit label (excludes negative-inventory pairs). The outcome columns are summed over these. |
| `n_pairs_profitable` | integer | count | Labelled pairs with conditional gross profit > 0. |
| `n_pairs_negative_inventory` | integer | count | Candidate pairs left unlabelled because the wallet sold more shares than it was seen buying (it must have held shares before the week, so profit cannot be computed). |
| `n_pairs_resolved_after_week` | integer | count | Labelled pairs whose market resolved after 2026-10-05 (latest 2026-10-18). Any trades between Oct 5 and resolution are missing, so these profits are less certain. |
| `labelled_buy_dollars` | numeric | USD | Total buy spending in labelled pairs (denominator of gross_roi). |
| `labelled_sell_dollars` | numeric | USD | Total sell proceeds in labelled pairs. |
| `median_pair_pnl_dollars` | numeric | USD | Median conditional gross profit across the wallet's labelled pairs. |
| `max_pair_pnl_dollars` | numeric | USD | Best single-market conditional gross profit. |
| `min_pair_pnl_dollars` | numeric | USD | Worst single-market conditional gross profit. |
| `mean_entry_buy_price_labelled` | numeric | USD per share (0-1) | Mean, across labelled pairs, of the average buy price in the wallet's first trade batch (first timestamp) in that market. Defined on the same markets as the outcome. |

### verified

| column | type | units | definition |
|---|---|---|---|
| `profitable_wallet_verified` | integer 0/1 | - | Same as profitable_wallet but using only the 2,661 markets whose outcome was confirmed on the blockchain (the original study's sample). Robustness check. |
| `gross_pnl_dollars_verified` | numeric | USD | gross_pnl_dollars over blockchain-confirmed markets only. |
| `pair_win_rate_verified` | numeric | proportion | pair_win_rate over blockchain-confirmed markets only. |
| `n_pairs_labelled_verified` | integer | count | Labelled pairs in blockchain-confirmed markets. n_pairs_labelled - this = pairs whose outcome comes only from the market file. |
| `n_events_labelled_verified` | integer | count | Distinct events (groups of related markets, e.g. all temperature brackets for one city-day) among confirmed labelled pairs. Event groups exist only for confirmed markets. |

### activity

| column | type | units | definition |
|---|---|---|---|
| `n_trades` | integer | count | Number of trade rows (fills) in the week, all markets, BUY + SELL. |
| `n_buy_trades` | integer | count | Number of BUY fills. |
| `n_sell_trades` | integer | count | Number of SELL fills. |
| `sell_trade_share` | numeric | proportion | n_sell_trades / n_trades. |
| `n_markets` | integer | count | Distinct markets traded. |
| `n_active_days` | integer | count (1-7) | Distinct UTC calendar days with at least one trade. |
| `n_distinct_timestamps` | integer | count | Distinct trade timestamps (second resolution); fills in the same second count once. |
| `first_trade_time` | character (ISO 8601 UTC) | timestamp | First trade in the week. Parse with as.POSIXct(x, format = '%Y-%m-%dT%H:%M:%SZ', tz = 'UTC'); the R loader does this. |
| `last_trade_time` | character (ISO 8601 UTC) | timestamp | Last trade in the week. |
| `active_span_hours` | numeric | hours | last_trade_time - first_trade_time. |
| `trades_per_active_day` | numeric | count/day | n_trades / n_active_days. |
| `trades_per_market` | numeric | count/market | n_trades / n_markets. |
| `median_gap_minutes` | numeric | minutes | Median time between consecutive distinct trade timestamps. NA if only one timestamp. Small values suggest automated trading. |
| `total_volume_dollars` | numeric | USD | Sum of dollars over all fills (buy + sell). |
| `buy_dollars` | numeric | USD | Total buy spending, all markets. |
| `sell_dollars` | numeric | USD | Total sell proceeds, all markets. |
| `net_cashflow_dollars` | numeric | USD | sell_dollars - buy_dollars. Not profit: it ignores the value of shares still held at settlement. |
| `mean_trade_dollars` | numeric | USD | Mean dollar size per fill. |
| `median_trade_dollars` | numeric | USD | Median dollar size per fill. |
| `max_trade_dollars` | numeric | USD | Largest single fill. |
| `buy_shares` | numeric | shares | Total contract shares bought. |
| `buy_vwap_price` | numeric | USD per share (0-1) | buy_dollars / buy_shares: average price paid per share. NA if no buys. A price is the market-implied probability of that outcome. |
| `buy_dollar_weighted_price` | numeric | USD per share (0-1) | Mean buy price weighted by dollars spent. NA if no buys. |
| `longshot_buy_share` | numeric | proportion | Share of buy dollars spent at price < 0.20 (betting on unlikely outcomes). NA if no buys. |
| `favorite_buy_share` | numeric | proportion | Share of buy dollars spent at price > 0.80 (betting on likely outcomes). NA if no buys. |
| `market_hhi` | numeric | 0-1 | Herfindahl index of the wallet's dollars across markets: sum of squared market shares. 1 = all volume in one market; near 0 = spread thin. |
| `top_market_dollar_share` | numeric | proportion | Share of the wallet's dollars in its single biggest market. |
| `share_dollars_crypto_updown` | numeric | proportion | Share of dollars in short-horizon crypto Up/Down markets (e.g. btc-updown-5m-...). Category assigned from the market slug by keyword rules. |
| `share_dollars_crypto_price` | numeric | proportion | Share of dollars in other crypto markets (price above/below, price targets). |
| `share_dollars_sports_esports` | numeric | proportion | Share of dollars in sports and esports match markets. |
| `share_dollars_weather` | numeric | proportion | Share of dollars in weather (highest/lowest temperature, precipitation) markets. |
| `share_dollars_politics_other` | numeric | proportion | Share of dollars in all remaining markets: mostly politics, geopolitics, economics/Fed, tech and culture. The five share_dollars_* columns sum to 1. |
| `share_trades_weekend` | numeric | proportion | Share of fills on Saturday/Sunday (UTC). |
| `share_trades_us_night` | numeric | proportion | Share of fills between 05:00 and 11:59 UTC (about 10 pm to 5 am US Pacific time). |
| `n_duplicate_rows` | integer | count | Fills identical to another fill on every field. The raw files are not deduplicated (consistent with the original study); these may be genuine repeated fills or duplicates. |
