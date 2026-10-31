# Full market results and executed wallet models

Open `results_report.html` for the concise results. The four notebooks under `02_notebooks/` contain the executed extraction, preparation, modeling and evaluation stages, with actual supplied-data outputs.

## What the market-results file already contains

Your `market_results.csv` is a full export for the markets in the seven supplied trade files. It is not restricted to the two-market pilot.

| Coverage | Count |
|---|---:|
| Requested markets / unique exported market rows | 78,543 |
| Markets with metadata | 78,518 |
| Markets missing metadata | 25 |
| Recorded binary settlement results | 13,794 |
| Markets without a usable recorded result | 64,749 |

The original snapshot was extracted October 19, 2026, around 03:12–03:13 UTC. It is not a newly queried October 26 snapshot. Missing result records are unknown, not losses and not proof that the underlying real-world market remains unresolved. Results recorded after the trading week are retained in the complete export.

## Where to find all results

| File | Contents |
|---|---|
| `data/weekly_market_metadata_complete.csv` | All 78,543 market rows, metadata and recorded settlements |
| `04_results/market_results/recorded_market_results.csv` | All 13,794 recorded binary settlements |
| `04_results/market_results/markets_without_recorded_results.csv` | The 64,749 missing/unknown-result market records |
| `04_results/market_results/all_market_result_coverage.csv` | One row per market, result coverage and original modeling-candidate membership |
| `04_results/market_results/coverage_summary.json` | Coverage counts and consistency checks |
| `data/candidate_markets_preliminary.csv` | The original 2,661-market candidate universe |

The recorded winner, `resolved_yes`, ordered `outcomes` and $1/$0 `price_at_close` are mutually consistent in this export. This check does not independently repeat the earlier project's on-chain settlement verification. `reference/source_snapshot.json` retains the original source-file hash and extraction timestamps.

## Refresh every required market using the same database method

`02_notebooks/refresh_all_market_results.py` uses the same course PostgreSQL joins as the supplied R workflow. It queries exactly the 78,543 slugs in `data/markets_in_yuzhe_files.csv`, in **79 batches of up to 1,000 markets**. LEFT JOINs preserve rows without metadata or recorded resolutions. No resolution-date filter is applied, so later settlements remain included. No public trades API is used.

Run this locally only if you need a newer snapshot:

```bash
python -m pip install -r 02_notebooks/requirements_database.txt
python 02_notebooks/refresh_all_market_results.py
```

It prompts for the course password without echoing it. No password is embedded or saved. The query is SELECT-only on a read-only connection; parameters are bound separately from SQL. The code checks each batch before finalizing the CSV. This task used your existing full export and made no new database connection.

## The original study's candidate sample was recovered

The candidate rule requires a recorded binary settlement, scheduled start and recorded resolution inside the supplied week, and no observed trade after recorded resolution. The full result export still includes markets that do not pass this modeling filter.

The rule recovers the original totals exactly:

- 2,661 markets and 239,434 trades;
- 96,184 wallet–market pairs;
- 89,736 nonmissing conditional gross-profit labels;
- 6,448 labels left unknown because of negative observed inventory;
- 265 candidate markets with at least 100 wallets, retained as a secondary listing.

The supplied 156-row independent pilot reference was checked against the full-data accounting: all provisional gross P&L amounts and binary labels match. Full-week and 24-hour history use all observed markets, not just the candidate markets. An independent direct raw-trade filter checked fixed-window counts, market breadth and buy dollars on 120 complete-coverage sampled cutoffs; the original history algorithm's endpoint/tie/HHI fixtures also passed.

Conservative event proxy families are preserved and merged by exact event slugs from the course database. The earlier independently matched official event-ID evidence was not provided, so this run explicitly records its event-link source as the course database. The resulting primary model cohort counts match the earlier study.

## Completed models

The random-forest package was run on actual data, with the same five feature variants and a fixed-C logistic reference. Every variant uses the same rows. Full 24-hour observed-file coverage is required.

| Cohort | Wallet–market rows |
|---|---:|
| Development training | 39,659 |
| Outer validation | 6,956 |
| Final refit before October 4 | 52,888 |
| October 4–5 test | 26,507 |

All times are UTC. Original chronological event purging and recorded-resolution maturity checks are retained. RF tuning uses two forward daily folds within development data, with training-event purging and training-only imputation. The eight candidate settings, 200 trees and 0.5 classification threshold were fixed before evaluating the test outcomes. All five variants selected `min_samples_leaf=50`, `max_features='sqrt'`, `max_depth=None` by row-weighted forward-CV log loss.

| Model / feature set | Accuracy | ROC-AUC | Log loss |
|---|---:|---:|---:|
| Logistic, baseline | 70.11% | 0.7754 | 0.5553 |
| RF, baseline | 68.90% | 0.7695 | 0.5728 |
| RF, + five 24-hour history features | 70.17% | 0.7730 | 0.5638 |

The logistic baseline reproduces the earlier report. Adding history improves the tested forest relative to its forest baseline, but its small accuracy advantage over logistic regression comes with lower AUC and higher log loss. This does not establish an overall RF improvement. The full ten-model table is included in the report and CSVs.

## Code and execution order

Install the base requirements, then open the notebooks:

```bash
python -m pip install -r 02_notebooks/requirements.txt
```

1. `00_all_market_results.ipynb`: audit/export all market results; optional fresh database extraction and raw-data rebuild.
2. `01_rf_data_preparation.ipynb`: verify and load the cached original-compatible feature and split exports.
3. `02_rf_models.ipynb`: tune and fit RF models and same-row logistic references.
4. `03_rf_validation.ipynb`: evaluate frozen models and draw ROC/PR/calibration/importance plots.

Executed notebook outputs are included. Their code cells were run sequentially in Python, rather than through a live Jupyter kernel in the creation environment. New notebook runs resolve the extracted package folder dynamically. A completed run is under `04_results/random_forest/full_data_run/`; notebook 01 creates a new timestamped run if you rerun the workflow. Notebook 02 refuses to overwrite model choices within a run that has already evaluated its test outcomes.

The seven raw Parquet files are not duplicated in this ZIP. The cached feature tables, cohorts, fitted models and results are included, so notebooks 01–03 do not need to reprocess the 6.8 million trades. To rebuild from raw, place your seven `trades-*.parquet` files under `data/weekly_trades/` and run:

```bash
python 02_notebooks/prepare_full_model_inputs.py --metadata /path/to/market_results.csv
```

The raw reconstruction deliberately checks the original candidate and label totals. A refreshed result export that changes the original cohort can still be exported with `--results-only`, but requires deliberate cohort revision before refitting this original-study workflow.

## Main model outputs

- `exploratory_test_metrics.csv`: all ten fitted-model comparisons plus naive references.
- `exploratory_test_predictions.csv`: same-row test predictions.
- `validation_grouped_permutation_importance.csv`: feature reliance for baseline and history models, assessed on outer validation.
- `forward_cv_audit.csv`: actual forward-fold counts and key fingerprints.
- `paired_event_log_loss_differences.csv`: descriptive paired loss comparisons.
- `selected_parameters.json`, `model_manifest.json`, `data_manifest.json`: choices, hashes and provenance.
- `figures/`: ROC, precision-recall, calibration and grouped-importance plots.
- `models/`: saved sklearn pipelines, produced with scikit-learn 1.8.0.

The primary importance measure is validation log-loss increase after grouped permutation. All three entry-price terms and their missing flag move together; timing sine/cosine and each ratio/missingness pair are also grouped. Shuffle standard deviations are not confidence intervals or p-values. Impurity importance is exported separately as a secondary training-based measure.

## Interpretation limits

These labels describe **conditional gross profitability**, not verified net profit or insider status. Opening inventory/costs, fees and non-trading position changes remain incomplete. Recorded resolution is a proxy for historical result availability; actual publication timestamps are unverified. Candidate selection is retrospective. The original October 4–5 test period had already been inspected, so these new comparisons are exploratory reuse. Event-only bootstrap intervals do not resample repeated-wallet dependence across events and are descriptive.
