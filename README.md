# STATS C161: Trade Profitability Prediction & Trade Outlier Detection

Public data, code, saved analysis results, and presentation for our STATS C161 group project.

**Group members:** Yuzhe He; Zhiru Guoli; Asuka Iwamoto; Lexie Zhang; Zhenlong Zhang.

**Slides:** [Submission PDF](slides/STATS_C161_Midterm_Submission.pdf). The first page includes the group members and this repository's public URL. For the original interactive presentation, download the repository and open [slides/html/index.html](slides/html/index.html) with its accompanying assets.

## Research questions
1. Can information available at the first observed entry into a market predict whether a wallet-market pair will have positive conditional gross profit?
2. Does recent wallet activity add predictive information beyond entry and market characteristics, and does that conclusion change with a random forest?
3. Which wallets have unusual trading profiles in a descriptive PCA and alpha-shape analysis?

The prediction task uses one row per **wallet-market pair**. The outlier analysis uses one row per **wallet** and whole-week behavior; it is a separate descriptive analysis, not an entry-time forecast or evidence of insider activity.

## Data and cohorts
The supplied seven daily files cover September 29 through October 5, 2026 (UTC), with **6,815,814 trade rows**. The screened prediction sample contains **96,184 wallet-market pairs**, **17,177 wallets**, and **2,661 markets**. Conditional profit labels are available for **89,736 pairs**; **6,448 pairs** with negative observed inventory remain unlabelled.

The slide outlier analysis uses **16,511 wallets** with labels from the verified-market cohort and flags **344 outliers**. The broader wallet table also includes a larger cohort based on recorded market outcomes. Those wallet counts and targets must not be substituted for the prediction sample or the verified PCA cohort.

## Repository map
| Location | Contents |
|---|---|
| `slides/` | Submission PDF and complete HTML presentation with local assets |
| `02_notebooks/` | Data preparation, original models, matched history validation, and Python requirements |
| `04_results/reproduction_inputs/weekly_trades/` | All seven supplied daily trading datasets in Parquet format |
| `04_results/reproduction_inputs/` | Market metadata, candidate definitions, and settlement evidence |
| `04_results/01_data_preparation/` | Derived wallet-market dataset and feature dictionaries |
| `04_results/02_original_models/` | Original model outputs |
| `04_results/03_history_validation/` | Matched-cohort validation outputs and original eight-test Holm results |
| `team_materials/wallet_full_analysis/` | Random-forest data package, saved models, predictions, and results |
| `team_materials/wallet_random_forest/` | Newer random-forest/logistic implementation, importance plots, and summary generation |
| `team_materials/wallet_level_table/` | Wallet-level tables, dictionaries, Python builders, and R package source |
| `team_materials/alpha_shape_wallets/` | PCA and alpha-shape analysis code, outlier profiles, and interactive visualizations |
| `team_materials/added_history_tests/` | Additional history tests and the separate twelve-test Holm correction |

The raw data are included directly, rather than only as a link to an inaccessible local folder. Large duplicate exports and archive wrappers are not needed to use these files. Original source archives remain outside this repository.

## Reproduction
The public package preserves completed analysis outputs; the assembled repository has **not** been rerun end-to-end. Saved outputs should be distinguished from a fresh reproduction. Some teammate scripts still require local path configuration.

### Original profitability workflow
Use Python 3.9-3.12 in a virtual environment. From the repository root:
```sh
python -m pip install -r 02_notebooks/requirements.txt
python prepare_paths.py
```
Then execute the three notebooks in `02_notebooks/` in numerical order. The preparation script copies supplied inputs and results into the relative locations expected by those notebooks, replacing the original Unix links. The three original notebooks retain their code but have cell outputs cleared; tabular analysis outputs remain in `04_results/`.

### Random forest and logistic comparison
Read [the implementation README](team_materials/wallet_random_forest/README.md) and [the full-data README](team_materials/wallet_full_analysis/README.md). The original Colab runner expects ZIP uploads. This repository provides their extracted contents: `wallet_full_analysis` is the data/project folder and `wallet_random_forest` supplies the newer code to overlay in its `02_notebooks/` directory. Work in a copy when overlaying. Cached features support the model comparison without rebuilding the raw-trade features.

The supplied workflow distinguishes `REFIT_MODELS=True` (fit and evaluate models) from `False` (regenerate summaries from saved results/models). The existing model files were preserved without loading them during repository assembly.

### Wallet-level PCA and outliers
Read [the wallet-table README](team_materials/wallet_level_table/README.md). The paired folders retain the relative layout used by the R analysis. From `team_materials/alpha_shape_wallets/`, `run_13features_verified.R` is the entry point for the slide cohort. Its saved outputs are in `alpha_results_pca3d_13_verified/`. Other result folders are earlier or alternative analyses, not interchangeable versions of the reported cohort. R dependencies are declared in the corresponding scripts and package source.

The outlier method is **alpha-shape-based detection in PCA space**. MST calculations are used internally for gap detection and sensitivity checks. The reported outlier flag is exclusion from the resulting alpha shape; the implementation records alpha-shape exclusion and MST-dropped flags separately (`point_status()` in `wallet_alpha_analysis.R`).

## Interpreting the saved results
- The matched prediction comparison uses 39,659 initial training rows, 52,888 final refit rows, and 26,507 evaluation rows. Short-history features provide limited incremental prediction in this sample.
- The original **eight-test** Holm family gives the joint week-to-date history test adjusted p = **0.847379**. The later exploratory **twelve-test** family gives that joint test adjusted p = **1.00**, and week-to-date average buy size adjusted p = **0.879285**. Identify the correction family when quoting values; do not combine them into a single correction table.
- Random forest provides an alternative supervised model for the same conditional-profit target. Similar aggregate performance is not proof that predictions are identical or that the two methods are statistically equivalent.
- The PCA and alpha-shape analysis describes unusual wallet behavior. A detected outlier is not a verified insider.

## Scope and limitations
Conditional gross profit is sell proceeds minus buy spending plus settlement value of remaining shares. It assumes zero opening holdings, complete trading records, and no fees. It does not establish verified net returns or trading skill. The test period has been inspected repeatedly, and the comparisons are exploratory. Event concentration and repeated wallets limit independence. A wallet is an address, not necessarily a unique person. Candidate selection is retrospective; recorded resolution times do not necessarily equal historical public announcement times.

## Provenance and integrity
This publication assembles the group's supplied project materials and final slide deck. Metadata, download manifests, dictionaries, and settlement evidence are retained with the inputs. `file_manifest.json` records SHA-256 checksums and sizes for the published files (excluding the manifest itself and Git internals). No new license or additional rights to third-party data are asserted; source and course terms continue to apply.
