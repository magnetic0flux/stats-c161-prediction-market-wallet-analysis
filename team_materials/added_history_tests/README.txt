Added significance tests for the two 5-feature history extras (prior buy concentration = buy_market_hhi,
prior average buy size = log_avg_buy_ticket), week-to-date ("expanding") and 24 h ("fixed24h").

Same method as 03_final_validation.ipynb (cells 10, 12, 14 run unchanged): market-demeaned linear probability
model on the 39,659 full-24h training rows, wallet x event two-way clustered SEs, one feature added to the
association baseline at a time; its missing flag (no prior buys: 720 week-to-date rows, 1,588 24 h rows) is
included as a control and only the feature's coefficient is tested.
The 24 h features are read from 03_history_validation/fixed24h_history_features.parquet (saved output of cell 6).
Check: the original 8 frozen tests reproduce to |p difference| < 1e-11.

These 4 tests are exploratory additions, not part of the pre-specified 8-test family.
all_12_tests_holm.csv: Holm over all 12 history tests (8 original + 4 added).
added_tests_hhi_avg_buy.csv: the 4 added tests, Holm within those 4.

usage: python3 added_history_tests.py <03_final_validation.ipynb> <fixed24h_history_features.parquet> <root with data/> <out dir> <nested_frozen_concept_tests.csv>
