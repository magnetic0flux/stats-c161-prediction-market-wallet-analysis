# Wallet RF and logistic implementation

This code package is supplied as `wallet_RFimplementation.zip`. The Colab entry point is `RF_logistic_results.ipynb` in the complete delivery. Upload it to Colab, then upload `wallet_full_analysis.zip` and `wallet_RFimplementation.zip` when prompted. The runner installs this code into the full project automatically.

The included `RF_logistic_resultssummary.html` is the latest edited real-data summary. Rerunning the workflow regenerates a new summary with that exact filename in the selected run's figures/ folder. ROC and calibration each show all four baseline/history model curves; no precision–recall panel or smaller-feature detail is displayed. RF tuning details are included.

Use `REFIT_MODELS=True` to prepare a fresh run, tune/fit all models and evaluate. Use False to regenerate the summary from supplied saved models and existing results. That route reads the existing test predictions/labels for plotting, without fitting models or selecting settings.

Keep `rf_common.py`, `feature_importance.py`, `comparison_summary.py`, the HTML template and the four component notebooks together under 02_notebooks/. The data package contains the cached features needed for reproducing this analysis; raw daily trades are not required unless deliberately rebuilding features from scratch. The notebook's final cell downloads results and executed notebooks.

The test period has already been inspected; comparisons remain exploratory. The target is conditional gross profitability, not verified net profit. Continuous logistic coefficients are standardized, missing indicators remain unscaled, and RF importance uses Gini impurity. Grouped validation permutation remains a supporting check.

