# Alpha shape on profitable wallets only (same PCA axes as the all-wallet analysis).
# Run from alpha_shape_wallets/ (about 15 minutes). Then: outlier_profile.R with
# results_dir = "alpha_results_pca3d_balanced_profitable", and export_3d_data.R.
source("wallet_alpha_analysis.R")
cfg = CONFIG
cfg$out_dir      = "alpha_results_pca3d_balanced_profitable"
cfg$pool_filter  = function(w) w$profitable_wallet %in% 1          # winners only; axes still fitted on all wallets
cfg$community_fn = function(w) ifelse(w$share_dollars_politics_other > 0.5, "Politics/other", "Sports/crypto/weather")
out = run_analysis(cfg)
print(out$metrics, digits = 3)
