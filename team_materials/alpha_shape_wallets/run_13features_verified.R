# ==============================================================================
# 13-feature alpha shape on the profitability study's wallets.
#
# Same as the original 13-feature run (alpha_results_pca3d: pca_features(), 13 whole-week trading features
# over all markets) but restricted to the 16,511 wallets of the profitability model, with the profit label
# counted on the 2,661 blockchain-confirmed markets (label_verified_markets()). PCA axes are fitted on these
# 16,511 wallets.
# Run from alpha_shape_wallets/. Writes alpha_results_pca3d_13_verified, its outlier profile and the data for
# interactive_3d/wallet_space.html.
# ==============================================================================
source("wallet_alpha_analysis.R")
source("outlier_profile.R")
source("export_3d_data.R")

RUN13 = CONFIG
RUN13$label_fn     = label_verified_markets()
RUN13$trait_fn     = function(w) traits_pca(w, n_pc = 3, feature_fn = pca_features)     # the 13 features
RUN13$space_filter = function(w) !is.na(w$profitable_wallet)                            # the 16,511 wallets
RUN13$pool_filter  = function(w) !is.na(w$profitable_wallet)
RUN13$out_dir      = "alpha_results_pca3d_13_verified"

run_13_verified = function(cfg = RUN13){
  t0 = Sys.time()
  out = run_analysis(cfg)
  print(out$metrics[, c("group", "n_wallets", "n_excluded_points", "excluded_share", "occupancy", "k")], digits = 3)
  pc = PROFILE_CONFIG; pc$results_dir = cfg$out_dir
  run_profile(pc)
  message("done in ", format(Sys.time() - t0))
  run_export()
}

if (sys.nframe() == 0) run_13_verified()
