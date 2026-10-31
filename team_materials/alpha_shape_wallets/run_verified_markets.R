# ==============================================================================
# Alpha shape on the profitability study's markets: the 2,661 blockchain-confirmed markets.
#
# Same as the main analysis except for the profit label (label_verified_markets()):
#   - wallets: the 16,511 wallets with a confirmed-market label (9,448 profitable), i.e. the
#     wallets of the profitability study (wallet_profit_research);
#   - profitable / ROI / win rate: computed on confirmed markets only.
# Unchanged: the behaviour features (whole-week activity, all markets) and the PCA axes, which are
# still fitted on all 43,472 labelled wallets, so these views share axes with the earlier ones.
#
# Run from alpha_shape_wallets/. Writes two results folders, then the outlier profiles and the
# data for interactive_3d/wallet_space.html (all four views).
# ==============================================================================
source("wallet_alpha_analysis.R")
source("outlier_profile.R")
source("export_3d_data.R")

VERIFIED_RUNS = list(
  # all confirmed-market wallets, communities = market focus x profitable
  verified = list(out_dir = "alpha_results_pca3d_balanced_verified",
                  pool_filter = function(w) !is.na(w$profitable_wallet),
                  community_fn = CONFIG$community_fn),
  # confirmed-market winners only
  verified_profitable = list(out_dir = "alpha_results_pca3d_balanced_verified_profitable",
                  pool_filter = function(w) w$profitable_wallet %in% 1,
                  community_fn = function(w) ifelse(w$share_dollars_politics_other > 0.5,
                                                    "Politics/other", "Sports/crypto/weather"))
)

run_verified = function(runs = VERIFIED_RUNS, label_fn = label_verified_markets()){
  for (nm in names(runs)){
    r = runs[[nm]]; t0 = Sys.time()
    cfg = CONFIG
    cfg$label_fn = label_fn
    cfg$out_dir = r$out_dir; cfg$pool_filter = r$pool_filter; cfg$community_fn = r$community_fn
    out = run_analysis(cfg)
    print(out$metrics[, c("group", "n_wallets", "n_excluded_points", "excluded_share", "occupancy", "k")], digits = 3)
    pc = PROFILE_CONFIG; pc$results_dir = r$out_dir
    run_profile(pc)
    message(sprintf("%s done in %s", nm, format(Sys.time() - t0)))
  }
  run_export()
}

if (sys.nframe() == 0) run_verified()
