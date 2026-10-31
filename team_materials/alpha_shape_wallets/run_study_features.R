# ==============================================================================
# Alpha shape in the profitability model's own feature space.
#
# Behaviour space = PCA of the study's 5 wallet-history features (03_final_validation.ipynb, the
# 5-feature history variant): trade count, market count, buy dollars, buy-dollar HHI across markets and
# average buy size, each per wallet over the week on the 2,661 blockchain-confirmed markets, built from
# the wallet x market pair table (+ build_scripts/pair_buy_fills.py for buy-fill counts).
# Wallets and profit label: the study's 16,511 confirmed-market wallets (label_verified_markets()).
# PCA axes are fitted on those 16,511 wallets; the profitable-only run keeps the same axes.
#
# Difference from the study that remains by design: the study measures these features before each bet
# (week-to-date or prior 24 h) for one wallet-market pair; here they are the wallet's whole-week totals.
#
# Run from alpha_shape_wallets/ after pair_buy_fills.py. Writes two results folders, outlier profiles and
# the data for interactive_3d/wallet_space.html (all views).
# ==============================================================================
source("wallet_alpha_analysis.R")
source("outlier_profile.R")
source("export_3d_data.R")

STUDY_CONFIG = CONFIG
STUDY_CONFIG$label_fn     = add_study_features(label_verified_markets())
STUDY_CONFIG$trait_fn     = function(w) traits_pca(w, n_pc = 3, feature_fn = pca_features_study)
STUDY_CONFIG$space_filter = function(w) !is.na(w$profitable_wallet)        # the 16,511 confirmed-market wallets

STUDY_RUNS = list(
  study = list(out_dir = "alpha_results_pca3d_study",
               pool_filter = function(w) !is.na(w$profitable_wallet),
               community_fn = CONFIG$community_fn),
  study_profitable = list(out_dir = "alpha_results_pca3d_study_profitable",
               pool_filter = function(w) w$profitable_wallet %in% 1,
               community_fn = function(w) ifelse(w$share_dollars_politics_other > 0.5,
                                                 "Politics/other", "Sports/crypto/weather"))
)

# Outlier profile in the same terms: the study features first, then two whole-week style descriptors
STUDY_PROFILE = PROFILE_CONFIG
STUDY_PROFILE$features = list(
  `Trades, confirmed markets (log10)`           = function(d) log10(d$sh_fills),
  `Markets, confirmed (log10)`                  = function(d) log10(d$sh_markets),
  `Buy amount, confirmed ($, log10)`            = function(d) log10(d$sh_buy_dollars),
  `Buy concentration (HHI)`                     = function(d) d$sh_buy_hhi,
  `Average buy size, confirmed ($, log10)`      = function(d) log10(d$sh_avg_buy),
  `Average buy price (whole week)`              = function(d) d$buy_vwap_price,
  `Share of money on long shots (whole week)`   = function(d) d$longshot_buy_share)
STUDY_PROFILE$key_features = c("Average buy size, confirmed ($, log10)", "Markets, confirmed (log10)",
                               "Buy concentration (HHI)", "Share of money on long shots (whole week)")
STUDY_PROFILE$model_terms = function(d) data.frame(
  log_avg_buy    = log10(d$sh_avg_buy),
  log_markets_c  = log10(d$sh_markets),      # trades (rho 0.89), HHI (-0.96) and buy dollars (0.80 with bet size) left out
  longshot_share = d$longshot_buy_share,     # whole-week style descriptors
  sell_share     = d$sell_trade_share)
STUDY_PROFILE$grid$row_title = "Average buy size (confirmed markets)"
STUDY_PROFILE$grid$row_fn    = function(d) d$sh_avg_buy

run_study = function(runs = STUDY_RUNS, base = STUDY_CONFIG, profile = STUDY_PROFILE){
  for (nm in names(runs)){
    r = runs[[nm]]; t0 = Sys.time()
    cfg = base; cfg$out_dir = r$out_dir; cfg$pool_filter = r$pool_filter; cfg$community_fn = r$community_fn
    out = run_analysis(cfg)
    print(out$metrics[, c("group", "n_wallets", "n_excluded_points", "excluded_share", "occupancy", "k")], digits = 3)
    pc = profile; pc$results_dir = r$out_dir
    run_profile(pc)
    message(sprintf("%s done in %s", nm, format(Sys.time() - t0)))
  }
  run_export()
}

if (sys.nframe() == 0) run_study()
