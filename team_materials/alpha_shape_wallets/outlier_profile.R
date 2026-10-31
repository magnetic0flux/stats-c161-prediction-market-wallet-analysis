# ==============================================================================
# Who are the alpha-shape outliers? Evidence that they are big bettors on unlikely outcomes.
#
# Input : <results_dir>/wallet_alpha_flags.csv and <results_dir>/mst_drop_by_k.rds (from wallet_alpha_analysis.R)
#         ../wallet_level_table/polymarket_wallets.csv
# Output: <results_dir>/outlier_profile/  tables + figures
#
# Four pieces of evidence, each a swappable function:
#   1. compare_features()  - outliers vs the rest on each behaviour feature: medians, effect size
#                            (Cliff's delta) and a Wilcoxon rank-sum test
#   2. enrichment_grid()   - two-way grid (default bet size x long-shot share): how many times more
#                            likely a wallet in each cell is to be an outlier than the base rate (lift)
#   3. outlier_model()     - logistic regression of outlier status on standardised features (swappable
#                            via PROFILE_CONFIG$model_terms):
#                            which features predict being left out, holding the others fixed
#                            (near-duplicate features left out so the odds ratios can be read one by one)
#   4. k_robustness()      - the same profile at other Tukey multipliers k (MST rule only)
# ==============================================================================

suppressMessages({ library(ggplot2); library(jsonlite) })

PROFILE_CONFIG = list(
  results_dir = "alpha_results_pca3d_balanced",                  # or "alpha_results_pca3d" (13 features)
  wallet_file = "../wallet_level_table/polymarket_wallets.csv",

  # features compared: name = function(wallet table) -> numeric
  features = list(
    `Total volume ($, log10)`        = function(d) log10(d$total_volume_dollars),
    `Average bet size ($, log10)`    = function(d) log10(d$mean_trade_dollars),
    `Largest bet ($, log10)`         = function(d) log10(d$max_trade_dollars),
    `Average buy price`              = function(d) d$buy_vwap_price,
    `Share of money on long shots`   = function(d) d$longshot_buy_share,
    `Share of money on favourites`   = function(d) d$favorite_buy_share,
    `Trades (log10)`                 = function(d) log10(d$n_trades),
    `Markets (log10)`                = function(d) log10(d$n_markets),
    `Market concentration (HHI)`     = function(d) d$market_hhi),
  # features drawn in feature_distributions.png (names from `features`)
  key_features = c("Average bet size ($, log10)", "Average buy price", "Share of money on long shots", "Total volume ($, log10)"),
  # predictors of outlier status: function(profile data) -> data.frame (standardised inside outlier_model)
  model_terms = function(d) data.frame(
    log_mean_bet       = log10(d$mean_trade_dollars),
    log_markets        = log10(d$n_markets),         # trades (rho 0.97) and HHI (rho -0.96) left out: near-copies
    longshot_share     = d$longshot_buy_share,
    favourite_share    = d$favorite_buy_share,
    sell_share         = d$sell_trade_share),
  # enrichment grid: rows and columns, each a variable cut into groups
  grid = list(
    row_title = "Average bet size", row_fn = function(d) d$mean_trade_dollars,
    row_breaks = c(0, 10, 100, 1000, Inf), row_labels = c("< $10", "$10-100", "$100-1k", "> $1k"), row_right = FALSE,
    col_title = "Share of buy money spent on long shots (price < 0.20)", col_fn = function(d) d$longshot_buy_share,
    col_breaks = c(-Inf, 0, 0.25, 0.5, 1), col_labels = c("none", "0-25%", "25-50%", "> 50%"), col_right = TRUE),
  outlier_col = "pool_excluded"
)

# ------------------------------------------------------------------------------
load_profile_data = function(cfg){
  f = read.csv(file.path(cfg$results_dir, "wallet_alpha_flags.csv"), colClasses = c(wallet = "character"))
  w = read.csv(cfg$wallet_file, colClasses = c(wallet = "character"))
  keep = setdiff(names(w), names(f))
  d = merge(f, w[, c("wallet", keep)], by = "wallet")
  d$outlier = d[[cfg$outlier_col]]
  d
}

# Cliff's delta from the rank-sum statistic: P(outlier > rest) - P(outlier < rest), in [-1, 1].
# |delta| < 0.15 negligible, < 0.33 small, < 0.47 medium, otherwise large (Romano et al. 2006).
cliffs_delta = function(x, y){
  wt = suppressWarnings(stats::wilcox.test(x, y, exact = FALSE))
  list(delta = 2 * unname(wt$statistic) / (length(x) * length(y)) - 1, p = wt$p.value)
}
delta_label = function(d) cut(abs(d), c(-Inf, 0.15, 0.33, 0.47, Inf), labels = c("negligible", "small", "medium", "large"))

# 1 --------------------------------------------------------------------------
compare_features = function(d, features){
  do.call(rbind, lapply(names(features), function(nm){
    v = features[[nm]](d); x = v[d$outlier]; y = v[!d$outlier]
    cd = cliffs_delta(x, y)
    data.frame(feature = nm,
               median_outliers = stats::median(x, na.rm = TRUE), median_rest = stats::median(y, na.rm = TRUE),
               iqr_outliers = paste(signif(stats::quantile(x, c(.25, .75), na.rm = TRUE), 3), collapse = " - "),
               iqr_rest = paste(signif(stats::quantile(y, c(.25, .75), na.rm = TRUE), 3), collapse = " - "),
               cliffs_delta = cd$delta, effect = delta_label(cd$delta), wilcoxon_p = cd$p)
  }))
}

plot_feature_distributions = function(d, features, which){
  long = do.call(rbind, lapply(which, function(nm) data.frame(
    feature = nm, value = features[[nm]](d),
    group = ifelse(d$outlier, "Left out of alpha shape", "In alpha shape"))))
  long$feature = factor(long$feature, levels = which)
  ggplot(long, aes(group, value)) +
    geom_violin(fill = "grey85", colour = NA, scale = "width") +
    geom_boxplot(width = 0.15, outlier.shape = NA, fill = "white", linewidth = 0.3) +
    facet_wrap(~feature, scales = "free_y", nrow = 1) +
    labs(x = NULL, y = NULL) + theme_classic() +
    theme(strip.background = element_blank(), strip.text = element_text(face = "bold"))
}

# 2 --------------------------------------------------------------------------
# rows x columns from spec (PROFILE_CONFIG$grid). lift = outlier rate in cell / overall rate
enrichment_grid = function(d, spec = PROFILE_CONFIG$grid){
  d$row = cut(spec$row_fn(d), spec$row_breaks, spec$row_labels, right = spec$row_right)
  d$col = cut(spec$col_fn(d), spec$col_breaks, spec$col_labels, right = spec$col_right)
  base = mean(d$outlier)
  g = stats::aggregate(outlier ~ row + col, d, function(x) c(n = length(x), rate = mean(x)))
  g = data.frame(g[, 1:2], wallets = g$outlier[, "n"], outlier_rate = g$outlier[, "rate"])
  g$outliers = round(g$wallets * g$outlier_rate)
  g$lift = g$outlier_rate / base
  attr(g, "base_rate") = base
  g
}

plot_enrichment = function(g, spec = PROFILE_CONFIG$grid){
  ggplot(g, aes(col, row, fill = lift)) +
    geom_tile(colour = "white") +
    geom_text(aes(label = sprintf("%.1fx\n%d / %d", lift, outliers, wallets)), size = 3) +
    scale_fill_gradient(low = "white", high = "grey35", name = "Lift") +
    labs(x = spec$col_title, y = spec$row_title,
         subtitle = sprintf("Lift = outlier rate in cell / overall rate (%.1f%%)", 100 * attr(g, "base_rate"))) +
    theme_classic() + theme(axis.line = element_blank(), axis.ticks = element_blank())
}

# 3 --------------------------------------------------------------------------
outlier_model = function(d, terms = PROFILE_CONFIG$model_terms){
  X = data.frame(outlier = d$outlier, terms(d))
  X = X[stats::complete.cases(X), ]
  X[-1] = scale(X[-1])                                   # odds ratio per 1 SD
  fit = stats::glm(outlier ~ ., data = X, family = stats::binomial())
  s = summary(fit)$coefficients[-1, , drop = FALSE]
  ci = suppressMessages(stats::confint.default(fit))[-1, , drop = FALSE]
  data.frame(term = rownames(s), odds_ratio_per_sd = exp(s[, 1]),
             ci_low = exp(ci[, 1]), ci_high = exp(ci[, 2]), p = s[, 4], row.names = NULL)
}

plot_model = function(m){
  m$term = factor(m$term, levels = m$term[order(m$odds_ratio_per_sd)])
  ggplot(m, aes(odds_ratio_per_sd, term)) +
    geom_vline(xintercept = 1, linetype = "dashed", linewidth = 0.3) +
    geom_errorbarh(aes(xmin = ci_low, xmax = ci_high), height = 0.2, linewidth = 0.4) +
    geom_point(size = 2) + scale_x_log10() +
    labs(x = "Odds ratio of being an outlier, per 1 SD (log scale, 95% CI)", y = NULL) + theme_classic()
}

# 4 --------------------------------------------------------------------------
# drop: logical matrix (rows = distinct points, rownames = representative wallet; cols = k values)
k_robustness = function(d, drop){
  pts_wallet = rownames(drop)
  do.call(rbind, lapply(colnames(drop), function(k){
    o = d$wallet %in% pts_wallet[drop[, k]]
    data.frame(k = as.numeric(sub("k_", "", k)), points_dropped = sum(drop[, k]),
               median_volume_out = stats::median(d$total_volume_dollars[o]),
               median_volume_rest = stats::median(d$total_volume_dollars[!o]),
               median_mean_bet_out = stats::median(d$mean_trade_dollars[o]),
               median_mean_bet_rest = stats::median(d$mean_trade_dollars[!o]),
               share_mostly_longshot_out = mean(d$longshot_buy_share[o] > 0.5),
               share_mostly_longshot_rest = mean(d$longshot_buy_share[!o] > 0.5))
  }))
}

# ------------------------------------------------------------------------------
run_profile = function(cfg = PROFILE_CONFIG){
  cfg$out_dir = file.path(cfg$results_dir, "outlier_profile"); cfg$k_file = file.path(cfg$results_dir, "mst_drop_by_k.rds")
  dir.create(cfg$out_dir, showWarnings = FALSE, recursive = TRUE)
  d = load_profile_data(cfg)

  cmp = compare_features(d, cfg$features)
  grid = enrichment_grid(d, cfg$grid)
  model = outlier_model(d, cfg$model_terms)
  rob = if (file.exists(cfg$k_file)) k_robustness(d, readRDS(cfg$k_file)) else NULL

  write.csv(cmp, file.path(cfg$out_dir, "feature_comparison.csv"), row.names = FALSE)
  write.csv(grid, file.path(cfg$out_dir, "enrichment_grid.csv"), row.names = FALSE)
  writeLines(toJSON(cfg$grid[c("row_title", "row_labels", "col_title", "col_labels")], auto_unbox = TRUE),
             file.path(cfg$out_dir, "enrichment_grid_spec.json"))        # axis names for the 3D page
  write.csv(model, file.path(cfg$out_dir, "outlier_logistic_model.csv"), row.names = FALSE)
  if (!is.null(rob)) write.csv(rob, file.path(cfg$out_dir, "k_robustness.csv"), row.names = FALSE)

  ggsave(file.path(cfg$out_dir, "feature_distributions.png"), plot_feature_distributions(d, cfg$features, cfg$key_features),
         width = 11, height = 3.6, dpi = 200, bg = "white")
  ggsave(file.path(cfg$out_dir, "enrichment_grid.png"), plot_enrichment(grid, cfg$grid), width = 7, height = 4.5, dpi = 200, bg = "white")
  ggsave(file.path(cfg$out_dir, "outlier_model.png"), plot_model(model), width = 7, height = 3.5, dpi = 200, bg = "white")
  invisible(list(data = d, comparison = cmp, grid = grid, model = model, robustness = rob))
}

if (sys.nframe() == 0) {
  out = run_profile()
  print(out$comparison, digits = 3); print(out$grid, digits = 3); print(out$model, digits = 3)
  if (!is.null(out$robustness)) print(out$robustness, digits = 3)
}
