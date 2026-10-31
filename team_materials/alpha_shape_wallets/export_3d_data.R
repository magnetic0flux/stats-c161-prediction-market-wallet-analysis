# ==============================================================================
# Export the data behind interactive_3d/wallet_space.html
# Run from alpha_shape_wallets/ after wallet_alpha_analysis.R and outlier_profile.R.
#
# Writes interactive_3d/wallet_data.js with one entry per view (all wallets; profitable wallets only;
# the same two on the 2,661 blockchain-confirmed markets, from run_verified_markets.R; and the same two in
# the profitability model's feature space, from run_study_features.R): wallet coordinates and attributes,
# the alpha-shape surface, the evidence tables and how the outliers compare with another analysis.
# Each view carries its own PCA loadings, variance and axis descriptions (views differ in features).
# ==============================================================================
suppressMessages(library(jsonlite))

# axis descriptions for the 13-feature confirmed-market view, read from alpha_results_pca3d_13_verified/pca_loadings.csv
AXIS13_TEXT = list(
  PC1 = c("focused: money in one market", "active: many trades, markets and days"),
  PC2 = c("small bets, more long shots", "big bets: large average and largest trade"),
  PC3 = c("buys long shots, sells more", "buys favourites"))

# axis descriptions for the study-feature views, read from alpha_results_pca3d_study/pca_loadings.csv
STUDY_AXIS_TEXT = list(
  PC1 = c("active: many trades and markets, more money", "all buying in one market (high HHI)"),
  PC2 = c("big buys: large average buy size", "small buys spread over more markets"),
  PC3 = c("few trades for its concentration", "many trades packed into one market"))

EXPORT_CONFIG = list(
  # views shown on the page (switch at the top). Views without feature_label / axis_text use the defaults
  # below (the balanced 5-feature axes); views not run yet are left off.
  views = list(
    all = list(title = "All wallets", results_dir = "alpha_results_pca3d_balanced",
               compare_dir = "alpha_results_pca3d", compare_label = "13-feature version",
               note = "Shape built from every wallet with a profit result (all 13,794 resolved markets). Outliers are judged against all wallets.",
               vs_title = "Same outliers with other features?",
               vs_text = "Outliers found with these features compared with the 13-feature version. If the outlier list depended on the feature choice, the overlap would be small."),
    profitable = list(title = "Profitable wallets only", results_dir = "alpha_results_pca3d_balanced_profitable",
               compare_dir = "alpha_results_pca3d_balanced", compare_label = "all-wallet analysis", all_profitable = TRUE,
               note = "Shape rebuilt from profitable wallets only (unprofitable wallets removed). Outliers are judged against other winners: an unusual way of winning.",
               vs_title = "Same outliers as the all-wallet analysis?",
               vs_text = "Profitable wallets flagged here compared with the profitable wallets flagged when all wallets were analysed together."),
    verified = list(title = "Confirmed markets", results_dir = "alpha_results_pca3d_balanced_verified",
               compare_dir = "alpha_results_pca3d_balanced", compare_label = "all-market analysis",
               note = "Only the 2,661 blockchain-confirmed markets of the profitability study: its 16,511 wallets, with profit, ROI and win rate counted on those markets alone. Features and axes are unchanged.",
               vs_title = "Same outliers as with all markets?",
               vs_text = "Wallets in both analyses: outliers here compared with the outliers of the all-market analysis. A large overlap means the outlier list does not depend on which markets set the label."),
    verified_profitable = list(title = "Confirmed markets, profitable", results_dir = "alpha_results_pca3d_balanced_verified_profitable",
               compare_dir = "alpha_results_pca3d_balanced_profitable", compare_label = "all-market profitable analysis", all_profitable = TRUE,
               note = "Wallets profitable on the 2,661 blockchain-confirmed markets only. Outliers are judged against other confirmed-market winners.",
               vs_title = "Same outliers as the all-market winners?",
               vs_text = "Wallets in both analyses: outliers here compared with the outliers among wallets profitable across all markets."),
    verified13 = list(title = "13 features, confirmed", results_dir = "alpha_results_pca3d_13_verified",
               compare_dir = "alpha_results_pca3d", compare_label = "13-feature analysis on all wallets",
               feature_label = "13 trading features",
               note = "The 13-feature version on the profitability model's 16,511 wallets: profit counted on the 2,661 confirmed markets, PCA fitted on these wallets. Features cover the whole week across all markets.",
               vs_title = "Same outliers as the 13-feature run on all wallets?",
               vs_text = "Wallets in both analyses: outliers here compared with the outliers of the original 13-feature run on all 43,472 wallets.",
               axis_text = AXIS13_TEXT),
    study = list(title = "Study features", results_dir = "alpha_results_pca3d_study",
               compare_dir = "alpha_results_pca3d_balanced_verified", compare_label = "balanced-feature analysis",
               feature_label = "the profitability model's 5 wallet-history features (confirmed markets)",
               note = "Aligned with the profitability model: the same 16,511 wallets and 2,661 confirmed markets, and the model's 5 history features (trade count, market count, buy amount, buy concentration, average buy size) as the behaviour space. Features are whole-week totals per wallet, not before each bet.",
               vs_title = "Same outliers as with the balanced features?",
               vs_text = "The same wallets analysed with the balanced 5 trading features. A small overlap means the outlier list depends on which features define behaviour.",
               axis_text = STUDY_AXIS_TEXT),
    study_profitable = list(title = "Study features, profitable", results_dir = "alpha_results_pca3d_study_profitable",
               compare_dir = "alpha_results_pca3d_balanced_verified_profitable", compare_label = "balanced-feature profitable analysis", all_profitable = TRUE,
               feature_label = "the profitability model's 5 wallet-history features (confirmed markets)",
               note = "Confirmed-market winners only, in the profitability model's feature space (same axes as the Study features view).",
               vs_title = "Same outliers as with the balanced features?",
               vs_text = "The same confirmed-market winners analysed with the balanced 5 trading features.",
               axis_text = STUDY_AXIS_TEXT)),
  feature_label = "5 balanced trading features",
  wallet_file = "../wallet_level_table/polymarket_wallets.csv",
  out_file    = "interactive_3d/wallet_data.js",
  # axis descriptions: c(negative end, positive end), read from the loadings (pca_loadings.csv)
  axis_text = list(
    PC1 = c("buys long shots", "buys favourites"),
    PC2 = c("large-scale trader: many markets, big bets, sells", "small, passive wallet"),
    PC3 = c("few big bets", "many small bets"))
)

coord_key = function(m) do.call(paste, c(as.data.frame(m), sep = "_"))

# surface: keep only the vertices the surface uses, renumbered from 0 for plotly
export_mesh = function(s){
  v = sort(unique(as.vector(s$faces))); remap = match(seq_len(nrow(s$pts)), v)
  list(x = round(s$pts[v, 1], 3), y = round(s$pts[v, 2], 3), z = round(s$pts[v, 3], 3),
       i = remap[s$faces[, 1]] - 1L, j = remap[s$faces[, 2]] - 1L, k = remap[s$faces[, 3]] - 1L)
}

export_points = function(f){
  out = list(
  w   = substr(f$wallet, 1, 10),
  x   = round(f$PC1, 3), y = round(f$PC2, 3), z = round(f$PC3, 3),
  o   = as.integer(f$pool_excluded),
  p   = as.integer(f$profitable_wallet),
  pol = as.integer(f$share_dollars_politics_other > 0.5),
  vol = round(f$total_volume_dollars, 0), bet = round(f$mean_trade_dollars, 1),
  px  = round(f$buy_vwap_price, 3), ls = round(f$longshot_buy_share, 3), roi = round(f$gross_roi, 3))
  if ("sh_fills" %in% names(f)) out$study = list(                    # study features (confirmed markets)
    tr = f$sh_fills, mk = f$sh_markets, bd = round(f$sh_buy_dollars, 0),
    hhi = round(f$sh_buy_hhi, 3), ab = round(f$sh_avg_buy, 1))
  out
}

# grid: older results name the columns bet_size / longshot; the spec (axis names) is saved by run_profile()
read_grid = function(pd){
  g = read.csv(file.path(pd, "enrichment_grid.csv"))
  names(g)[names(g) == "bet_size"] = "row"; names(g)[names(g) == "longshot"] = "col"
  sf = file.path(pd, "enrichment_grid_spec.json")
  spec = if (file.exists(sf)) fromJSON(sf) else
    list(row_title = "Average bet size", row_labels = c("< $10", "$10-100", "$100-1k", "> $1k"),
         col_title = "Long shots", col_labels = c("none", "0-25%", "25-50%", "> 50%"))
  list(cells = g, spec = spec)
}

# overlap of the outlier sets of two feature choices
compare_outliers = function(f, other_dir, label){
  path = file.path(other_dir, "wallet_alpha_flags.csv")
  if (is.null(other_dir) || !file.exists(path)) return(NULL)
  g = read.csv(path, colClasses = c(wallet = "character"))[, c("wallet", "pool_excluded")]
  m = merge(f[, c("wallet", "pool_excluded")], g, by = "wallet", suffixes = c("_this", "_other"))
  a = m$pool_excluded_this; b = m$pool_excluded_other
  list(label = label, this = sum(a), other = sum(b), both = sum(a & b),
       jaccard = sum(a & b) / sum(a | b),
       share_of_other_kept = sum(a & b) / sum(b),
       lift_other_in_this = mean(a[b]) / mean(a))   # how much more likely the other set's outliers are to be outliers here
}

# one view of the page: points, surface and evidence of one analysis
build_set = function(rd, w, compare_dir, compare_label, title, text = list(), cfg = EXPORT_CONFIG){
  s = readRDS(file.path(rd, "pool_shape_3d.rds"))
  f = read.csv(file.path(rd, "wallet_alpha_flags.csv"), colClasses = c(wallet = "character"))
  f = merge(f, w[, c("wallet", "mean_trade_dollars")], by = "wallet")
  # every wallet maps to a point of the shape, and its flag matches that point's status
  idx = match(coord_key(as.matrix(f[, c("PC1", "PC2", "PC3")])), coord_key(round(s$pts, 6)))
  stopifnot(!anyNA(idx), all(f$pool_excluded == !s$in_shape[idx]))
  pd = file.path(rd, "outlier_profile"); rob = file.path(pd, "k_robustness.csv")
  list(title = title, points = export_points(f), mesh = export_mesh(s),
       grid = read_grid(pd),
       comparison = read.csv(file.path(pd, "feature_comparison.csv")),
       model = read.csv(file.path(pd, "outlier_logistic_model.csv")),
       robustness = if (file.exists(rob)) read.csv(rob) else NULL,
       versus = compare_outliers(f, compare_dir, compare_label),
       k = s$k, n_wallets = nrow(f), n_outliers = sum(f$pool_excluded), n_faces = nrow(s$faces),
       note = text$note, vs_title = text$vs_title, vs_text = text$vs_text,
       all_profitable = isTRUE(text$all_profitable),
       loadings = read.csv(file.path(rd, "pca_loadings.csv")), variance = read.csv(file.path(rd, "pca_variance.csv")),
       axis_text = if (is.null(text$axis_text)) cfg$axis_text else text$axis_text,
       feature_label = if (is.null(text$feature_label)) cfg$feature_label else text$feature_label)
}

run_export = function(cfg = EXPORT_CONFIG){
  w = read.csv(cfg$wallet_file, colClasses = c(wallet = "character"))
  views = cfg$views[sapply(cfg$views, function(v) file.exists(file.path(v$results_dir, "pool_shape_3d.rds")))]
  skipped = setdiff(names(cfg$views), names(views))
  if (length(skipped)) message("not run yet, left off the page: ", paste(skipped, collapse = ", "))
  sets = lapply(views, function(v) build_set(v$results_dir, w, v$compare_dir, v$compare_label, v$title,
                                             v[c("note", "vs_title", "vs_text", "all_profitable", "axis_text", "feature_label")], cfg))
  rd = cfg$views[[1]]$results_dir
  out = list(sets = sets, view_order = names(views),
             loadings = read.csv(file.path(rd, "pca_loadings.csv")),
             variance = read.csv(file.path(rd, "pca_variance.csv")),
             axis_text = cfg$axis_text, feature_label = cfg$feature_label)
  dir.create(dirname(cfg$out_file), showWarnings = FALSE)
  writeLines(paste0("window.WALLET_DATA=", toJSON(out, dataframe = "rows", digits = NA, auto_unbox = TRUE, null = "null"), ";"),
             cfg$out_file)
  for (v in names(sets)) message(sprintf("%s: %d wallets, %d outliers, %d surface triangles", v,
                                         sets[[v]]$n_wallets, sets[[v]]$n_outliers, sets[[v]]$n_faces))
  message(sprintf("%s: %.2f MB", cfg$out_file, file.size(cfg$out_file) / 1e6))
  invisible(out)
}

if (sys.nframe() == 0) run_export()
