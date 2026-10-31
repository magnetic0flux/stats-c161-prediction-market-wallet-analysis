# ==============================================================================
# Alpha shape analysis of wallet behaviour (AlphaRichness on Polymarket wallets)
#
# Species -> wallets; trait space -> behaviour space; community -> group of wallets.
# Questions:
#   1. How much of behaviour space does each group of wallets occupy, and how gappy is it?
#      (alpha volume, convex hull volume, occupancy = alpha / hull, clusters)
#   2. Which wallets sit outside everyone else's behaviour (left out of the alpha shape)?
#      -> behavioural outliers, a candidate list for insider screening.
#
# Run from this script's folder (setwd() to Machine Learning/alpha_shape_wallets), then source() it.
# Everything is configured in CONFIG. The behaviour space is built by a swappable trait function:
#   traits_pca()     - PCA of the activity features, first n_pc components (default, 3D)
#   traits_columns() - hand-picked columns, z-scored (the earlier 2D version)
# Feature sets for traits_pca(): pca_features (13), pca_features_balanced (5, default) and
# pca_features_study (the profitability study's 5 wallet-history features; run_study_features.R).
# AlphaRichness is sourced unchanged into its own environment `ar`. One function is replaced there:
# mst_edges() -> mst_edges_delaunay(), the same Euclidean MST built from Delaunay edges instead of
# all n^2 pairs (identical results once coordinates are rounded; checked in every run).
# ==============================================================================

suppressMessages({ library(geometry); library(igraph); library(ggplot2) })

# ------------------------------------------------------------------------------
# Behaviour-space builders (swap via CONFIG$trait_fn). Each returns a numeric matrix with one row
# per wallet (rownames = wallet) and, optionally, attributes describing how it was built.
# ------------------------------------------------------------------------------
pca_features = function(w) data.frame(
  log_volume     = log1p(w$total_volume_dollars),   # how much money traded
  log_trades     = log1p(w$n_trades),
  log_markets    = log1p(w$n_markets),
  active_days    = w$n_active_days,
  log_span_hours = log1p(w$active_span_hours),
  log_mean_trade = log1p(w$mean_trade_dollars),     # bet size
  log_max_trade  = log1p(w$max_trade_dollars),
  sell_share     = w$sell_trade_share,               # buy-and-hold vs trading back and forth
  buy_price      = w$buy_vwap_price,                 # price style
  longshot       = w$longshot_buy_share,
  favorite       = w$favorite_buy_share,
  hhi            = w$market_hhi,                     # concentration
  top_share      = w$top_market_dollar_share)

# Balanced set: volume left out (it is trades x bet size) and no two features with |Spearman rho| > 0.75,
# so no behaviour is counted twice. Max remaining |rho| = 0.70 (buy price vs long-shot share).
pca_features_balanced = function(w) data.frame(
  log_markets    = log1p(w$n_markets),               # breadth (stands in for trades, HHI, top-market share)
  log_mean_trade = log1p(w$mean_trade_dollars),     # bet size (stands in for largest bet)
  buy_price      = w$buy_vwap_price,                 # price level (stands in for favourite share)
  longshot       = w$longshot_buy_share,             # long-shot tilt
  sell_share     = w$sell_trade_share)               # buy-and-hold vs trading back and forth

# The profitability study's 5 wallet-history features (03_final_validation.ipynb: EXPERIENCE + the two
# 5-feature extras), measured per wallet over the whole week on the chosen markets. Same definitions:
# ln(1 + fills), ln(1 + markets), ln(1 + buy dollars), buy-dollar HHI across markets (NA if no buy dollars),
# ln(1 + buy dollars / buy fills) (NA if no buy fills). Columns come from add_study_features().
pca_features_study = function(w) data.frame(
  log_trades      = log1p(w$sh_fills),
  log_markets     = log1p(w$sh_markets),
  log_buy_dollars = log1p(w$sh_buy_dollars),
  buy_hhi         = w$sh_buy_hhi,
  log_avg_buy     = log1p(w$sh_avg_buy))

# Per-wallet study features from the pair table (+ buy-fill counts, build_scripts/pair_buy_fills.py).
# market_filter picks the pairs: default = the 2,661 blockchain-confirmed markets. All pairs on those
# markets count, labelled or not (activity, not profit).
study_history_features = function(pairs_file = "../wallet_level_table/polymarket_wallet_market_pairs.csv.gz",
                                  buy_fills_file = "../wallet_level_table/polymarket_wallet_market_pair_buy_fills.csv.gz",
                                  market_filter = function(p) p$verified_market == 1){
  p = read.csv(pairs_file, colClasses = c(wallet = "character"))[, c("wallet", "market", "verified_market", "fills", "buy_dollars")]
  b = read.csv(buy_fills_file, colClasses = c(wallet = "character"))
  p = merge(p, b, by = c("wallet", "market"))
  p = p[market_filter(p), ]
  tot = tapply(p$buy_dollars, p$wallet, sum)
  share = p$buy_dollars / tot[p$wallet]
  out = data.frame(wallet = names(tot),
    sh_fills       = as.numeric(tapply(p$fills, p$wallet, sum)[names(tot)]),
    sh_markets     = as.numeric(table(p$wallet)[names(tot)]),
    sh_buy_dollars = as.numeric(tot),
    sh_buy_hhi     = as.numeric(tapply(share^2, p$wallet, sum)[names(tot)]),
    sh_buy_fills   = as.numeric(tapply(p$buy_fills, p$wallet, sum)[names(tot)]))
  out$sh_buy_hhi[out$sh_buy_dollars == 0] = NA
  out$sh_avg_buy = ifelse(out$sh_buy_fills > 0, out$sh_buy_dollars / out$sh_buy_fills, NA)
  out
}

# label_fn wrapper: add the study features to the wallet table (wallets without such pairs get NA)
add_study_features = function(label_fn = label_verified_markets(), ...){
  f = study_history_features(...)
  function(w){
    w = label_fn(w)
    w = merge(w, f, by = "wallet", all.x = TRUE, sort = FALSE)
    w[order(w$wallet), ]
  }
}

traits_pca = function(w, n_pc = 3, feature_fn = pca_features){
  X = feature_fn(w)
  p = stats::prcomp(X, scale. = TRUE)               # standardise features, then rotate
  S = p$x[, seq_len(n_pc), drop = FALSE]            # raw PC scores: keep each PC's own variance
  rownames(S) = w$wallet
  attr(S, "loadings") = p$rotation[, seq_len(n_pc), drop = FALSE]
  attr(S, "variance_explained") = summary(p)$importance[2, seq_len(n_pc)]
  S
}

traits_columns = function(w, axes = list(
    log_volume    = function(w) log1p(w$total_volume_dollars),
    concentration = function(w) w$market_hhi)){
  S = scale(sapply(axes, function(f) f(w)))
  rownames(S) = w$wallet
  S
}

# ------------------------------------------------------------------------------
# Profit label (swap via CONFIG$label_fn). Each takes the wallet table and returns it with the
# outcome columns (profitable_wallet, gross_pnl_dollars, gross_roi, pair_win_rate, n_pairs_labelled)
# set to the chosen label. The all-market label is always kept as profitable_wallet_all_markets,
# so the PCA axes can still be fitted on the same wallets whichever label is used.
# ------------------------------------------------------------------------------
label_all_markets = function(w){                    # default: all 13,794 resolved markets
  w$profitable_wallet_all_markets = w$profitable_wallet
  w
}

# Blockchain-confirmed label: only the 2,661 markets whose outcome was confirmed on-chain (the
# profitability study's markets). ROI is rebuilt from the pair table (buy spending in confirmed,
# labelled pairs); the summed profit is checked against gross_pnl_dollars_verified.
label_verified_markets = function(pairs_file = "../wallet_level_table/polymarket_wallet_market_pairs.csv.gz"){
  function(w){
    p = read.csv(pairs_file, colClasses = c(wallet = "character"))
    p = p[p$verified_market == 1 & p$negative_inventory == 0, ]
    buy = tapply(p$buy_dollars, p$wallet, sum); pnl = tapply(p$gross_pnl_dollars, p$wallet, sum)
    w$profitable_wallet_all_markets = w$profitable_wallet
    w$profitable_wallet = w$profitable_wallet_verified
    w$gross_pnl_dollars = w$gross_pnl_dollars_verified
    w$pair_win_rate     = w$pair_win_rate_verified
    w$n_pairs_labelled  = w$n_pairs_labelled_verified
    w$gross_roi         = unname(pnl[w$wallet] / buy[w$wallet])
    ok = !is.na(w$profitable_wallet)
    stopifnot(all(abs(pnl[w$wallet[ok]] - w$gross_pnl_dollars[ok]) < 1e-3))
    w
  }
}

CONFIG = list(
  wallet_file = "../wallet_level_table/polymarket_wallets.csv",  # wallet table (v0.2)
  pkg_dir     = "../../Alpha Shape/AlphaRichness",               # AlphaRichness source (relative to this script)
  out_dir     = "alpha_results_pca3d_balanced",
  # profit label: label_all_markets (default) or label_verified_markets() (2,661 confirmed markets)
  label_fn    = label_all_markets,
  # behaviour space: balanced 5-feature PCA (default). Original 13-feature version:
  #   trait_fn = function(w) traits_pca(w, n_pc = 3, feature_fn = pca_features), out_dir = "alpha_results_pca3d"
  trait_fn    = function(w) traits_pca(w, n_pc = 3, feature_fn = pca_features_balanced),
  axis_labels = c("PC1", "PC2", "PC3"),
  # wallets the PCA axes are fitted on, and the wallets analysed (a subset keeps the same axes)
  space_filter = function(w) !is.na(w$profitable_wallet_all_markets),  # all 43,472 labelled wallets
  pool_filter  = function(w) !is.na(w$profitable_wallet),        # e.g. w$profitable_wallet %in% 1 for winners only
  # communities: function(wallet table) -> label per wallet (NA = not in any community)
  community_fn = function(w) paste(
    ifelse(w$share_dollars_politics_other > 0.5, "Politics/other", "Sports/crypto/weather"),
    ifelse(w$profitable_wallet == 1, "profitable", "unprofitable"), sep = " | "),
  # what sets each community's gap cutoff: "own" = its own MST (fair when communities are much sparser
  # than the pool, as here); "pool" = the pool's MST (AlphaRichness's two-step default)
  community_gap_ref = "own",
  round_digits = 6,          # rounding makes near-duplicates exact duplicates (removed by clean_dataset)
  k_range      = c(1.5, 3),
  k_check      = c(1.5, 2, 2.5, 3),   # other k values for the robustness check (MST rule only)
  n_rarefy     = 10,         # random equal-size draws per community for size-fair comparison
  seed         = 1,
  plot_3d_html = TRUE        # interactive 3D plots (needs plotly); static 2D projections are always made
)

# ------------------------------------------------------------------------------
# 0. AlphaRichness, sourced into its own environment
# ------------------------------------------------------------------------------
load_alpharichness = function(pkg_dir){
  ar = new.env()
  for (f in list.files(file.path(pkg_dir, "R"), pattern = "\\.R$", full.names = TRUE)) sys.source(f, envir = ar)
  ar
}

# Euclidean MST from the Delaunay edges: same output as ar$mst_edges(), O(n log n) instead of O(n^2).
mst_edges_delaunay = function(pts){
  n = nrow(pts)
  if (n < 2) return(data.frame(from = integer(0), to = integer(0), len = numeric(0)))
  if (n <= ncol(pts) + 1) return(mst_edges_full(pts))
  simp  = delaunay_simplices(pts)
  pairs = utils::combn(ncol(simp), 2)
  e = do.call(rbind, lapply(seq_len(ncol(pairs)), function(j) simp[, pairs[, j]]))
  e = unique(cbind(pmin(e[, 1], e[, 2]), pmax(e[, 1], e[, 2])))
  len = sqrt(rowSums((pts[e[, 1], , drop = FALSE] - pts[e[, 2], , drop = FALSE])^2))
  g = igraph::make_graph(t(e), n = n, directed = FALSE)
  igraph::E(g)$weight = len
  tr = igraph::mst(g)
  ends = igraph::as_edgelist(tr, names = FALSE)
  data.frame(from = ends[, 1], to = ends[, 2], len = igraph::E(tr)$weight)
}

use_fast_mst = function(ar){
  ar$mst_edges_full = ar$mst_edges                    # keep the original for checks
  f = mst_edges_delaunay; environment(f) = ar         # look up delaunay_simplices() etc. in ar
  ar$mst_edges = f
  ar
}

# The k scan also runs the full algorithm per distinct excluded set, only to draw plot_stable_k().
# With tens of thousands of points that is hours, so it is switched off; k is chosen exactly as before.
skip_scan_display = function(ar){
  ar$alpha_group_count = function(pts, k) c(n_clusters = NA_real_, n_excluded = NA_real_, n_groups = NA_real_)
  ar
}

# Original vs fast MST on a random subsample: same excluded points and same alpha shape?
check_mst_equivalence = function(ar, pts, n = 2000, k = 2, seed = 1){
  set.seed(seed)
  sub = pts[sample(nrow(pts), min(n, nrow(pts))), , drop = FALSE]
  fast = ar$mst_edges
  ar$mst_edges = ar$mst_edges_full; r1 = ar$find_alpha_general(sub, k = k)
  ar$mst_edges = fast;              r2 = ar$find_alpha_general(sub, k = k)
  c(n_points = nrow(sub),
    same_excluded = identical(r1$point_cluster == 0, r2$point_cluster == 0),
    same_shape = identical(r1$simp_included, r2$simp_included))
}

# ------------------------------------------------------------------------------
# 1. Behaviour space
# ------------------------------------------------------------------------------
build_traits = function(w, trait_fn, round_digits){
  S = trait_fn(w)
  out = round(S, round_digits)
  for (a in setdiff(names(attributes(S)), c("dim", "dimnames"))) attr(out, a) = attr(S, a)
  out
}

# clean_dataset() keeps one wallet per set of identical coordinates; map every wallet to that point
coord_key = function(m) do.call(paste, c(as.data.frame(unclass(m)), sep = "_"))
map_to_points = function(traits, pts) match(coord_key(traits), coord_key(pts))

# ------------------------------------------------------------------------------
# 2. Metrics for one community
# ------------------------------------------------------------------------------
community_metrics = function(ar, traits, pool_pts, k, gap_ref = pool_pts){
  res = ar$alpha_richness(traits, gap_ref = gap_ref, k = k)
  pts = res$pts
  hull_own  = ar$convex_hull_volume(pts)
  hull_pool = ar$convex_hull_volume(pool_pts)
  list(result = res,
       metrics = data.frame(n_wallets = nrow(traits), n_points = nrow(pts),
                            alpha_volume = res$volume, hull_volume = hull_own,
                            occupancy = res$volume / hull_own,             # 1 - void ratio
                            void_ratio = 1 - res$volume / hull_own,
                            share_of_pool_hull = res$volume / hull_pool,
                            alpha_val = res$alpha_val, n_clusters = res$n_clusters,
                            n_mst_dropped = sum(res$point_cluster == 0),
                            n_excluded_points = res$cut_count,
                            excluded_share = res$cut_count / nrow(pts)))
}

# Equal-size random draws so communities of different size are compared fairly
rarefy_metrics = function(ar, traits, pool_pts, k, n_draw, reps, seed, gap_ref = pool_pts){
  set.seed(seed)
  do.call(rbind, lapply(seq_len(reps), function(i){
    sub = traits[sample(nrow(traits), n_draw), , drop = FALSE]
    community_metrics(ar, sub, pool_pts, k, gap_ref)$metrics
  }))
}

summarise_draws = function(d) data.frame(
  rarefied_n = d$n_wallets[1],
  occupancy_mean = mean(d$occupancy), occupancy_sd = sd(d$occupancy),
  share_of_pool_hull_mean = mean(d$share_of_pool_hull), share_of_pool_hull_sd = sd(d$share_of_pool_hull),
  n_clusters_mean = mean(d$n_clusters), excluded_share_mean = mean(d$excluded_share))

# ------------------------------------------------------------------------------
# 3. Wallet-level flags
# ------------------------------------------------------------------------------
point_status = function(res){
  pts = res$pts
  in_shape = seq_len(nrow(pts)) %in% res$simp_included
  cl = rep(NA_integer_, nrow(pts))                     # shape cluster of each point
  if (length(res$simplex_cluster)) {
    v = as.vector(res$simp_included); cl[v] = rep(res$simplex_cluster, times = ncol(res$simp_included))
  }
  data.frame(excluded = !in_shape, mst_dropped = res$point_cluster == 0, shape_cluster = cl,
             row.names = rownames(pts))
}

flag_wallets = function(traits, res, prefix){
  st = point_status(res)
  out = st[map_to_points(traits, res$pts), ]; rownames(out) = rownames(traits)
  names(out) = paste0(prefix, "_", names(out))
  out
}

summarise_outliers = function(flags) do.call(rbind, lapply(split(flags, flags$pool_excluded), function(d) data.frame(
  pool_excluded = d$pool_excluded[1], wallets = nrow(d),
  share_profitable = mean(d$profitable_wallet), median_roi = median(d$gross_roi, na.rm = TRUE),
  median_volume = median(d$total_volume_dollars), median_trades = median(d$n_trades),
  median_hhi = median(d$market_hhi), median_buy_price = median(d$buy_vwap_price),
  median_longshot_share = median(d$longshot_buy_share),
  share_politics_dominant = mean(d$share_dollars_politics_other > 0.5))))

# ------------------------------------------------------------------------------
# 4. Plots (minimal)
# ------------------------------------------------------------------------------
plot_k_choice = function(ks){
  ggplot(ks$scan, aes(k, n_dropped)) +
    annotate("rect", xmin = ks$pool_range[1], xmax = ks$pool_range[2], ymin = -Inf, ymax = Inf, fill = "grey92") +
    geom_step(linewidth = 0.5) + geom_vline(xintercept = ks$k, linetype = "dashed", linewidth = 0.4) +
    labs(x = "k", y = "Wallets dropped by the MST rule (pool)", subtitle = sprintf("chosen k = %s", ks$k)) +
    theme_classic()
}

plot_loadings = function(L, ve){
  d = data.frame(feature = rep(rownames(L), ncol(L)), pc = rep(colnames(L), each = nrow(L)), loading = as.vector(L))
  d$pc = factor(d$pc, levels = colnames(L), labels = sprintf("%s (%.0f%%)", colnames(L), 100 * ve))
  d$feature = factor(d$feature, levels = rev(rownames(L)))
  ggplot(d, aes(loading, feature)) + geom_col(width = 0.6, fill = "grey40") +
    geom_vline(xintercept = 0, linewidth = 0.3) + facet_wrap(~pc) +
    labs(x = "Loading", y = NULL) + theme_classic()
}

# Static view: every pair of axes, included wallets grey, wallets left out of the shape black
plot_projections = function(traits, excluded){
  pr = utils::combn(ncol(traits), 2)
  d = do.call(rbind, lapply(seq_len(ncol(pr)), function(j) data.frame(
    x = traits[, pr[1, j]], y = traits[, pr[2, j]], excluded = excluded,
    panel = sprintf("%s (y) vs %s (x)", colnames(traits)[pr[2, j]], colnames(traits)[pr[1, j]]))))
  d = d[order(d$excluded), ]
  ggplot(d, aes(x, y, colour = excluded)) + geom_point(size = 0.15) +
    scale_colour_manual(values = c(`FALSE` = "grey75", `TRUE` = "black"),
                        labels = c("in alpha shape", "left out"), name = NULL) +
    facet_wrap(~panel, scales = "free") + labs(x = NULL, y = NULL) +
    guides(colour = guide_legend(override.aes = list(size = 2))) +
    theme_classic() + theme(legend.position = "bottom")
}

save_shape_plot = function(ar, res, title, labels, pool_pts, file){
  p = ar$plot_alpha_general(res$pts, hull_data = pool_pts, result = res, point_size = 0.6,
                            fill_col = "grey70", axis_labels = labels)
  if (ncol(res$pts) == 2) {
    ggsave(file, p + ggtitle(title), width = 7, height = 6, dpi = 200, bg = "white")
  } else {
    p = plotly::layout(p, title = title)
    htmlwidgets::saveWidget(p, file, selfcontained = TRUE)
  }
}

# ------------------------------------------------------------------------------
# 5. Run
# ------------------------------------------------------------------------------
# "pool": two steps (pool range, refined by communities); "own": one step on the pool, the same k then
# applied to every community with its own MST
# Pool shape surface (for the 3D page) and the MST-dropped set at other k (robustness), saved once
save_pool_shape = function(ar, res, k, k_check, out_dir){
  pts = res$pts
  saveRDS(list(pts = pts, faces = ar$boundary_faces(res$simp_included), k = k,
               in_shape = seq_len(nrow(pts)) %in% res$simp_included, mst_dropped = res$point_cluster == 0),
          file.path(out_dir, "pool_shape_3d.rds"))
  mst = ar$mst_edges(pts); q = stats::quantile(mst$len, c(.25, .75)); kg = sort(unique(c(k_check, k)))
  drop = sapply(kg, function(kk){
    gap = mst$len > q[[2]] + kk * (q[[2]] - q[[1]])
    comp = ar$edge_components(nrow(pts), mst$from[!gap], mst$to[!gap]); size = tabulate(comp)
    size[comp] < ncol(pts) + 1 })
  colnames(drop) = paste0("k_", kg); rownames(drop) = rownames(pts)
  saveRDS(drop, file.path(out_dir, "mst_drop_by_k.rds"))
}

find_k = function(ar, comm_traits, pool_pts, k_range, community_gap_ref = "own"){
  if (community_gap_ref == "pool") ar$find_stable_k(unname(comm_traits), gap_ref = pool_pts, k_range = k_range)
  else ar$find_stable_k(pool_pts, k_range = k_range)
}

run_analysis = function(cfg = CONFIG){
  dir.create(cfg$out_dir, showWarnings = FALSE)
  ar = skip_scan_display(use_fast_mst(load_alpharichness(cfg$pkg_dir)))

  w = read.csv(cfg$wallet_file, colClasses = c(wallet = "character"))
  w = cfg$label_fn(w)
  w = w[cfg$space_filter(w), ]
  traits = build_traits(w, cfg$trait_fn, cfg$round_digits)      # axes fitted on the space wallets
  keep = stats::complete.cases(traits) & cfg$pool_filter(w); w = w[keep, ]
  info = attributes(traits)[c("loadings", "variance_explained")]      # subsetting drops attributes
  traits = traits[keep, , drop = FALSE]
  for (a in names(info)) attr(traits, a) = info[[a]]
  pool_pts = ar$clean_dataset(traits)
  community = cfg$community_fn(w)
  comm_traits = split.data.frame(unclass(traits), community)
  message(sprintf("%d wallets -> %d distinct points in %dD; communities: %s", nrow(traits), nrow(pool_pts),
                  ncol(traits), paste(sprintf("%s (%d)", names(comm_traits), sapply(comm_traits, nrow)), collapse = "; ")))

  check = check_mst_equivalence(ar, pool_pts)
  message("fast MST check: ", paste(names(check), check, collapse = ", "))

  ks = find_k(ar, comm_traits, pool_pts, cfg$k_range, cfg$community_gap_ref)
  message(sprintf("k = %s (pool range %s-%s)", ks$k, ks$pool_range[1], ks$pool_range[2]))
  comm_gap = if (cfg$community_gap_ref == "pool") pool_pts else NULL

  pool_run  = community_metrics(ar, traits, pool_pts, ks$k, gap_ref = NULL)          # all wallets
  message("pool shape done")
  save_pool_shape(ar, pool_run$result, ks$k, cfg$k_check, cfg$out_dir)
  comm_runs = lapply(comm_traits, community_metrics, ar = ar, pool_pts = pool_pts, k = ks$k, gap_ref = comm_gap)
  message("community shapes done")
  n_min = min(sapply(comm_traits, nrow))
  rare = lapply(comm_traits, rarefy_metrics, ar = ar, pool_pts = pool_pts, k = ks$k,
                n_draw = n_min, reps = cfg$n_rarefy, seed = cfg$seed, gap_ref = comm_gap)
  message("rarefaction done")

  metrics = cbind(group = c("All labelled wallets", names(comm_runs)),
                  rbind(pool_run$metrics, do.call(rbind, lapply(comm_runs, `[[`, "metrics"))),
                  rbind(NA, do.call(rbind, lapply(rare, summarise_draws))))
  metrics$k = ks$k; metrics$community_gap_ref = cfg$community_gap_ref; metrics$dimensions = ncol(traits)

  flags = data.frame(wallet = rownames(traits), community = community, unclass(traits), check.names = FALSE)
  flags = cbind(flags, flag_wallets(traits, pool_run$result, "pool"))
  comm_flags = do.call(rbind, lapply(names(comm_runs), function(g)
    flag_wallets(comm_traits[[g]], comm_runs[[g]]$result, "community")))
  flags = cbind(flags, comm_flags[flags$wallet, ])
  flags = merge(flags, w[, c("wallet", "profitable_wallet", "gross_pnl_dollars", "gross_roi", "pair_win_rate",
                             "n_pairs_labelled", "n_trades", "total_volume_dollars", "market_hhi", "buy_vwap_price",
                             "longshot_buy_share", "share_dollars_politics_other",
                             grep("^sh_", names(w), value = TRUE))], by = "wallet")       # study features, if added
  outlier_summary = summarise_outliers(flags)

  # save
  write.csv(metrics, file.path(cfg$out_dir, "community_metrics.csv"), row.names = FALSE)
  write.csv(flags, file.path(cfg$out_dir, "wallet_alpha_flags.csv"), row.names = FALSE)
  write.csv(outlier_summary, file.path(cfg$out_dir, "outlier_summary.csv"), row.names = FALSE)
  write.csv(ks$scan[, c("k", "n_dropped")], file.path(cfg$out_dir, "k_scan.csv"), row.names = FALSE)
  ggsave(file.path(cfg$out_dir, "k_choice.png"), plot_k_choice(ks), width = 6, height = 3.5, dpi = 200, bg = "white")
  if (!is.null(attr(traits, "loadings"))){
    L = attr(traits, "loadings"); ve = attr(traits, "variance_explained")
    write.csv(data.frame(feature = rownames(L), round(L, 4)), file.path(cfg$out_dir, "pca_loadings.csv"), row.names = FALSE)
    write.csv(data.frame(pc = names(ve), variance_explained = ve), file.path(cfg$out_dir, "pca_variance.csv"), row.names = FALSE)
    ggsave(file.path(cfg$out_dir, "pca_loadings.png"), plot_loadings(L, ve), width = 8, height = 4, dpi = 200, bg = "white")
  }
  ggsave(file.path(cfg$out_dir, "outliers_projections.png"),
         plot_projections(unclass(traits), flags$pool_excluded[match(rownames(traits), flags$wallet)]),
         width = 10, height = 4, dpi = 200, bg = "white")
  if (ncol(traits) == 2 || (cfg$plot_3d_html && requireNamespace("plotly", quietly = TRUE))){
    ext = if (ncol(traits) == 2) ".png" else ".html"
    save_shape_plot(ar, pool_run$result, "All labelled wallets", cfg$axis_labels, pool_pts,
                    file.path(normalizePath(cfg$out_dir), paste0("alpha_shape_all_wallets", ext)))
    for (g in names(comm_runs))
      save_shape_plot(ar, comm_runs[[g]]$result, g, cfg$axis_labels, pool_pts,
                      file.path(normalizePath(cfg$out_dir), paste0("alpha_shape_", gsub("[^A-Za-z]+", "_", g), ext)))
  }
  invisible(list(ar = ar, config = cfg, traits = traits, k_scan = ks, check = check,
                 pool = pool_run, communities = comm_runs, rarefied = rare,
                 metrics = metrics, flags = flags, outlier_summary = outlier_summary))
}

if (sys.nframe() == 0) {
  t0 = Sys.time()
  out = run_analysis()
  print(out$metrics, digits = 3)
  print(out$outlier_summary, digits = 3)
  message("total time: ", format(Sys.time() - t0))
}
