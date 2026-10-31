# polywallet 0.2.0 - wallet-level Polymarket data, week 2026-09-29 .. 2026-10-05

# ---- files -------------------------------------------------------------------
pw_file <- function(name = c("data", "dictionary", "pairs", "pair_dictionary")) {
  name <- match.arg(name)
  f <- switch(name,
              data            = "polymarket_wallets.csv.gz",
              dictionary      = "polymarket_wallets_dictionary.csv",
              pairs           = "polymarket_wallet_market_pairs.csv.gz",
              pair_dictionary = "polymarket_wallet_market_pairs_dictionary.csv")
  path <- system.file("extdata", f, package = "polywallet")
  if (!nzchar(path)) stop("bundled file not found: ", f)
  path
}

wallet_dictionary <- function() read.csv(pw_file("dictionary"), stringsAsFactors = FALSE)
pair_dictionary   <- function() read.csv(pw_file("pair_dictionary"), stringsAsFactors = FALSE)

.open <- function(file) if (grepl("\\.gz$", file)) gzfile(file) else file

.parse_times <- function(df, cols) {
  for (v in intersect(cols, names(df)))
    df[[v]] <- as.POSIXct(df[[v]], format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  df
}

.wallet_col_classes <- function(dict = wallet_dictionary()) {
  cls <- ifelse(grepl("^integer", dict$type), "integer",
         ifelse(grepl("^numeric", dict$type), "numeric", "character"))
  setNames(cls, dict$column)
}

# ---- loaders -----------------------------------------------------------------
# sample: "all"      every wallet seen trading                        (97,613)
#         "profit"   traded >= 1 market with a recorded outcome        (45,696)
#         "labelled" non-NA profitable_wallet                          (43,472)
#         "verified" non-NA profitable_wallet_verified (original study)(16,511)
.wallet_samples <- list(
  all      = function(d) rep(TRUE, nrow(d)),
  profit   = function(d) d$in_profit_sample == 1L,
  labelled = function(d) !is.na(d$profitable_wallet),
  verified = function(d) !is.na(d$profitable_wallet_verified)
)

load_wallets <- function(file = pw_file("data"),
                         sample = c("all", "profit", "labelled", "verified"),
                         parse_times = TRUE) {
  sample <- match.arg(sample)
  df <- read.csv(.open(file), colClasses = .wallet_col_classes(), na.strings = "NA",
                 stringsAsFactors = FALSE)
  if (parse_times) df <- .parse_times(df, c("first_trade_time", "last_trade_time"))
  df <- df[.wallet_samples[[sample]](df), ]
  rownames(df) <- NULL
  df
}

load_pairs <- function(file = pw_file("pairs"), labelled_only = FALSE, parse_times = TRUE) {
  df <- read.csv(.open(file), na.strings = "NA", stringsAsFactors = FALSE,
                 colClasses = c(wallet = "character", market = "character",
                                event_group = "character"))
  if (parse_times) df <- .parse_times(df, c("first_trade_time", "resolved_at"))
  if (labelled_only) df <- df[!is.na(df$profit_label), ]
  rownames(df) <- NULL
  df
}

# ---- feature sets ------------------------------------------------------------
# group: "id", "outcome", "verified" or "activity"
wallet_features <- function(group = "activity", exclude_times = TRUE) {
  d <- wallet_dictionary()
  v <- d$column[d$group %in% group]
  if (exclude_times) v <- setdiff(v, c("first_trade_time", "last_trade_time"))
  v
}

default_log_vars <- function() c(
  "n_trades", "n_buy_trades", "n_sell_trades", "n_markets", "n_distinct_timestamps",
  "trades_per_active_day", "trades_per_market", "median_gap_minutes",
  "total_volume_dollars", "buy_dollars", "sell_dollars", "mean_trade_dollars",
  "median_trade_dollars", "max_trade_dollars", "buy_shares", "n_duplicate_rows",
  "active_span_hours")

# Complete-case modelling frame: wallet + target + predictors (log1p on heavy tails)
wallet_model_data <- function(df = load_wallets(sample = "labelled"),
                              target = "profitable_wallet",
                              features = wallet_features("activity"),
                              log_vars = default_log_vars(),
                              min_pairs = 1L,
                              drop_na = TRUE) {
  stopifnot(target %in% names(df), all(features %in% names(df)))
  n_col <- if (grepl("_verified$", target)) "n_pairs_labelled_verified" else "n_pairs_labelled"
  df <- df[!is.na(df[[target]]) & df[[n_col]] >= min_pairs, ]
  out <- df[, c("wallet", target, features)]
  for (v in intersect(log_vars, features)) out[[v]] <- log1p(out[[v]])
  if (drop_na) out <- out[complete.cases(out), ]
  rownames(out) <- NULL
  out
}

# ---- insider screening -------------------------------------------------------
insider_columns <- function() {
  list(
    id = "wallet",
    performance = c("profitable_wallet", "gross_pnl_dollars", "gross_roi", "pair_win_rate",
                    "max_pair_pnl_dollars", "n_pairs_labelled", "labelled_buy_dollars",
                    "mean_entry_buy_price_labelled"),
    data_quality = c("n_pairs_labelled_verified", "n_pairs_resolved_after_week",
                     "profitable_wallet_verified"),
    wallet_profile = c("n_trades", "n_markets", "n_active_days", "first_trade_time",
                       "active_span_hours", "median_gap_minutes", "sell_trade_share"),
    bet_style = c("total_volume_dollars", "mean_trade_dollars", "max_trade_dollars",
                  "buy_vwap_price", "longshot_buy_share"),
    concentration = c("market_hhi", "top_market_dollar_share"),
    market_type = c("share_dollars_politics_other", "share_dollars_sports_esports",
                    "share_dollars_weather", "share_dollars_crypto_price",
                    "share_dollars_crypto_updown")
  )
}

insider_table <- function(df = load_wallets(),
                          cols = unlist(insider_columns(), use.names = FALSE),
                          row_filter = function(d) !is.na(d$profitable_wallet)) {
  stopifnot(all(cols %in% names(df)))
  out <- df[row_filter(df), cols]
  rownames(out) <- NULL
  out
}

dropped_columns <- function(df, cols = unlist(insider_columns(), use.names = FALSE)) {
  setdiff(names(df), cols)
}
