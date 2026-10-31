# Insider-screening table from the full wallet table (no package needed).
# Rows: wallets with a profit label (all resolved markets).
# Columns: see insider_columns(); everything else is listed by dropped_columns().

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

build_insider_table <- function(wallets, cols = unlist(insider_columns(), use.names = FALSE),
                                row_filter = function(d) !is.na(d$profitable_wallet)) {
  stopifnot(all(cols %in% names(wallets)))
  out <- wallets[row_filter(wallets), cols]
  rownames(out) <- NULL
  out
}

dropped_columns <- function(wallets, cols = unlist(insider_columns(), use.names = FALSE)) {
  setdiff(names(wallets), cols)
}

if (sys.nframe() == 0) {
  wallets <- read.csv("polymarket_wallets.csv", stringsAsFactors = FALSE,
                      colClasses = c(wallet = "character"))   # keep 0x addresses as text
  ins <- build_insider_table(wallets)
  write.csv(ins, "polymarket_wallets_insider.csv", row.names = FALSE, na = "NA")
  cat("rows:", nrow(ins), " columns:", ncol(ins), "\n")
  cat("dropped:", paste(dropped_columns(wallets), collapse = ", "), "\n")
}
