\name{polywallet}
\alias{polywallet}
\alias{load_wallets}
\alias{load_pairs}
\alias{wallet_dictionary}
\alias{pair_dictionary}
\alias{wallet_features}
\alias{wallet_model_data}
\alias{default_log_vars}
\alias{insider_columns}
\alias{insider_table}
\alias{dropped_columns}
\alias{pw_file}
\title{Wallet-level Polymarket data}
\description{
One row per wallet for 2026-09-29 to 2026-10-05. Activity columns come from all
6.8 million raw fills. Profit columns use every market with a recorded outcome
(13,794); columns ending in \code{_verified} use only the 2,661 markets whose
outcome was confirmed on the blockchain. \code{load_pairs} returns the
wallet-by-market table the profit columns were built from.
}
\usage{
load_wallets(file = pw_file("data"),
             sample = c("all", "profit", "labelled", "verified"),
             parse_times = TRUE)
load_pairs(file = pw_file("pairs"), labelled_only = FALSE, parse_times = TRUE)
wallet_dictionary()
pair_dictionary()
wallet_features(group = "activity", exclude_times = TRUE)
default_log_vars()
wallet_model_data(df = load_wallets(sample = "labelled"),
                  target = "profitable_wallet",
                  features = wallet_features("activity"),
                  log_vars = default_log_vars(),
                  min_pairs = 1L, drop_na = TRUE)
insider_columns()
insider_table(df = load_wallets(),
              cols = unlist(insider_columns(), use.names = FALSE),
              row_filter = function(d) !is.na(d$profitable_wallet))
dropped_columns(df, cols = unlist(insider_columns(), use.names = FALSE))
pw_file(name = c("data", "dictionary", "pairs", "pair_dictionary"))
}
\arguments{
  \item{file}{Path to a CSV (plain or .gz). Defaults to the bundled copy.}
  \item{sample}{"all" (97,613), "profit" (45,696), "labelled" (43,472) or "verified" (16,511).}
  \item{parse_times}{Convert timestamp columns to POSIXct (UTC).}
  \item{labelled_only}{Drop pairs without a profit label.}
  \item{group}{Dictionary group(s): "id", "outcome", "verified" or "activity".}
  \item{exclude_times}{Drop the two timestamp columns from the feature list.}
  \item{df}{A data frame from \code{load_wallets}.}
  \item{target}{Outcome column to model, e.g. "profitable_wallet" or "profitable_wallet_verified".}
  \item{features}{Predictor columns.}
  \item{log_vars}{Columns transformed with \code{log1p}.}
  \item{min_pairs}{Minimum labelled pairs per wallet (counted on the verified set for a verified target).}
  \item{drop_na}{Keep complete cases only.}
  \item{cols}{Columns to keep.}
  \item{row_filter}{Function of the data frame returning a logical row filter.}
  \item{name}{Which bundled file.}
}
\value{Data frames, or character vectors / a list for the column helpers.}
\examples{
w   <- load_wallets(sample = "labelled")
ins <- insider_table(w)
md  <- wallet_model_data(w)
fit <- glm(profitable_wallet ~ ., data = md[, -1], family = binomial)
}
