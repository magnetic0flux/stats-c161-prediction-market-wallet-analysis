# Example: load the wallet table and fit a first model
# Option A - plain CSV, no package needed
wallets  <- read.csv("polymarket_wallets.csv", stringsAsFactors = FALSE,
                     colClasses = c(wallet = "character"))   # keep 0x addresses as text
labelled <- subset(wallets, !is.na(profitable_wallet))      # 43,472 wallets with a target

# Option B - the polywallet package
# install.packages("polywallet_0.2.0.tar.gz", repos = NULL, type = "source")
library(polywallet)
w     <- load_wallets(sample = "labelled")    # or "verified" for the blockchain-confirmed subset
pairs <- load_pairs(labelled_only = TRUE)     # one row per wallet x market
md    <- wallet_model_data(w, target = "profitable_wallet")
X     <- md[, setdiff(names(md), "wallet")]

# activity and profit come from the same week: descriptive, not a forecast
set.seed(1)
idx <- sample(nrow(X), 0.7 * nrow(X))
fit <- glm(profitable_wallet ~ ., data = X[idx, ], family = binomial)
p   <- predict(fit, X[-idx, ], type = "response")
mean((p > 0.5) == X$profitable_wallet[-idx])   # hold-out accuracy
mean(X$profitable_wallet[-idx])                # share profitable (baseline)

# robustness: same model on blockchain-confirmed markets only
mv <- wallet_model_data(load_wallets(sample = "verified"), target = "profitable_wallet_verified")
