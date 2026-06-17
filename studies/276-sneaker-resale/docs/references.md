# References & literature map — Study 276 (Sneaker-Resale)

## The claim under test

The pitch, repeated endlessly in financial-lifestyle media since ~2017: *sneaker
reselling on StockX / GOAT is a real alternative asset class — "better than the stock
market," with double-digit returns and lower drawdowns.* The marketplaces themselves
leaned into the framing (StockX literally borrowed stock-exchange language: bid/ask,
"the stock market of things"). We test whether the broad sneaker-resale price index
actually beat the S&P 500 over 2007–2024, and whether its attractive risk profile is
real or an artifact.

## Why the "low-risk outperformance" is mostly an illusion

- **Stale-pricing / appraisal smoothing.** Sneaker resale prices are reported as
  infrequent, aggregated appraisals, not a continuously-traded tape. Smoothed series
  exhibit strong positive autocorrelation and *understated* volatility — the same
  problem that flatters private real estate, private equity, and hedge-fund NAVs.
  - **Geltner, D. (1991).** "Smoothing in Appraisal-Based Returns." *Journal of Real
    Estate Finance and Economics*, 4(3), 327–345. The AR(1) unsmoothing we apply.
  - **Geltner, D. (1993).** "Estimating Market Values from Appraised Values without
    Assuming an Efficient Market." *Journal of Real Estate Research*, 8(3), 325–345.
  - **Asness, C., Krail, R. & Liew, J. (2001).** "Do Hedge Funds Hedge?" *Journal of
    Portfolio Management*, 28(1), 6–19. Smoothed/illiquid marks inflate Sharpe ratios
    and hide beta — directly analogous to the sneaker index here.

- **Tiny n.** Eighteen annual observations cannot resolve a return difference smaller
  than ~9%/yr at |t| = 2 given ~18% equity volatility. The observed sneaker−S&P gap is
  negative and far inside the noise band.

- **Survivorship and curation.** Any "sneaker index" is curated from the pairs that
  *had* a liquid resale market — the hyped Jordans, Yeezys, and Dunks. Models that
  never resold, deadstock that rotted in warehouses, and pairs that traded below retail
  are quietly excluded. This is a survivorship bias on the Signal axis.

## The market's boom and bust (the shape the index encodes)

- **StockX, GOAT, and the industrialisation of resale (2016–2021).** The secondary
  sneaker market exploded from a forum-and-eBay cottage industry to a multi-billion-
  dollar marketplace business. Estimates of market size ran from ~$2B (mid-2010s) to
  $6–10B+ by 2020–2021 (Cowen, Piper Sandler, StockX "Big Facts" reports).
- **The 2022–2024 bust.** Resale premiums compressed sharply as hype cooled, supply
  normalised, and macro tightened. StockX cut staff (2022–2023), private valuations
  fell from their ~$3.8B peak, and broad resale price indices gave back much of the
  pandemic-era surge — the downturn captured in the index's 2022–2024 levels.

## Collectibles as an asset class — the academic backdrop

- **Dimson, E. & Spaenjers, C. (2011).** "Ex Post: The Investment Performance of
  Collectible Stamps." *Journal of Financial Economics*, 100(2), 443–458. The canonical
  finding: collectibles deliver modest real returns, high idiosyncratic risk, high
  transaction costs, and look better than they are once illiquidity is priced in.
- **Dimson, E., Rousseau, P. & Spaenjers, C. (2015).** "The Price of Wine." *Journal of
  Financial Economics*, 118(2), 431–449. Same lesson for another hyped "alternative."

## Method lineage

- **One-sample / Welch t-test** on the annual excess return (sneaker − S&P) vs 0.
- **Newey-West (HAC) t-stat** (`lags=1`) — the honest standard error for a serially
  correlated annual series; this is the REAL-tape bar (|t| ≥ 2) for a Signal verdict.
- **Geltner AR(1) unsmoothing** to recover economic volatility and the honest Sharpe.
- **Lagged momentum overlay** with a one-year execution lag (no look-ahead) and a
  one-way frictional cost on the sneaker leg.

## Data sources

- **^GSPC (S&P 500 price index).** Repo-level cache `_cache/^GSPC_split_only.parquet`
  (yfinance origin), December-to-December last-trading-day price returns, 2007–2024.
  Cache-only by default; `fetch=True` lazily refreshes via yfinance (never in CI).
- **Sneaker-resale index.** Curated annual levels hardcoded in `data.py`, built to
  mirror the *shape* publicly reported by StockX/GOAT and the resale press (there is no
  clean public daily resale tape). The series is intentionally appraisal-style.

## Related desk studies

- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: folklore-table teardown that
  this study mirrors structurally (hardcoded series vs real market returns).
- **[Study 223 — Same-Month-Seasonality](../../223-same-month-seasonality/)**: the
  synthetic-panel + cached-real-tape pattern.
