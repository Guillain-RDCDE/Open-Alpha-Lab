# References & literature map — Study 721 (Most-Admired)

## The claim under test

- **The folklore.** Fortune's **World's Most Admired Companies** (WMAC) are, by construction,
  the best-run firms on earth — top-scored by peers and analysts on management quality,
  products, innovation, and long-term value. The optimist infers an **admiration premium**:
  owning the list is owning quality, and quality compounds, so the admired basket should beat
  the market. Apple has topped the overall ranking **every year from 2008 through 2024**, with
  Amazon, Microsoft, Berkshire Hathaway, Disney, Alphabet, Starbucks, Nike, Costco and
  JPMorgan as perennial All-Stars.
- **The contrarian counter-claim.** A firm becomes *admired* only after a long run of good
  news and a soaring stock — so the label may mark a stock that is already loved, richly
  priced, and due to **mean-revert**. On this view the *spurned* (least-admired) out-earn the
  *admired*. We steelman and test **both** directions.

## The academic anchors — the premium *and* the reversal

- **Antunovich, Laster & Mishra (2000), *Are High-Quality Firms Also High-Quality
  Investments? The Long-Run Performance of Stocks Selected on the Basis of Fortune's Most
  Admired Companies List*, FRBNY Economic Policy Review 6(1).** The canonical "premium" paper:
  portfolios of the most-admired firms **out-performed** the least-admired (and the market)
  over 1983–1998, and the gap was not explained by risk. The optimist's academic anchor.
- **Anginer & Statman (2010), *Stocks of Admired and Spurned Companies*, Journal of Portfolio
  Management 36(3)** and **Statman, Fisher & Anginer (2008), *Affect in a Behavioral Asset-
  Pricing Model*, Financial Analysts Journal 64(2).** The "reversal" anchor: using Fortune's
  reputation scores they find the **spurned** (low-admiration) stocks *out-earned* the admired
  — admiration is an *affect* that gets over-priced, a cousin of the value effect. The
  literature is genuinely **split**, which is why this tape (not a citation) has to decide.
- **Shefrin & Statman (1995), *Making Sense of Beta, Size, and Book-to-Market*, JPM**, and
  **Solomon, Soltes & Sosyura (2014), *Winners in the spotlight: Media coverage of fund
  holdings as a driver of flows*, JFE** — the broader "good company ≠ good stock" and
  attention-reallocation literature the admiration effect sits inside.

## Why our tape is biased *for* the premium — and named on the Signal axis

- **Look-ahead selection (timing).** A *current* WMAC list is known only today; owning it back
  to 2004 is look-ahead. The **LAGGED** variant owns a name only from the February *after*
  Fortune first crowns it, removing the timing bias.
- **Roster survivorship.** Even lagged, the roster is the set of firms *still* admired in 2026
  — the winners that **stayed** on top. Firms admired in the 2000s that later fell off (and
  sometimes collapsed) are absent, biasing the admired book **up** (Brown, Goetzmann, Ibbotson
  & Ross, 1992, *Survivorship bias in performance studies*, RFS).
- **Spurned-leg survivorship (the mirror).** The genuine bottom of the reputation ranking
  **delisted** — Sears, Lehman Brothers, "old GM", Kodak, Enron, Blockbuster, AMR — leaving no
  clean series. A priced *spurned* book is therefore biased **up**, biasing admired−spurned
  **down**: a positive-but-insignificant long/short is a *conservative* refutation of the
  premium, not an endorsement.
- **Factor confound.** An equal-weight basket of mega-caps beats the **cap-weighted** `SPY`
  partly by a size-within-large-cap / rebalance tilt (our placebo measures ~+3.2%/yr of it) and
  carries a **beta > 1** — so a market-model **alpha** (Jensen, 1968) and a random-large-cap
  placebo are the honest yardsticks, not raw out-performance.

## Method lineage (the desk's shared engine)

- **Newey–West (HAC) inference.** [`strategy.newey_west_t`](../most_admired/strategy.py) and
  the HAC covariance in [`strategy.market_model_alpha`](../most_admired/strategy.py) implement
  the Bartlett-kernel long-run variance of Newey & West (1987), *A Simple, Positive Semi-
  Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica
  55(3), with the automatic lag of Newey & West (1994). Monthly equity returns are mildly
  autocorrelated and fat-tailed; an iid *t* overstates significance.
- **Market-model alpha.** [`strategy.market_model_alpha`](../most_admired/strategy.py) regresses
  the book on `SPY` and reports the beta-adjusted intercept with a HAC *t* — the honest
  "admiration alpha", net of the beta you were always paid for (Jensen 1968; MacKinlay 1997,
  *Event studies in economics and finance*, JEL).
- **Placebo / randomisation null.** [`strategy.placebo_pvalue`](../most_admired/strategy.py)
  draws random equal-weight large-cap books and asks how often chance matches the admired book's
  excess over `SPY` (Fisher's randomisation logic; Efron & Tibshirani, 1993, *An Introduction to
  the Bootstrap*).
- **Deterministic synthetic control.**
  [`data.synthetic_admired`](../most_admired/data.py) plants a known annual premium `edge`; with
  `edge=0` the HAC inference must NOT manufacture significance, and a large `edge` must light up.
  Runs offline.

## Data sources used here

- **yfinance** month-end adjusted closes for the 15 admired All-Stars + a 6-name survivor
  spurned proxy + a broad large-cap pool + `SPY`, 2004-01-31 → 2026-06-30 (as-of 2026-06-30),
  cached under `_cache/admired_prices.csv` (fingerprint `80c5c891a901`). The admired/spurned
  tables (tickers, first-crown years) are hardcoded in
  [`data.ADMIRED`](../most_admired/data.py) / [`data.SPURNED`](../most_admired/data.py);
  famously-delisted low-reputation firms are listed in [`data.DELISTED`](../most_admired/data.py)
  for the survivorship caveat. Headline numbers are pinned in [`docs/results.md`](results.md)
  and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 389 — Name-Change-Effect](../../389-name-change-effect/)**: the sibling label study
  — does rebranding toward the hot theme pay? Same family (a label, not a fundamental), same
  survivorship pathology.
- **[Study 391 — CEO-Turnover](../../391-ceo-turnover/)**: the adjacent corporate-event study —
  market-model CARs on a hardcoded, cited announcement table.
- **[Study 358 — Watch-Index](../../358-watch-index/)**: the desk's template for a **labelled
  proxy** series presented honestly, never under a real-tape banner.
