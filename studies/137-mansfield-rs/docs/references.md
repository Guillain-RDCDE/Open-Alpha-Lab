# References & literature map — Study 137 (Mansfield-RS)

## The claim under test

**Stan Weinstein's Stage Analysis**, from *Secrets for Profiting in Bull and Bear
Markets* (1988, McGraw-Hill): stocks move through four stages — base (Stage 1),
advance (Stage 2), top (Stage 3), decline (Stage 4).  The actionable rule is: buy when
a stock enters Stage 2 — price crosses above and holds above a rising 30-week SMA —
provided the **Mansfield Relative Strength** (the stock's price-to-30w-SMA ratio
divided by the benchmark's ratio) is positive, confirming the stock is outperforming the
market.  The joint condition is meant to filter momentum stocks from both the trending
and relative-strength dimensions, reducing exposure to "false breakouts."

## Why the steelman is coherent

- **Price momentum is real** (in cross-section at the monthly horizon): Jegadeesh &
  Titman (1993), *Returns to Buying Winners and Selling Losers* (Journal of Finance).
  Stocks with 12-1 month prior returns tend to outperform over the next six months.
  Stage 2 is, in part, a trend filter selecting stocks with positive intermediate-term
  momentum — and that effect is documented.
- **Relative strength (RS) momentum also has empirical support**: Blitz & Van Vliet
  (2008), *Global Tactical Cross-Asset Allocation* (Journal of Portfolio Management);
  and at the stock level, Grundy & Martin (2001), *Understanding the Nature of the
  Risks and the Source of the Rewards to Momentum Investing* (Review of Financial
  Studies).  Stocks that have been outperforming their sector or benchmark tend to
  continue outperforming in the short-to-medium term.
- **Moving-average-based trend filters have a long academic history**: Brock,
  Lakonishok & LeBaron (1992), *Simple Technical Trading Rules and the Stochastic
  Properties of Stock Returns* (Journal of Finance), found predictive power in MA rules
  on the DJIA 1897–1986 — though Park & Irwin (2007), *What Do We Know About the
  Profitability of Technical Analysis?* (Journal of Economic Surveys), document how
  much evaporates out-of-sample and after costs.
- **The 30-week SMA as a trend filter**: Grinold & Kahn (1999), *Active Portfolio
  Management* (McGraw-Hill), discuss moving-average signals as proxies for
  regime-detection; the 30-week (≈ 150-day) MA is a well-established medium-term
  trend indicator.

## Why the steelman breaks down in this test

- **Transition-based filters capture late momentum**: By requiring a *transition* from
  Stage 1 to Stage 2 (price crossing above a *rising* 30-week SMA with positive RS),
  the filter enters stocks *after* the initial breakout — by construction, it waits for
  the trend to be confirmed.  McLean & Pontiff (2016), *Does Academic Research Destroy
  Stock Return Predictability?* (Journal of Finance), show that momentum effects decay
  as they become widely known.  More fundamentally, the filter buys the confirmation,
  not the signal.
- **The random-entry baseline beats Stage-2 entries**: Our test shows Stage-2 entries
  earn −88.7 bps less than random entries at a 13-week horizon.  This is consistent with
  Asness, Moskowitz & Pedersen (2013), *Value and Momentum Everywhere* (Journal of
  Finance), who show momentum pays at the *12-month* horizon and mean-reverts at very
  short horizons — but the *transition* into Stage 2, occurring mid-trend, captures
  neither the beginning nor a fresh impulse.
- **Survivorship bias inflates the measured Stage-2 excess**: Using only current S&P
  500 constituents (AAPL, AMZN, etc.) overstates the strategy's returns; Shumway &
  Warther (1999), *The Delisting Bias in CRSP's Nasdaq Data* (Journal of Finance), and
  Elton, Gruber & Blake (1996) document how survivorship-biased samples inflate
  apparent strategy returns.  Our AAPL and AMZN t-stats would disappear in a
  full-universe test.
- **Beta, not timing**: Stage-2 stocks are by definition high-momentum stocks with
  positive RS; they tend to be higher-beta names in an up-market.  The apparent excess
  return may be beta, not alpha.  Fama & French (1993), *Common Risk Factors in the
  Returns on Stocks and Bonds* (Journal of Financial Economics), provide the framework
  for distinguishing systematic risk exposure from true alpha.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat**: Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*
  (Econometrica) — implemented in [`strategy.summarize`](../mansfield_rs/strategy.py).
- **Relative-strength normalisation**: The Mansfield RS formula (dividing by the
  30-week SMA of both stock and benchmark) is described in Weinstein (1988), Chapter 5.
  A simpler RS ratio (stock / benchmark) is used in IBD's Relative Strength Rating and
  in Levy (1968), *Relative Strength as a Criterion for Investment Selection* (Journal
  of Finance).
- **Synthetic positive control with AR(1) RS momentum**: Verifying that the engine
  recovers an edge when RS momentum is planted, following the desk's discipline from
  [Study 72 — Loaded-Dice](../../72-loaded-dice/) and [Study 106 — Supertrend](../../106-supertrend/).

## Related desk studies

- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the daily-bar 50/200 SMA golden
  cross — the same MA-trend family; also a NONE signal.
- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the 5-minute SMA(5/10) crossover
  scalp — foundational template for this study's engine; Signal NONE.
- **[Study 106 — Supertrend](../../106-supertrend/)**: Supertrend (ATR-band trend
  indicator) — Signal REAL but Tradability FRAGILE; confirms that some trend-following
  signals can be real on daily bars while short-term and composite signals are not.
- **[Study 78 — Crossed-Wires](../../78-crossed-wires/)**: MACD signal on daily bars —
  same moving-average family; Signal NONE.

## Data sources used here

- **Yahoo Finance weekly bars** (via `yfinance`, resampled from daily to weekly
  Friday-close OHLCV), for SPY and 15 large-cap US equities.  25 years of weekly data
  (~1,275 usable bars per ticker after the 30-week SMA warmup).  The offline core and
  tests run on a deterministic synthetic generator, never the network.  Results pinned
  with per-tape content fingerprints (see [`docs/results.md`](results.md)).
