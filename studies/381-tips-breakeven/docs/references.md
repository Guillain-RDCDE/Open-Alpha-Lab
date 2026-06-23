# References & literature map — Study 381 (TIPS-Breakeven)

## The claim under test

- **Breakeven inflation as a forecast.** The **breakeven inflation rate** is the difference
  between the yield on a nominal Treasury and the yield on a matched-maturity Treasury
  Inflation-Protected Security (TIPS); it is the market's implied forecast of average inflation
  over the bond's life. FRED publishes the 10-year series as **`T10YIE`** (10-Year Breakeven
  Inflation Rate, Federal Reserve Bank of St. Louis). The folklore tested here: that this
  market-implied forecast is a **tradable macro timing signal** — high/rising breakeven says
  buy inflation hedges (gold, TIPS) and lean out of equities; low/falling breakeven says the
  reverse.
- **What breakeven actually contains.** Breakeven is *expected inflation + an inflation risk
  premium − a TIPS liquidity premium*. Decompositions: **Gürkaynak, Sack & Wright (2010)**,
  *The TIPS Yield Curve and Inflation Compensation* (American Economic Journal: Macroeconomics);
  **D'Amico, Kim & Wei (2018)**, *Tips from TIPS* (Journal of Financial and Quantitative
  Analysis); **Christensen, Lopez & Rudebusch (2010)** on inflation expectations from TIPS. The
  premia mean breakeven is a *noisy* read on expected inflation — already a warning against
  treating it as a clean signal.

## Why true breakeven is not on yfinance — and what we do instead

- **FRED access.** The canonical input (`T10YIE`, plus the nominal/real constituents `DGS10`,
  `DFII10`) lives on FRED. In this environment the FRED CSV endpoint is **not reliably
  reachable** (network-restricted), so we **construct a transparent proxy** from yfinance ETFs:
  `be = log(TIP / IEF)`, where `TIP` is the iShares TIPS Bond ETF and `IEF` is the iShares
  7–10 Year Treasury ETF (a duration-matched nominal leg). When the market raises its inflation
  forecast, the inflation-protected leg outperforms the nominal leg, so `log(TIP/IEF)` rises
  with breakeven. This is a *return-space* analogue of the nominal-minus-real yield, not the
  literal level — a methodological choice, named throughout, with every input a public adjusted
  close.
- **The proxy's built-in asterisk.** Because the proxy is built from TIP, asking it to predict
  forward *TIP* returns partly predicts the signal's own mean reversion. We flag this on the
  Signal axis and weight the genuinely independent targets (equities, gold).

## Why a regression-scan of a macro variable is a multiple-testing trap

- **Predictive regressions & overlap.** Forward multi-month returns overlap, inducing serial
  correlation that inflates naive OLS *t*-stats. We use a **Newey-West HAC** standard error
  (**Newey & West, 1987**, *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica) with Bartlett kernel and
  bandwidth set to the horizon — the standard fix.
- **Small-sample predictability is fragile.** **Stambaugh (1999)**, *Predictive regressions*
  (Journal of Financial Economics) and **Goyal & Welch (2008)**, *A Comprehensive Look at the
  Empirical Performance of Equity Premium Prediction* (Review of Financial Studies) show that
  in-sample predictive slopes on persistent macro regressors routinely fail out-of-sample — a
  direct precedent for the gold "hit" that evaporates as a trade here.
- **Multiple testing / data mining.** Scanning two signals × three assets × three horizons is
  18 tests; ~0.9 cross `|t| ≥ 2` by chance at the 5% level. **Harvey, Liu & Zhu (2016)**,
  *…and the Cross-Section of Expected Returns* (Review of Financial Studies) and **Bailey,
  Borwein, López de Prado & Zhu (2014)**, *Pseudo-Mathematics and Financial Charlatanism*
  (Notices of the AMS) formalise why a single surviving cell from a scan needs a far higher bar.
  We back the bar with a **placebo / randomization null** (circularly shuffle the signal,
  recompute the HAC *t*) — Fisher's randomization logic; **Efron & Tibshirani (1993)**, *An
  Introduction to the Bootstrap*.

## Method lineage (the desk's shared engine)

- **HAC predictive regression + placebo.** [`strategy.hac_regression`](../tips_breakeven/strategy.py)
  and [`strategy.placebo_pvalue`](../tips_breakeven/strategy.py) — the Signal-axis tests:
  Newey-West *t* of the slope and a shuffle null sized to the sample.
- **Deterministic synthetic control.**
  [`data.synthetic_macro`](../tips_breakeven/data.py) plants a known slope of next-month return
  on the breakeven z-score; the offline core runs with no network. The control confirms the
  detector is unbiased (`edge=0 ⇒ t≈0`) and powered (`edge>0 ⇒ |t|≫2`), so the real-tape null
  is a true null.
- **Traded sign rule with execution lag + costs.**
  [`strategy.sign_timer`](../tips_breakeven/strategy.py) enters one month after the signal (no
  look-ahead) and charges 10 bps one-way × turnover — the Tradability-axis test that turns the
  "significant" gold regression into a money-loser.

## Data sources used here

- **yfinance** daily adjusted closes for `TIP`, `IEF`, `SPY`, `GLD`, 2004-11-18 → 2026-06-18
  (common window), cached under `_cache/proxy_prices.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 119 — Real-Rate-Regime](../119-real-rate-regime/)**: the sibling rates signal — the
  *real* yield (TIPS yield) as a regime/timing variable. Same family, same multiple-testing
  hazard.
- **[Study 118 — Fed-Model](../118-fed-model/)** and
  **[Study 120 — Excess-CAPE-Yield](../120-excess-cape-yield/)**: regress-a-macro/valuation-
  variable-against-forward-returns studies — the canonical place where in-sample slopes look
  real and out-of-sample edges vanish.
- **[Study 152 — Inflation-Hedge](../152-inflation-hedge/)**: the asset-side question (do gold
  and TIPS actually hedge inflation?), the natural complement to "does breakeven time them?"
