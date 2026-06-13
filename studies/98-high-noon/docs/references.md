# References & literature map — Study 98 (High-Noon)

## The claim under test

A piece of folk wisdom repeated by retail investors and a fair amount of market
commentary: *"Never buy at an all-time high — it's the riskiest moment, a classic sell
signal. Wait for a pullback."* The intuition is that a new high means the asset is
"expensive" / "overbought" / "due for a reversion", so buying there is buying the top.
We take the strong version literally and ask whether buying SPY *at or near an all-time
high* leads to **worse** forward returns than buying when **not** near a high.

- The fear is a fixture of retail psychology and headlines whenever an index prints a
  record ("is it safe to buy at all-time highs?").

## Why the steelman is almost coherent

- **Mean-reversion is real at some horizons and in some assets** — short-term reversal
  in single stocks (Jegadeesh 1990; Lehmann 1990) and valuation-driven reversion in
  expensive markets (Shiller's CAPE work). If highs systematically coincided with
  stretched valuations *and* those reverted quickly, the rule could carry information.
- **"Overbought" oscillators** (RSI, stochastics) institutionalise the same instinct: an
  extreme reading is read as a fade signal.

## Why it is likely to fail *as stated* ("ATHs are bearish")

- All-time highs are, by construction, **where an uptrend currently sits** — and equity
  index drift is positive, so highs cluster in exactly the regimes that keep trending.
  Several well-known practitioner studies make this point on data: **Meb Faber**'s work
  and the recurring *"what happens after all-time highs"* analyses (e.g. RBC GAM,
  JPMorgan Asset Management *Guide to the Markets*) consistently find forward returns
  after an ATH are **similar to or better than** unconditional forward returns, not worse.
- The **52-week-high momentum** literature points the *opposite* way to the folk rule:
  **George & Hwang (2004)**, *The 52-Week High and Momentum Investing*, JF — stocks
  nearest their 52-week high go on to **outperform**; nearness to a high is a *bullish*
  momentum signal, not a sell signal. Buying high and selling higher is the documented
  edge, not the trap.
- **Time-series momentum** (Moskowitz, Ooi & Pedersen, *Time Series Momentum*, JFE 2012)
  says recent strength persists — again the inverse of "fade the high".

## Method lineage

- **Newey–West HAC standard errors** for the mean of an autocorrelated (here, *overlapping*)
  forward-return series, and for the difference of the two conditional means: Newey & West
  (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation
  Consistent Covariance Matrix*, Econometrica. HAC lags are set to the forward horizon,
  the exact overlap length.
- **Wilson score interval** for the conditional win-rates: Wilson (1927), *Probable
  Inference, the Law of Succession, and Statistical Inference*, JASA — the standard
  small-and-large-n proportion CI used across the desk.

## Data sources used

- **SPY**, daily, **total-return adjusted** (dividends folded in) via `quantlab.data`
  (Yahoo Finance), cached to parquet under `_cache/`. Total return is the right tape both
  because forward returns should include dividends and because the *running all-time high*
  that matters to a compounding investor is the total-return high. The avoid-the-highs
  timer holds cash at **0%** — a stated, conservative choice that biases *toward* the
  folk rule (a T-bill on cash would only narrow the loss further).

## Related desk studies

- [Study 91 — Death-Cross](../../91-death-cross/) — the sibling "timing rule vs
  buy-and-hold" teardown and the same backtest engine.
- [Study 87 — Center-Line](../../87-center-line/) — another mean-reversion-folklore test
  (does price get pulled back to VWAP?).
- [Study 85 — Dr-Copper](../../85-dr-copper/) — a forecasting claim that turns out to echo,
  not predict.
