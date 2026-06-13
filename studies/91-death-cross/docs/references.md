# References & literature map — Study 91 (Death-Cross)

## The claim under test

The "death cross" is one of the most-quoted signals in financial media: when the
50-day simple moving average crosses **below** the 200-day SMA, it is read as the start
of a major downtrend — sell, or hedge — and the reverse "golden cross" is read as the
all-clear to re-enter. The strong, sold-at-full-strength version is that this simple
timer **dodges the big crashes and beats buy-and-hold**.

- Popular framing, e.g. Investopedia, *"Death Cross"* and *"Golden Cross"* definitions:
  <https://www.investopedia.com/terms/d/deathcross.asp>
- The signal is a fixture of CNBC / Bloomberg market commentary at every 50/200 cross.

## Why the steelman is almost coherent

- **Trend-following has a real, documented premium across assets** (Moskowitz, Ooi &
  Pedersen, *Time Series Momentum*, JFE 2012; Hurst, Ooi & Pedersen, *A Century of
  Evidence on Trend-Following Investing*, AQR 2017). A slow moving-average filter is a
  crude time-series-momentum rule, so it is not pure superstition.
- Moving-average timing **does** reliably reduce volatility and drawdown by cutting
  equity exposure during sustained declines (Faber, *A Quantitative Approach to Tactical
  Asset Allocation*, JWM 2007 — the 10-month SMA timer). The risk reduction is the part
  that survives.

## Why it is likely to fail *as stated* ("beats buy-and-hold")

- The risk reduction is mostly **lower average equity exposure** — i.e. lower beta — not
  forecasting skill. A like-for-like comparison must therefore ask: does the *timing*
  add value beyond just being in cash part of the time? (Zakamulin, *Market Timing with
  Moving Averages*, 2017, makes exactly this point and finds the live edge thin and
  regime-dependent.)
- The 50/200 cross is **slow** (it lags turning points by months), so in fast V-shaped
  drawdowns it sells low and buys back high — the whipsaw cost.
- Switching has **transaction and tax** costs that the headline backtest usually ignores.

## Method lineage

- **Newey–West HAC standard errors** for the mean of an autocorrelated return series:
  Newey & West (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica.
- **Matched random-timing control** — an exposure-matched placebo (same multiset of
  in/out run lengths, reshuffled) isolates whether the crossover *dates* carry
  information beyond the exposure profile. This is the desk's standard "beats a coin?"
  control (cf. the random-direction control in Study 87 — Center-Line).

## Data sources used

- **SPY**, daily, **total-return adjusted** (dividends folded in) via `quantlab.data`
  (Yahoo Finance), cached to parquet under `_cache/`. Total return is the fair benchmark
  for a strategy that sits in cash part of the time. Cash is assumed to earn **0%** — a
  stated, conservative choice that biases *against* the timer.

## Related desk studies

- [Study 87 — Center-Line](../../87-center-line/) — the "beats a coin?" control pattern.
- [Study 77 — Golden-Mean](../../77-golden-mean/) — another moving-average-lore teardown.
- [Study 68 — All-Weather](../../68-all-weather/) — risk reduction that *is* real and the
  bar a de-risking story has to clear.
