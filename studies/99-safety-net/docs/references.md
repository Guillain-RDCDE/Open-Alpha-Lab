# References & literature map — Study 99 (Safety-Net)

## The claim under test

The single most ubiquitous piece of retail trading advice: *"Always use a stop-loss.
**Cut your losses and let your winners run.** A **trailing stop** — exit when price falls
X% from its peak — protects your capital, limits your drawdown, and improves your returns."*
The strong, sold-at-full-strength version is that disciplined stop-loss use is *free*
risk reduction that also *raises* long-run return.

- Popular framing, e.g. Investopedia, *"Trailing Stop"* and *"Stop-Loss Order"*
  definitions: <https://www.investopedia.com/terms/t/trailingstop.asp> and
  <https://www.investopedia.com/terms/s/stop-lossorder.asp>
- The maxim *"cut your losses short and let your profits run"* is attributed to David
  Ricardo and is a fixture of nearly every introductory trading text and broker tutorial.

## The academic nuance — why the steelman is *conditionally* coherent

- **Kaminski & Lo (2014), *When Do Stop-Loss Rules Stop Losses?*, Journal of Financial
  Markets 18, 234–254.** The pivotal paper. Stops add value precisely when returns are
  **positively autocorrelated / trending** (the stop escapes a persisting downtrend), and
  they *subtract* value when returns are **mean-reverting** (the stop sells the dip and
  misses the rebound — pure whipsaw). The sign of the stopping premium is the sign of the
  serial correlation. Our trending vs mean-reverting synthetic tapes are built to reproduce
  exactly this dichotomy.
- **Trend-following has a real, documented premium across assets** (Moskowitz, Ooi &
  Pedersen, *Time Series Momentum*, JFE 2012; Hurst, Ooi & Pedersen, *A Century of Evidence
  on Trend-Following Investing*, AQR 2017) — a trailing stop is a crude one-sided
  trend filter, so it is not pure superstition where trends persist.
- **Moving-average / drawdown timing reliably reduces volatility and drawdown by cutting
  equity exposure** (Faber, *A Quantitative Approach to Tactical Asset Allocation*, JWM
  2007). The risk reduction is the part that survives; the return claim is the part that
  usually does not.

## Why it is likely to fail *as stated* ("improves returns") on a long-biased index

- A broad equity index like the S&P 500 is **long-biased with substantial short-horizon
  mean reversion** (the V-shaped 2018-Q4, 2020-COVID and 2025 drawdowns snapped back in
  weeks). On exactly this kind of tape, Kaminski-Lo predict stops should *reduce drawdown
  somewhat but underperform on return* — selling the dip, missing the rebound.
- The risk reduction is mostly **lower average equity exposure** — i.e. lower beta — not
  forecasting skill. A like-for-like test must ask whether the stop's *timing* beats an
  exposure-matched random exit, not merely whether being in cash sometimes lowers vol.
- A **too-tight** stop fires on noise and whipsaws catastrophically (our 5% result);
  switching also carries **transaction and tax** costs the headline maxim ignores.

## Method lineage

- **Newey–West HAC standard errors** for the mean of an autocorrelated return-difference
  series: Newey & West (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica 55(3).
- **Matched random-exit control** — an exposure-matched placebo (same multiset of in/out
  run lengths, reshuffled to random dates) isolates whether the stop's *exit dates* carry
  information beyond the exposure profile. This is the desk's standard "beats a coin?"
  control, mirroring the matched random-timing coin of [Study 91 — Death-Cross](../../91-death-cross/).

## Data sources used

- **SPY**, daily, **total-return adjusted** (dividends folded in) via `quantlab.data`
  (Yahoo Finance), cached to parquet under `_cache/`. Total return is the fair benchmark
  for a strategy that sits in cash part of the time. Cash is assumed to earn **0%** — a
  stated, conservative choice that biases *against* the stop (giving it the T-bill while in
  cash would only flatter it).

## Related desk studies

- [Study 91 — Death-Cross](../../91-death-cross/) — the matched-random "beats a coin?" control
  and the same alpha-vs-beta read on a moving-average timer.
- [Study 87 — Center-Line](../../87-center-line/) — the "beats a coin?" control pattern.
- [Study 68 — All-Weather](../../68-all-weather/) — risk reduction that *is* real and the bar
  a de-risking story has to clear.
