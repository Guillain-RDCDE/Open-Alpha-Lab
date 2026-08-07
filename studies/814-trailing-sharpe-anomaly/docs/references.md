# References & literature map — Study 814 (Trailing-Sharpe Anomaly)

## The claim under test

- **The source idea — risk-adjusted momentum.** Svetlozar **Rachev, Teo Jašić, Stoyan
  Stoyanov & Frank Fabozzi**, *"Momentum strategies based on reward-risk stock selection
  criteria"* (Journal of Banking & Finance, 2007), and the companion **Biglova, Ortobelli,
  Rachev & Stoyanov** (2004) work, replace the raw past-return ranking of Jegadeesh-Titman
  momentum with a **reward-to-risk** ranking. In its simplest form that reward-to-risk
  score is the stock's **trailing Sharpe ratio** — mean of daily returns divided by their
  standard deviation over the formation window. The pitch: risk-adjusting should throw out
  the high-volatility "lottery" winners whose momentum is fragile, keeping a cleaner,
  higher-quality winner book.
- **The Jegadeesh-Titman skeleton.** **Jegadeesh, N. & Titman, S. (1993)**, *"Returns to
  Buying Winners and Selling Losers"* (Journal of Finance), define the **12-1** convention
  we inherit: rank on the formation window while **skipping the most recent month** to
  sidestep the short-term reversal. This study skips the same month in the Sharpe estimate.
- **The honesty question this study is built around.** A Sharpe ratio is *momentum
  numerator ÷ volatility denominator*. So a Sharpe sort is mechanically a blend of a
  **momentum** signal and an inverse-**volatility** (low-vol) signal. The real test is not
  "does the Sharpe book make money" but **"does risk-adjusting add anything over plain 12-1
  momentum, or is it just momentum + low-vol repackaged?"** — which we answer by running
  all three sorts head-to-head and reporting their cross-sectional rank overlap.

## What we measure, and the honesty rails

- **Trailing 12-1 Sharpe, no free model.** For each name, the rolling `lookback`-day
  (≈252, 12m) mean and population std (ddof=0) of daily simple returns, formed on the
  window **ending `skip` (≈21, 1m) days ago** — value on row `t` uses returns through
  `t-skip`, skipping the most recent month exactly as 12-1 momentum does.
- **Point-in-time sort, one documented lag.** The ranking signal is the Sharpe **known at
  the close of `t-1`** (`.shift(1)` on top of the skip); the book is held on day `t`. Zero
  look-ahead. Long the **top** 30% (high Sharpe), short the **bottom** 30%, equal weight.
- **The comparator IS the point.** We run plain **12-1 momentum** (cumulative
  formation-window return) and a pure **low-vol** sort on the identical universe, dates,
  and machinery, and report the average per-day Spearman rank correlation between the
  Sharpe signal and each — the direct measure of "repackaging".
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short
  spread — an overlapping-formation signal is serially correlated, so a plain *t* would
  overstate significance. A one-sample *t* and a pooled Welch *t* (high book vs low book)
  cross-check. A **1,000-permutation placebo** breaks the signal → forward-return link.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership**
  set of ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are
  absent, so the cross-sectional magnitudes are an **upper bound**.
- **The timer is graded separately.** Costs are 2 sides × one-way × NAV per day on the
  long-short book, and the short book pays borrow — the honest test of whether a small
  daily spread survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Sharpe, W. F. (1966, 1994)** — the reward-to-variability ratio itself, here estimated
  *ex-post* from the trailing daily return tape rather than forecast.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [507-cross-sectional-momentum](../../507-cross-sectional-momentum/) — **plain 12-1
  price momentum**, no risk adjustment. This study divides that same signal by realized
  volatility and grades it *against* 507's book directly (they turn out ~0.95
  rank-correlated on this universe).
- [8-true-strength](../../8-true-strength/) — a smoothed **trend / oscillator** strength
  indicator, not a mean-over-std moment ratio of the raw return distribution.
- [330-low-volatility-anomaly](../../330-low-volatility-anomaly/) — sorting on
  **volatility alone** (the Sharpe *denominator*). A Sharpe sort couples that denominator
  to a momentum numerator; we decompose which leg does the work (here: the numerator).
- [237-residual-momentum](../../237-residual-momentum/) — momentum in **factor-model
  residuals** (idiosyncratic return). This study risk-scales **total** return momentum by
  its own realized volatility, not a residual.

None of the siblings rank on **total-return momentum divided by its own realized
volatility** — the reward-to-risk / trailing-Sharpe signal — which is this study's own
axis, tested precisely to see whether that division *earns its keep*.
