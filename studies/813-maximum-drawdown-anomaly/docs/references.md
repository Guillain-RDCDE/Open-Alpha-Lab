# References & literature map — Study 813 (Maximum-Drawdown Anomaly)

## The claim under test

- **The drawdown-as-risk lineage.** Chekhlov, Uryasev & Zabarankin, *"Drawdown Measure in
  Portfolio Optimization"* (International Journal of Theoretical & Applied Finance, 2005),
  formalise **maximum drawdown** and conditional-drawdown-at-risk as coherent risk
  measures — the largest peak-to-trough decline of cumulative return as a stand-alone
  measure of pain. If drawdown is a priced risk, the deepest-drawdown names should command
  a **premium** (subsequently out-earn) or, under a distress reading, subsequently
  **under-earn**. This study asks which — sorting the cross-section on trailing MaxDD.
- **The distress-underperformance side.** Campbell, Hilscher & Szilagyi, *"In Search of
  Distress Risk"* (Journal of Finance, 2008), document that fundamentally distressed firms
  earn **anomalously low** returns — a distress *discount*, not premium. A deep trailing
  drawdown is a coarse, price-only distress proxy; the distress reading predicts the
  deep-drawdown names keep under-earning (our `spread = calm − distressed` > 0).
- **The reversal side.** De Bondt & Thaler, *"Does the Stock Market Overreact?"* (Journal
  of Finance, 1985) and Jegadeesh, *"Evidence of Predictable Behavior of Security Returns"*
  (Journal of Finance, 1990) document short- and long-horizon **reversal**: past losers
  rebound. A name in a deep drawdown is a recent loser; the reversal reading predicts it
  **out-earns** (our spread < 0) — which is what we in fact find on this universe.
- **Behavioural drawdown aversion.** Investors are documented to be **drawdown-averse**
  beyond variance (e.g. the peak-to-trough experience drives redemptions and de-grossing);
  a distressed name can be *sold to the point of a rebound*, the mechanism behind the
  reversal sign we observe.
- **The specific test here.** We take the self-contained daily version: sort a liquid US
  cross-section on its **trailing 252-day maximum drawdown** and measure the forward return
  of the equal-weight long-calm / short-distressed book, with a Newey-West *t*, a
  permutation placebo, a two-era robustness cut, a costed timer (both directions), and a
  seeded synthetic positive control that plants the *distress* sign.

## What we measure, and the honesty rails

- **Maximum drawdown, no free model.** For each name, the rolling `window`-day deepest
  peak-to-trough decline of the cumulative total-return price (running peak inside the
  window via `maximum.accumulate` over a sliding view), returned as a positive magnitude.
- **Point-in-time sort, one documented lag.** The ranking signal is the trailing MaxDD
  **known at the close of `t−1`** (`.shift(1)`); the book is held on day `t`. Zero
  look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short
  spread — an overlapping-window signal is serially correlated, so a plain *t* would
  overstate significance. A one-sample *t* and a pooled Welch *t* (calm book vs distressed
  book) cross-check. A **1,000-permutation placebo** breaks the signal → forward-return
  link. A **two-era cut** is the decisive robustness test — and it is where this signal
  fails, halving the case to Weak.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership**
  set of ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). The deepest drawdowns — permanent
  losers — are absent, so the distressed leg is a *survivor's* distress and the magnitudes
  are an upper bound.
- **The timer is graded separately.** Costs are one-way × NAV on the long-short book, and
  the short book pays borrow — the honest test of whether a small daily spread survives
  friction, tested in **both** sort directions.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Magdon-Ismail & Atiya (2004)** — the expected maximum drawdown of a Brownian path, the
  analytic backbone for reasoning about how vol and drift set drawdown depth.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [333-recovery-speed](../../333-recovery-speed/) — how **fast** a name climbs back out of a
  drawdown (a recovery-rate measure). This study sorts on the **depth** of the worst
  decline, not the speed of the bounce.
- [816-drawdown-duration](../../816-drawdown-duration/) — how **long** a name spends
  underwater (time-in-drawdown, the horizontal axis). This study is the **vertical** axis:
  the magnitude of the peak-to-trough fall, not its duration.
- [540-distress-risk](../../540-distress-risk/) — a **fundamental** default/distress score
  (Campbell-Hilscher-Szilagyi accounting + market inputs). This study uses a purely
  **price-based** trailing drawdown, no fundamentals.
- [332-downside-beta](../../332-downside-beta/) — a name's **beta in down markets** (a
  systematic co-movement with the market's declines). This study sorts on a name's **own**
  realized peak-to-trough drawdown, not its co-movement.

None of the siblings sort on the **depth of a name's own trailing 12-month maximum
drawdown** — this study's own axis.
