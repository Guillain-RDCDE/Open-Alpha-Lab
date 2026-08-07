# References & literature map — Study 816 (Drawdown Duration)

## The claim under test

- **The idea.** A drawdown has two moments: its **depth** (how far below the peak a name
  fell) and its **duration** (how *long* it stayed there). This study measures the
  duration side directly as **time-underwater** — the fraction of the trailing year a
  name's cumulative total return sat **below its running high-water mark**. A high
  time-underwater is *persistent-drawdown* risk: the name has been a laggard, rarely
  ratcheting a new high. The cross-sectional question is whether the market **pays** a
  premium for holding such names (a positive long-high / short-low spread — the
  distressed-risk / rebound reading) or whether they simply **keep sinking** (a negative
  spread — the low-quality / momentum reading). We report the honest sign.
- **The academic anchors.** Time-underwater is a path statistic of the drawdown process.
  **Burghardt, Duncan & Liu (2003)**, *"Deciphering Drawdowns"*, and **Magdon-Ismail &
  Atiya (2004)**, *"Maximum Drawdown"* (Risk), formalise drawdown depth *and* the
  time-to-recover / time-underwater as distinct risk dimensions. On the cross-sectional
  pricing side the signal is a cousin of two documented effects: **George & Hwang (2004)**,
  *"The 52-Week High and Momentum Investing"* (Journal of Finance) — names near their
  52-week high (i.e. *low* time-underwater) earn higher subsequent returns — and the
  distress-risk literature (**Campbell, Hilscher & Szilagyi 2008**, *"In Search of
  Distress Risk"*, Journal of Finance), where deeply and persistently drawn-down names
  historically *under*-earn rather than command a premium.
- **The behavioural / risk reading.** If time-underwater proxies distress or
  low-quality, the "market pays for risk" premium fails and the names keep sinking
  (a George-Hwang / distress tilt). If instead it proxies temporarily-depressed but
  sound names, a rebound premium could appear. On liquid mega-caps we find **neither** at
  any significance.
- **The specific test here.** Sort a liquid US cross-section on its **trailing-252-day
  time-underwater** and measure the forward return of the equal-weight long-high /
  short-low book, with a Newey-West *t*, a two-sided permutation placebo, a two-era
  robustness cut, a costed timer, and a seeded synthetic positive control.

## What we measure, and the honesty rails

- **Time-underwater, no free model.** For each name, the cumulative total-return curve
  `cumprod(1+r)`, its expanding **high-water mark** `cummax`, the 0/1 underwater
  indicator `curve < HWM`, and the rolling `window`-day mean of that indicator — all
  vectorised (column-wise `cumprod`/`cummax` + a pandas rolling mean).
- **Point-in-time sort, one documented lag.** The ranking signal is the time-underwater
  **known at the close of `t-1`** (`.shift(1)`); the book is held on day `t`. Zero
  look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short
  spread — a 252-day overlapping-formation signal is strongly serially correlated, so a
  plain *t* would badly overstate significance. A one-sample *t* and a pooled Welch *t*
  (high-UW book vs low-UW book) cross-check. A **1,000-permutation two-sided placebo**
  breaks the signal → forward-return link (the edge could be either sign, so we do not
  privilege a direction).
- **Survivorship is named on the Signal axis.** The universe is a **current-membership**
  set of ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). The names that stayed underwater
  and died are absent, so any "losers keep sinking" tilt is *understated*.
- **The timer is graded separately.** Costs are one-way × NAV on the long-short book, and
  the short book pays borrow — the honest test of whether a small daily spread survives
  friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [813-max-drawdown](../../813-max-drawdown/) — the **depth** of the worst peak-to-trough
  fall (how *far* under water a name went). This study measures the **duration** — the
  *fraction of time* spent below the high-water mark — an orthogonal moment of the same
  drawdown curve. A name can have a deep-but-brief crash (high depth, low duration) or a
  shallow-but-grinding decline (low depth, high duration).
- [333-recovery-speed](../../333-recovery-speed/) — how **fast** a name climbs back to a
  new high *after* a drawdown (the slope of the recovery leg), not the *share of time*
  spent below the high-water mark across the whole trailing year.
- [330-low-volatility](../../330-low-volatility/) — the **volatility** low-risk anomaly.
  Time-underwater is drift/vol-driven and so is *loosely* correlated with volatility, but
  it is a path-dependent drawdown statistic (a function of the whole price path relative
  to its running max), not the return standard deviation.

None of the siblings sort on the **fraction of the trailing year a name spent below its
high-water mark** — the drawdown-*duration* axis — which is this study's own signal.
