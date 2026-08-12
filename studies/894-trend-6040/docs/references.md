# References & literature map — Study 894 (Trend Overlay on 60/40)

## The rule under test

- **The balanced book.** The **60/40** portfolio (60% equities / 40% intermediate
  Treasuries) is the default institutional and retail allocation — one asset for growth,
  one for ballast. Its Achilles heel is a joint drawdown: 2008 (equities −37%) and 2022
  (stocks *and* bonds down together) are the episodes it is built to survive and doesn't.
- **The trend overlay.** A well-worn tactical idea (Faber's *"A Quantitative Approach to
  Tactical Asset Allocation"*, 2007/2013): hold a risky asset only while its price is above
  its long moving average (Faber uses the 10-month / ~200-day SMA), step to cash when it
  falls below. Applied to a *multi-asset* book, the natural extension is to trend-filter
  **each leg independently** — the equity sleeve on SPY's 200-day MA, the bond sleeve on
  IEF's — so the book de-risks whichever leg has rolled over. The pitch is a "free lunch":
  keep most of the 60/40's return while dodging its worst drawdowns.
- **The specific test here.** SPY (equity) and IEF (7-10y Treasuries), 60/40 target
  weights, each leg in its asset when above its **200-day SMA** and in **BIL** (T-bills)
  otherwise; signal known at the close of `t−1` (one `shift`, zero look-ahead), acted on at
  `t`. We grade it **excess-of-cash vs excess-of-cash** against the static 60/40, with a HAC
  *t* on the return difference, a **paired block-bootstrap** CI for the Sharpe advantage, a
  two-era cut, a calendar table, a switching-cost grid, a short-term-gains **tax drag**, and
  a 12-seed synthetic positive control.

## What we measure, and the honesty rails

- **Excess vs excess.** Both the overlay and the static book are taken **minus the BIL cash
  leg** before any Sharpe is formed, so the race is not flattered by the 2007-08 and 2022-26
  periods when cash itself paid 4-5%. The Sharpe advantage is what remains.
- **One documented lag, warm-up dropped.** The 200-day signal uses closes through `t−1` and
  is acted on at `t`. The MA warm-up is **NaN**, not a silent cash default, and both arms
  start only once the MA is defined — so the overlay is never handed a free 200 days in cash
  while the static book is invested.
- **Robust inference.** A Newey-West (HAC, Bartlett, 10-lag) *t* on the daily return
  difference (an overlay signal is serially correlated). A **paired** circular-block
  bootstrap gives a CI for the *Sharpe advantage* itself — the honest test of whether a
  risk-adjusted edge is distinguishable from zero, which a single point estimate hides.
- **Costs and tax are graded separately.** Switching costs are one-way × the NAV fraction
  that flips per rebalance (a 200-day rule trades rarely, so these are small). The **tax
  drag** models the short-term gain *realised* every time a leg is forced to cash — a
  friction a buy-and-hold 60/40 defers, and the one that actually decides the verdict.
- **Short history is named on the Signal axis.** BIL (the cash leg) launches 2007-05, so the
  book spans ~19 years with essentially one deep equity bear (2008) and one bond bear (2022).
  A trend rule looks best on a single-crash sample; the era cut is the guard.

## Shared method citations

- **Faber, M. (2007/2013)** — *"A Quantitative Approach to Tactical Asset Allocation"* — the
  10-month / 200-day moving-average timing rule this study overlays on the balanced book.
- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  (HAC) covariance; the *t* on the daily return difference.
- **Politis, D. & Romano, J. (1994)** — the stationary / circular block bootstrap used for
  the Sharpe-advantage confidence interval on serially dependent daily returns.
- **Lo, A. (2002)** — *"The Statistics of Sharpe Ratios"* — why a Sharpe needs a standard
  error, and how serial correlation inflates the naive annualisation.
- **Wilson, E. B. (1927)** — score interval for a binomial share (the primitives library).

## Data sources

- **yfinance daily total-return Close** (`auto_adjust=True`), SPY / IEF / AGG / BIL,
  2007-05-30 → 2026-06-30, cached under this study's own `_cache/`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py). As-of **2026-06-30**, fingerprint
  `0dd2af7e1636`.

## Related desk studies (the dedup map — what this study is NOT)

- [110-faber-timing](../../110-faber-timing/) — the **single-asset** Faber rule on the
  equity index (SPY / ^GSPC in vs out of cash). This study overlays the *same* 200-day
  filter on a **two-leg 60/40 book**, timing the equity *and* the bond sleeve independently,
  and grades it against the static balanced book — not buy-and-hold equity.
- [97-balancing-act](../../97-balancing-act/) — the **static** 60/40 itself (does the
  fixed balanced book earn its keep vs its pieces). This study asks a different question:
  does *adding a trend overlay* to that static book improve it.
- [592-dual-momentum-gem](../../592-dual-momentum-gem/) — Antonacci's **dual momentum**:
  a *relative* + absolute momentum rotation *between* assets. This study is an *absolute*
  trend filter (price vs its own MA) applied *within* fixed 60/40 weights, with no rotation.
- [626-unemployment-trend-timing](../../626-unemployment-trend-timing/) — timing equities on
  a **macro (unemployment) trend**. This study times each leg on its **own price** trend, a
  purely technical 200-day MA, with no macro input.

None of the siblings lay a **per-leg 200-day price trend filter over the static 60/40 book**
and grade the drawdown/Sharpe trade-off excess-vs-excess net of switching costs and tax —
this study's own axis.
