# References & literature map — Study 903 (Sector-Neutral Low-Vol)

## The claim under test

- **The anomaly.** Malcolm **Baker, Brendan Bradley & Jeffrey Wurgler**, *"Benchmarks as
  Limits to Arbitrage: Understanding the Low-Volatility Anomaly"* (Financial Analysts Journal,
  2011): low-risk stocks have historically delivered **higher risk-adjusted** returns than
  high-risk stocks — the security-market line is too flat — a puzzle they attribute to
  benchmark-relative institutional mandates that deter arbitrage of the effect.
- **The risk-parity reading.** Andrea **Frazzini & Lasse Pedersen**, *"Betting Against Beta"*
  (Journal of Financial Economics, 2014): a beta-neutral long-low-beta / short-high-beta book
  (BAB) earns a significant positive alpha, driven by leverage-constrained investors bidding up
  high-beta assets. Low volatility and low beta are close cousins in the cross-section.
- **The critique this study tests.** A large part of a naive low-**volatility** sort is a
  **sector** allocation: utilities, staples and health care are structurally calm, tech and
  energy structurally wild, so a bottom-vol / top-vol book is mechanically long-defensive /
  short-cyclical. If the "anomaly" is mostly that defensive-sector tilt, it is not a
  stock-level effect. See e.g. Asness, Frazzini & Pedersen on industry-adjusted (intra-industry)
  low-risk sorts, and the standard practice of running low-vol **within** GICS sectors.

## What we measure, and the honesty rails

- **Trailing volatility, no free model.** For each name, the rolling 63-day (≈one quarter)
  standard deviation of daily simple returns, known at the close of `t−1` (`.shift(1)`) and
  held on day `t` — one documented lag, zero look-ahead.
- **Sector-neutralisation by demean.** On each day we subtract, from every name's trailing
  vol, the **cross-sectional median vol of its own GICS sector**, then sort on the residual.
  The extreme-low and extreme-high residuals are drawn ~evenly across sectors, so the resulting
  long-short book is ~sector-neutral (verified: the raw long book is 45% defensive vs a
  neutralised −1.8% long-minus-short tilt). A raw (un-demeaned) variant is kept for comparison.
- **The risk-adjusted claim, on its own terms.** Because the anomaly is about return *per unit
  of risk*, we race each leg's own annualised **Sharpe** (≈ excess-of-cash on a daily book),
  not just the raw spread — the honest test of whether the calm leg is *better*, not merely
  *less volatile*.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short spread
  (an overlapping-formation signal is serially correlated, so a plain *t* overstates
  significance); a one-sample *t* and a pooled Welch *t* cross-check; a **1,000-permutation
  placebo** breaks the signal → forward-return link.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership** set of
  ~50 liquid mega-caps, pulled through the `quantlab.universe` guard (`allow_survivorship_bias=True`,
  an explicit opt-in). Delisted / de-rated names are absent, so magnitudes are an **upper bound**;
  the low-vol premium is also strongest in small/illiquid names, absent here by construction.
- **The timer is graded separately.** Costs are 2 sides × one-way × NAV per day on the
  long-short book, and the short (high-vol) book pays borrow — the honest test of whether a
  small daily spread survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Asness, C., Frazzini, A. & Pedersen, L. (2014)** — *"Low-Risk Investing Without Industry
  Bets"*: the same intra-industry (sector-neutral) low-risk sort, the direct methodological
  parent of this study.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- Fixed GICS-style sector labels for the current-membership universe (`data.SECTORS`).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [330-low-volatility-anomaly](../../330-low-volatility-anomaly/) — the **full-strength** low-vol
  anomaly in its retail ETF embodiment (SPLV vs SPHB), with **no** sector control. This study is
  precisely the follow-up: does the edge survive once the defensive-**sector** bet inside a
  low-vol sort is stripped out?
- [501-idiosyncratic-volatility](../../501-idiosyncratic-volatility/) — sorts on **residual**
  (idiosyncratic) volatility after a factor model, the Ang-Hodrick-Xing-Zhang effect. This study
  sorts on **total** trailing volatility and neutralises the **sector**, not a factor residual.
- [58-bunker](../../58-bunker/) — the **USMV** min-volatility ETF as a defensive holding; a
  single fund's realised experience, not a cross-sectional low-minus-high sort or a sector-neutral
  decomposition.
- [246-defensive-sectors](../../246-defensive-sectors/) — a **sector-timing** canary (XLP+XLU
  relative strength forecasting SPY). This study does the opposite: it *removes* the sector bet to
  isolate the stock-level low-vol effect, rather than trading the sectors themselves.

None of the siblings run a **within-sector (sector-neutral) trailing-volatility sort** to ask
whether the low-vol edge is a stock-level effect or a sector tilt — this study's own axis.
