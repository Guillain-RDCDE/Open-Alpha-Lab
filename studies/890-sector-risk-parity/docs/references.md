# References & literature map — Study 890 (Sector Risk-Parity)

## The claim under test

- **The recipe.** Cap-weight buries the S&P 500 in a handful of mega-cap sectors — as of 2026
  roughly a third of SPY is Information Technology. The risk-parity answer is to weight the
  eleven GICS sectors so each carries an **equal share of portfolio risk**: naively by
  **inverse volatility** (`w_i ∝ 1/σ_i`), or exactly by **equal risk contribution** (ERC),
  where each asset's marginal contribution to portfolio variance is equalised. The pitch is
  "All-Weather, but *within* equities": diversify the risk budget and the ride should be
  smoother at a better Sharpe, with shallower drawdowns.
- **The intellectual lineage.** Risk parity as an allocation principle traces to **Bridgewater's
  All-Weather** (Ray Dalio, 1996) and was popularised by **Edward Qian** ("Risk Parity
  Portfolios", PanAgora 2005), who coined the term and showed that in a cap- or dollar-weighted
  book a few high-vol assets dominate the *risk* even when they are a minority of the *capital*.
  **Maillard, Roncalli & Teïletche (2010)**, "The Properties of Equally-Weighted Risk
  Contribution Portfolios" (Journal of Portfolio Management), give the formal ERC construction
  and show it sits between minimum-variance and equal-weight in risk terms.
- **The honest question.** Unlevered, a risk-parity book is *expected* to earn **less** than a
  cap-weight equity index (it under-weights the crowded, high-vol, high-momentum sectors), so
  raw return is the wrong yardstick. The fair test — and the one this study runs — is the
  **excess-of-cash Sharpe** and the **drawdown**: does equalising risk across sectors actually
  buy a better risk-adjusted outcome, net of costs, or does it just deliver a lower-beta version
  of equities?

## What we measure, and the honesty rails

- **Total-return, excess of cash.** All prices are `auto_adjust=True` (dividends reinvested);
  every Sharpe is measured **excess of BIL** (both the sector book and SPY minus the cash leg),
  so the comparison is excess-vs-excess, not gross-vs-gross.
- **Point-in-time weights, one documented lag.** Weights on a rebalance date come from the
  trailing 63-day covariance **known at the close of the day before** (`lag = 1`); the book is
  then held with realistic daily drift until the next quarterly rebalance. Zero look-ahead.
- **Two panels, short history named.** Nine of the sector ETFs date from 1998, but **XLRE
  (2015-10)** and **XLC (2018-06)** are young, so the joint eleven-sector window is only ~8 years
  and dominated by the tech / AI bull. We therefore also run a **nine-sector** panel back to
  BIL's 2007 inception — the fairer, longer read — and cut it into eras. The short-history
  caveat is named on the **Signal** axis.
- **Robust inference.** A **paired circular-block bootstrap** on the *Sharpe difference* (RP −
  SPY) keeps the two legs' cross-correlation and the returns' serial dependence intact; a
  **Newey-West (HAC, 10-lag)** *t* on the mean daily excess-return difference cross-checks. An
  **era cut** tests whether any advantage holds across sub-periods.
- **The timer is graded separately.** Costs are one-way × turnover per quarterly rebalance; a
  **levered-to-SPY-vol** variant pays financing on the borrowed exposure — the honest test of
  whether a better-Sharpe-but-lower-return book can be turned into a bankable *return* edge.
- **The synthetic control proves the machinery only.** A seeded world with equal Sharpes but
  dispersed vols and a concentrated (∝ vol²) cap-weight benchmark; it confirms inverse-vol
  out-Sharpes cap-weight *iff* vols are dispersed, and never supports the real-tape stamp.

## Shared method citations

- **Qian, E. (2005)** — "Risk Parity Portfolios: Efficient Portfolios Through True
  Diversification" (PanAgora): the risk-contribution view of concentration.
- **Maillard, S., Roncalli, T. & Teïletche, J. (2010)** — the ERC portfolio and its properties.
- **Griveau-Billion, B., Richard, J.-C. & Roncalli, T. (2013)** — "A Fast Algorithm for
  Computing High-Dimensional Risk Parity Portfolios": the cyclical coordinate descent used by
  `erc_weights`.
- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* on the return-difference series).
- **Politis, D. & Romano, J. (1994)** — the stationary / circular block bootstrap behind the
  paired Sharpe-difference confidence interval.
- **Wilson, E. B. (1927)** — score interval for a binomial share.

## Data sources

- **yfinance daily closes** (`auto_adjust=True`, total-return): 11 SPDR Select-Sector ETFs +
  SPY + BIL, cached under `_cache/` as parquet. Eleven-sector window 2018-06 → 2026-06-30;
  nine-sector window 2007-05 → 2026-06-30.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [68-all-weather](../../68-all-weather/) — risk parity **across asset classes** (stocks /
  Treasuries / gold / commodities). This study applies the same inverse-vol / ERC machinery
  **within equities**, across the eleven GICS **sectors**, benchmarked to cap-weight SPY.
- [225-sector-rotation](../../225-sector-rotation/) — **timing / rotating** into sectors on a
  momentum or macro signal (a forecasting bet). This study **forecasts nothing**: it holds all
  eleven sectors always, only re-sizing them by risk.
- [28-carousel](../../28-carousel/) — a **rotation carousel** that cycles capital between
  sleeves. Again a timing bet, not a static risk-balanced hold.
- [94-level-pegging](../../94-level-pegging/) — **equal-weight** (1/N) as the antidote to
  cap-weight concentration. This study equalises **risk** (inverse-vol / ERC), not **dollars**;
  equal-weight is included here only as a reference leg.

None of the siblings equal-**risk**-weight the GICS sectors and race the result excess-of-cash
against cap-weight SPY — this study's own axis.
