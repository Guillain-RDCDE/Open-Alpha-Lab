# Study 223 -- Same-Month Seasonality

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Top-minus-bottom decile spread HAC *t* = **+5.57** across 330 months; block-bootstrap Sharpe 95% CI = [+0.56, +1.20], fully above zero; signal stable across all three sub-periods (t = 3.92 / 3.65 / 2.15). Survivorship-biased upper bound on ~8 stocks per decile. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Monthly turnover ~78% on 8-stock deciles; short leg requires hard-to-borrow distressed names; a live Russell-1000 implementation (100 per decile) would have a materially lower gross spread; gross edge dominated by survivorship inflation and extreme concentration. |
| **Survivorship bias** | ![Named](https://img.shields.io/badge/Survivorship--biased-8b949e?style=flat-square) | Universe = current large-cap survivors projected backwards; true effect on a delisting-inclusive panel is substantially weaker. |

> **In one sentence:** the Heston-Sadka same-month seasonality shows a statistically robust spread (*t* = 5.57) on the survivorship-biased large-cap panel, but the effect is inflated by extreme concentration (~8 stocks per decile), ~78% monthly turnover, and the fact that only July and December deliver t-stats above 2 at the individual-month level -- making this a *real-but-fragile* anomaly unlikely to survive serious implementation constraints.

## The claim

> *Does a stock keep outperforming in the same calendar month it always has?*

## What we tested

The Heston & Sadka (2008) recipe: at the end of each month, rank the large-cap
universe by each stock's average historical return in the *coming* calendar month
(formed from at least 5 prior occurrences, with an 11-month lag to avoid look-ahead).
Long the top decile (stocks that historically excel in the coming month); short the
bottom decile (historically weak in the coming month); hold 1 month and rebalance.
We pin the result against (a) the equal-weight market, (b) a random-portfolio control
of identical decile size, and (c) a sub-period breakdown from 1999 to 2026.
We also report per-calendar-month t-stats, a turnover cost sweep, and a beta
decomposition.  A deterministic synthetic positive control confirms the engine
recovers planted seasonality when it exists.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the same-month story, the top-decile race in plain language, why concentration and survivorship inflate the headline, and which months actually drive the effect |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t*-stats, bootstrap Sharpe CI, sub-period decay, per-calendar-month breakdown, turnover cost sweep, survivorship-bias caveat, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`same_month_seasonality/`](same_month_seasonality/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
