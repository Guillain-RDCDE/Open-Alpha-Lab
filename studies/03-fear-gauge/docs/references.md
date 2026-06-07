# References — Fear-Gauge 🌡️

*Sources for [Study 03](../README.md). The literature map matters here because,
unlike the Falling-Knife, the effect under test is a **named, documented risk
premium** — so the honest verdict turns on telling alpha from premium.*

## The claims under test

- **@TheProf — "the VIX rule"** (X/Twitter): *"Buy stocks when VIX hits 30, double
  down when VIX hits 50."* — the **level** trigger (V1) and the **martingale** (V4).
- **@jaltucher — the spike chart** (X/Twitter): *"S&P 500 after every VIX 30%+
  single-day spike, Jun 2016 – Jun 2026, 23 events: avg +0.42% / +0.42% / +2.66% at
  +1d / +1w / +1m; 12/14/21 of 23 positive."* — the **spike** trigger (V2). Source
  cited on the chart: CBOE VIX & S&P 500 daily closes (FRED).

> Both are folk-statements of the same intuition: a high/spiking fear gauge marks a
> buyable bottom. We steelman them, then test the family — see [the claim](../README.md#1--the-claim).

## Data

- **`^VIX`** — CBOE Volatility Index, daily close. Available since **1990**
  (back-cast to 1986 via VXO in some sources). Source: FRED `VIXCLS` / Yahoo `^VIX`.
- **`^GSPC`** — S&P 500 spot index (deep history, not tradeable).
- **`SPY`** — the tradeable S&P 500 ETF (since 1993); real prints for the cost layer.
- Note: **spot VIX is not investable.** The "tradeable" leg of any VIX rule is an
  S&P instrument (SPY) or a vol product (VIX futures / VXX-type ETN) carrying its
  own roll and decay cost — see [beat 6](../README.md#6--could-you-trade-it).

## Literature map — why the signal may be *real* but uninvestable

- **Variance risk premium (VRP).** Implied vol sits systematically above subsequent
  realised vol; selling vol is paid on average. *Carr & Wu (2009), "Variance Risk
  Premiums", RFS; Bollerslev, Tauchen & Zhou (2009), RFS.* — the prime suspect for
  any "rebound after a spike".
- **Vol mean-reversion.** VIX is strongly mean-reverting; a spike mechanically
  forecasts a decline in VIX, which co-moves with an equity rebound. *Whaley (2009),
  "Understanding the VIX", J. Portfolio Mgmt.*
- **Short-vol skew / "picking up nickels".** The VRP payoff is negatively skewed —
  small gains, rare large losses (Feb 2018 "Volmageddon", Mar 2020). The reason a
  true positive mean can still be uninvestable.
- **Mean-reversion after extremes / "buy the dip".** The price-space cousin —
  cross-link to [Study 02](../../02-falling-knife/) for the random-day-null method
  and the clustering bootstrap reused verbatim here.
- **Data-snooping / selection.** White (2000) Reality Check; the 2016–2026 window's
  exclusion of 2008 is a textbook selected-sample problem.

## Method cross-links

- Random-day null, block bootstrap, deflated/White Reality Check, alpha-vs-beta:
  the shared desk protocol, [`quantlab/`](../../../quantlab/) and the
  [methodology](../../../METHODOLOGY.md).
- The cross-study control — *does VIX add information over the price drop?* — reuses
  Study 02's price triggers directly.

*(Add precise citations / permalinks as the teardown is written.)*
