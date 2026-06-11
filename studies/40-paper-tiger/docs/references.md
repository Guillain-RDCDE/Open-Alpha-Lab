# References & literature map — Study 40 (Paper-Tiger)

The strategy under test, and the apparatus used to judge it.

## The strategy and its source

- **Antonacci, G. (2014).** *Dual Momentum Investing: An Innovative Strategy for Higher Returns with
  Lower Risk.* McGraw-Hill. — the canonical statement of **Global Equities Momentum (GEM)**: relative
  momentum to pick the stronger equity, absolute (time-series) momentum vs T-bills as the crash filter.
- **Antonacci, G. (2013).** *Risk Premia Harvesting Through Dual Momentum.* SSRN
  [1585517](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1585517) — the paper the vendor lists
  as *"Momentum Asset Allocation Strategy"*.
- **Vendor entry** — [paperswithbacktest / awesome-systematic-trading](https://github.com/paperswithbacktest/awesome-systematic-trading),
  *"Momentum Asset Allocation Strategy"* (Sharpe `0.321`), with a QuantConnect implementation. This study
  is the desk's independent replication & teardown of that listing. See the backlog triage in
  [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## The premia it rests on

- **Moskowitz, T., Ooi, Y. H., & Pedersen, L. H. (2012).** *Time Series Momentum.* Journal of
  Financial Economics 104(2), 228–250 — the absolute-momentum leg.
- **Jegadeesh, N., & Titman, J. (1993).** *Returns to Buying Winners and Selling Losers.* Journal of
  Finance 48(1), 65–91 — cross-sectional (relative) momentum.
- **Faber, M. (2007).** *A Quantitative Approach to Tactical Asset Allocation.* Journal of Wealth
  Management — the 10-month-SMA cousin the vendor's QuantConnect file actually implements (a faithful-
  replication caveat: the listed code is GTAA trend, not GEM proper; we implement GEM as the paper defines it).

## The apparatus (why the verdict is honest)

- **Lo, A. (2002).** *The Statistics of Sharpe Ratios.* Financial Analysts Journal 58(4), 36–52 — the
  i.i.d. standard error behind the Sharpe t-stat (`quantlab.analytics.sharpe_with_se`).
- **Newey, W., & West, K. (1987).** *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix.* Econometrica 55(3), 703–708 — the HAC mean t-stat
  (`quantlab.analytics.mean_tstat_hac`).
- **McLean, R. D., & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return
  Predictability?* Journal of Finance 71(1), 5–32 — the post-publication-decay frame behind the
  sub-period comparison.

## Data

- **Yahoo! Finance** — monthly total-return (auto-adjusted) prices for **SPY, EFA, AGG** and the
  **^IRX** 13-week T-bill discount yield (the cash hurdle), 2003-09 → 2026-06. ETFs are the cheapest,
  longest, most liquid proxies for GEM's three doors; the offline synthetic world exercises the
  machinery without the network.

*Companion study: [31 Trade-Winds](../../31-trade-winds/) (time-series momentum as a portfolio
diversifier) — the contrast that makes Paper-Tiger's `FRAGILE` honest: trend *raises* a portfolio's
Sharpe, GEM only trims its drawdown.*
