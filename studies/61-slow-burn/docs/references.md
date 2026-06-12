# References & literature map — Study 61 (Slow-Burn)

## The mechanics

- **Cheng, M., & Madhavan, A. (2009).** *The Dynamics of Leveraged and Inverse Exchange-Traded Funds.*
  Journal of Investment Management — the daily-rebalancing path dependence and volatility decay.
- **Avellaneda, M., & Zhang, S. (2010).** *Path-Dependence of Leveraged ETF Returns.* SIAM Journal on
  Financial Mathematics — the 0.5·L·(L−1)·σ² drag, derived.
- **Vendor / folk belief** — "leveraged ETFs decay to zero"; backlog:
  [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## On regime dependence

- In a sustained trend, daily compounding *amplifies* gains beyond static leverage; in choppy/bear
  regimes the reset plus drag destroy capital (2022: TQQQ −79%). The path, not just the average, matters.

## Data

- **Yahoo! Finance** — TQQQ (ProShares UltraPro QQQ, 3×) and QQQ, daily total return, 2010–2026
  (TQQQ inception). The offline synthetic world generates an underlying with tunable volatility so the
  0.5·L·(L−1)·σ² drag is provable offline (and a zero-vol null).

*A product-mechanics teardown; companion to the leverage-cost lesson in [30 House-Edge](../../30-house-edge/)
and [43 Free-Lunch](../../43-free-lunch/).*
