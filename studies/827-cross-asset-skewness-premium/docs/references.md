# References & literature map — Study 827 (Cross-Asset Skewness Premium)

## The claim under test

- **The source effect (single-name).** Diego **Amaya, Peter Christoffersen, Kris Jacobs &
  Aurelio Vasquez**, *"Does Realized Skewness Predict the Cross-Section of Equity Returns?"*
  (Journal of Financial Economics, 2015). Sorting individual **stocks** on their **realized
  return skewness**, they find a robust **negative** relation to next-period returns: the most
  right-skewed (lottery-like) names under-earn, so a long low-skew / short high-skew book earns a
  positive spread. This study asks whether that relation has an **asset-class analogue**.
- **The behavioural reading, lifted to asset classes.** If skewness-loving investors over-pay for
  a fat upside tail *within* the stock cross-section, perhaps they do the same *across* asset
  classes — bidding up whichever class has recently looked most lottery-like (a commodity spike, an
  EM equity melt-up) and depressing its forward return. The natural test is a cross-asset sort on
  each class's own realized skew.
- **The specific test here.** Nine liquid asset-class ETFs; each class's **trailing-126-day realized
  skewness** of daily returns; a monthly point-in-time sort (signal at month-end `m−1`, hold month
  `m`) into a long-bottom-⅓ / short-top-⅓ book; a Newey-West *t* on the monthly spread, a permutation
  placebo, a two-era and multi-window robustness cut, a costed timer, and a seeded synthetic positive
  control.

## Where the cross-asset skew literature sits

- **Skewness preference / lottery demand.** **Barberis & Huang (2008)**, *"Stocks as Lotteries"*,
  and **Kumar (2009)**, *"Who Gambles in the Stock Market?"* — the demand-for-positive-skew mechanism
  that the single-name effect exploits. Whether that demand is strong enough to move whole asset
  classes (vs individual lottery stocks) is exactly the open question this study probes.
- **Bali, Cakici & Whitelaw (2011)**, *"Maxing Out"* — the MAX single-tail proxy; a coarser sibling
  of realized skewness, tested at the single-name level in Study 365.
- **Cross-asset factor premia.** **Asness, Moskowitz & Pedersen (2013)**, *"Value and Momentum
  Everywhere"*, and **Koijen, Moskowitz, Pedersen & Vrugt (2018)**, *"Carry"* — the template for
  running a single signal across a common cross-section of asset classes (the "…everywhere" design).
  This study applies that design to the **third moment** rather than value, momentum or carry, and
  finds no premium — a useful null in that literature.

## What we measure, and the honesty rails

- **Realized skewness, no free model.** For each class, the rolling `window`-day sample skewness of
  daily simple returns (population third standardised moment), computed vectorised via the moment
  identity `m3 / m2**1.5`.
- **Point-in-time sort, one documented lag.** The ranking signal is the trailing skewness **known at
  month-end `m−1`**; the book is held over month `m`. Zero look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 6-lag) *t* on the monthly long-short spread; a
  one-sample *t* and a pooled Welch *t* (low-skew book vs high-skew book) cross-check; a
  **1,000-permutation asset-label placebo** breaks the signal → forward-return link.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership** set of nine
  canonical class-proxy ETFs — small, fixed, low-turnover, so the survivorship exposure is far milder
  than a single-name universe, but the caveat is still carried.
- **The timer is graded separately.** Costs are one-way × NAV per monthly rebalance on the long-short
  book, and the short book pays borrow — the honest test of whether a monthly spread survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent covariance
  (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.

## Data sources

- **yfinance daily total-return closes** (`auto_adjust=True`), nine asset-class ETFs (SPY, EFA, EEM,
  TLT, LQD, HYG, GLD, DBC, VNQ), 2007-01-03 → 2026-06-30, cached under `_cache/`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [803-realized-skewness-reversal](../../803-realized-skewness-reversal/) — the **single-name**
  realized-skewness reversal (individual stocks). This study is its **asset-class** analogue: one
  skewness per asset class, a nine-ETF cross-section, not a stock cross-section.
- [660-carry-everywhere](../../660-carry-everywhere/) — cross-asset **carry** (the yield / roll
  signal) run across asset classes. Same "…everywhere" cross-section, a **different signal** (the
  first moment / income, not the third moment / skew).
- [638-value-momentum-everywhere](../../638-value-momentum-everywhere/) — cross-asset **value** and
  **momentum** (level & trend). Again the same cross-section, a different signal — neither sorts on
  realized skewness.

None of the siblings sort a cross-section of **asset classes** on their **own realized third moment**
— the cross-asset skewness premium — which is this study's own axis.
