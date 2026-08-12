# References — Study 895 (Defensive Momentum)

## The claim's source

The "defensive momentum" / "momentum without the crashes" idea — that momentum's fat left
tail (its violent reversals) can be tamed by combining it with a low-volatility sleeve, for
a smoother ride at a similar long-run return:

- **BlackRock / iShares, factor-combination marketing** — the retail version of the pitch
  that pairing momentum (MTUM) with min-vol (USMV) gives "the growth engine with a seat
  belt." <https://www.ishares.com/us/strategies/factors>
- **iShares MSCI USA Momentum Factor ETF (MTUM)** — the momentum sleeve (launched
  2013-04-16; its 2013 inception is why the 2008-09 momentum crash is out of this sample).
  <https://www.ishares.com/us/products/251614/>
- **iShares MSCI USA Min Vol Factor ETF (USMV)** — the min-vol sleeve (launched 2011-10-18).
  <https://www.ishares.com/us/products/239695/>

## Key papers

- **Daniel, K., Moskowitz, T. (2016), "Momentum Crashes",** *Journal of Financial Economics*
  122(2) — the canonical account of momentum's rare, violent reversals (1932, 2009) and the
  entire motivation for a "defensive" overlay. <https://doi.org/10.1016/j.jfineco.2015.12.002>
- **Barroso, P., Santa-Clara, P. (2015), "Momentum has its moments",** *JFE* 116(1) —
  volatility-scaling / risk-managed momentum roughly doubles the strategy's Sharpe by cutting
  the crash tail; the closest academic cousin of the inverse-vol blend tested here.
  <https://doi.org/10.1016/j.jfineco.2014.11.010>
- **Jegadeesh, N., Titman, S. (1993), "Returns to Buying Winners and Selling Losers",**
  *Journal of Finance* 48(1) — the momentum premium being wrapped.
  <https://doi.org/10.1111/j.1540-6261.1993.tb04702.x>
- **Ang, A., Hodrick, R., Xing, Y., Zhang, X. (2006), "The Cross-Section of Volatility and
  Expected Returns",** *Journal of Finance* 61(1) — the low-volatility anomaly USMV wraps.
  <https://doi.org/10.1111/j.1540-6261.2006.00836.x>
- **Asness, C., Frazzini, A., Pedersen, L. (2012), "Leverage Aversion and Risk Parity",**
  *Financial Analysts Journal* 68(1) — why inverse-vol / risk-parity weighting sits *on* the
  efficient line unless the sleeves' Sharpes and correlation cooperate. <https://doi.org/10.2469/faj.v68.n1.1>
- **Markowitz, H. (1952), "Portfolio Selection",** *Journal of Finance* 7(1) — the mean-
  variance frame that says a blend earns a diversification *lift* only from imperfect
  correlation, not from mixing per se. <https://doi.org/10.1111/j.1540-6261.1952.tb01525.x>

## Desk siblings (dedup guard)

- [**508-momentum-crashes**](../../508-momentum-crashes/) — grades the momentum-crash
  *mechanism itself* (the Daniel-Moskowitz left tail on the academic factor). This study does
  not re-litigate that; it asks whether **blending two shipped ETFs** buys a better
  risk-adjusted deal than momentum alone.
- [**330-low-volatility-anomaly**](../../330-low-volatility-anomaly/) — the academic low-vol
  cross-section (the anomaly USMV wraps), graded as a stand-alone factor, not as a blend
  ingredient.
- [**601-factor-etf-live-test**](../../601-factor-etf-live-test/) — audits USMV / MTUM / QUAL
  as **single live wrappers** vs SPY (exposure delivered, alpha not). Study 895 is the
  **combination** question those single-fund audits leave open.
- [**237-residual-momentum**](../../237-residual-momentum/) — a *different* fix for momentum's
  crashiness (stripping market beta from the momentum signal), not a min-vol blend.

## Data sources

- **yfinance** (public, no key) — daily auto-adjusted (total-return) closes: MTUM, USMV,
  QUAL, SPY and BIL (1-3M T-bill ETF, the cash / risk-free leg — its own total return is the
  monthly risk-free rate, so excess-of-cash needs no yield-to-return conversion).
  <https://github.com/ranaroussi/yfinance>
- Method citations shared by the desk: Newey-West (1987) HAC errors; Welch (1947) unequal-
  variance *t*; Efron / Künsch moving-block bootstrap for the Sharpe-advantage CI.
