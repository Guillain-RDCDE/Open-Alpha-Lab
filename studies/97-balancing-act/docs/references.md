# References & literature map — Study 97 (Balancing-Act)

## The claim under test

The **60/40 portfolio** — 60% stocks, 40% bonds — is the canonical "balanced fund": the
sensible default an adviser hands a typical investor. The strong, sold-at-full-strength
version is that it gives you *most of the stock return with much less risk*, a **better
risk-adjusted outcome** than 100% stocks, and that **bonds cushion every equity crash**
because they rally when stocks fall (the "flight to quality").

- The balanced-fund tradition: Vanguard's *Balanced Index Fund* (VBINX, since 1992) and the
  *Wellington Fund* (1929) are the archetypal ~60/40 vehicles sold as the default core.
- The "60/40 is the standard" framing is ubiquitous in adviser and fund-company material,
  e.g. Vanguard, *"The 60/40 portfolio"* commentary, and Morningstar's balanced-allocation
  category definition.

## The 2022 "death of 60/40" debate

- 2022 broke the central premise: stocks **and** bonds fell together as the Fed hiked into
  inflation. A wave of pieces declared 60/40 dead — e.g. *The Wall Street Journal*,
  *"The 60-40 Investment Strategy Is Back After Tough Year"* (and the 2022 pieces it
  answers); Bank of America and others called 2022 one of the worst years for the mix on
  record. The rebuttal (AQR, Vanguard) argued the diversification logic still holds *on
  average* and that 2022 was a stock+bond *valuation* reset, not a permanent regime.

## Why the steelman is almost coherent

- **Diversification is real when correlation is low or negative.** A fixed-weight blend of
  two imperfectly-correlated assets has lower variance than the weighted average of the
  legs' variances (Markowitz, *Portfolio Selection*, JF 1952). When stock/bond correlation
  is negative — as it largely was in 2002–2021 — the blend's Sharpe genuinely exceeds
  stocks' alone.
- **Bonds were a great diversifier in the disinflation era.** From the early 2000s the
  rolling stock/bond correlation was mostly negative, so Treasuries rallied in most equity
  selloffs (2002, 2008, 2018, 2020) — the cushion was observed, repeatedly.

## Why it is likely to fail *as stated* ("a free structural law", "cushions every crash")

- **The cushion is regime-dependent, not a law.** Stock/bond correlation is itself
  time-varying and flips positive in inflation/rate-shock regimes — historically the norm
  before ~2000 (Campbell, Sunderam & Viceira, *Inflation Bets or Deflation Hedges?*, 2017).
  2022 is the live counter-example: bonds fell *with* stocks.
- **Much of the realised Sharpe edge is the bond bull market** — a 40-year decline in
  yields you were *paid* for (beta), not a free structural diversification law. The same
  caveat the desk drew for Study 68 (All-Weather): the leverage/return story rests on a
  bond bull that 2022 ended.

## Method lineage

- **Newey–West HAC standard errors** for the mean of an autocorrelated return-difference
  series: Newey & West (1987), Econometrica.
- **Circular block bootstrap** for a CI on the Sharpe *difference* — resampling in blocks
  preserves volatility clustering and the cross-asset co-movement that i.i.d. resampling
  destroys (Politis & Romano, 1994; Ledoit & Wolf, *Robust performance hypothesis testing
  with the Sharpe ratio*, JEF 2008).
- **Excess-of-cash comparison.** SHY (1-3y Treasuries) is the cash proxy so the Sharpe race
  between a part-bond blend and 100% stocks is excess-of-cash vs excess-of-cash, per the
  desk house rule.

## Data sources used

- **SPY** (stocks), **IEF** (7-10y Treasuries), **TLT** (20y+ Treasuries) and **SHY**
  (1-3y Treasuries, cash proxy), daily, **total-return adjusted** (dividends + splits) via
  the shared cross-asset panel (`yfinance auto_adjust=True`) with a `quantlab.data` fallback
  per ticker. The Treasury ETFs list **2002-07-30**, which bounds the joint window honestly
  — stated as a decision, not buried.

## Related desk studies

- [Study 68 — All-Weather](../../68-all-weather/) — risk parity / *volatility*-weighted
  diversification. **This study is distinct**: 68 weights by inverse volatility; 97 holds
  *fixed* 60/40 weights, rebalanced on the calendar. Cross-reference for the bond-bull and
  leverage caveats that apply to both.
