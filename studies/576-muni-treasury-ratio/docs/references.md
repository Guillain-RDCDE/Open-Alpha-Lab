# References & literature map — Study 576 (Muni-Treasury-Ratio)

## The claim, at full strength

- **The muni-Treasury ratio ("M/T ratio")** is the single most-quoted rich/cheap gauge in the
  municipal market: the tax-exempt AAA-GO (Municipal Market Data / MMD) yield divided by the
  comparable-maturity Treasury yield. Because muni coupons escape federal income tax, the ratio
  normally prints *below* 1.0 (a top-bracket investor is indifferent at ratio ≈ 1 − marginal tax
  rate). Muni strategists (e.g. the daily MMD 10-year ratio quoted across the sell side) treat a
  *high* ratio as munis being *cheap* and a *low* ratio as *rich* — the folklore this study tests
  as a timing signal.
- **Chalmers (1998)**, *"Default Risk Cannot Explain the Muni Puzzle."* *Review of Financial
  Studies* 11(2). Documents that muni-Treasury yield spreads are far larger than default risk
  justifies — the "muni puzzle" that makes the ratio move for tax/liquidity/segmentation reasons
  as much as for relative value, so its *timing* content is not obvious a priori.
- **Green, Hollifield & Schürhoff (2007)**, *"Financial Intermediation and the Costs of Trading in
  an Opaque Market."* *Review of Financial Studies* 20(2). The muni market's opacity and dealer
  markups — why a retail timing rule off a coarse yield proxy faces frictions a Treasury trade
  does not.
- **Ang, Bhansali & Xing (2010)**, *"Taxes on Tax-Exempt Bonds."* *Journal of Finance* 65(2). The
  tax wedge that sets the muni/Treasury ratio's *level* (and moves it as marginal tax expectations
  change) — a reminder that the ratio is a **tax-driven artifact** as much as a valuation signal,
  the study's central caution.

## The signal we build

- The tradable M/T ratio a desk quotes is MMD-AAA-GO yield / Treasury yield, which is not free on a
  no-key retail stack. This study proxies it with the **trailing-12-month distribution yield** of
  MUB (muni ETF) over that of IEF (7-10Y Treasury ETF). The distribution-yield proxy is
  levels-biased (mean ≈ 1.30 vs the ~0.85 MMD ratio), so the signal is the **trailing z-score** of
  the ratio, which strips the level and keeps only relative rich/cheap moves — the proxy gap is
  named on the SIGNAL axis.

## Neighbours on this bench (the dedup map)

- **[Study 132 — Yield-Curve-Steepener](../../132-yield-curve-steepener/)** — the Treasury
  *curve-slope* timer for the long bond. Study 576 is a **cross-sector relative-value** ratio (muni
  vs Treasury), not a single-curve slope; the instrument is the muni/Treasury *ratio*, not the
  10Y-3M spread.
- **[Study 119 — Real-Rate-Regime](../../119-real-rate-regime/)** / **[Study 120 — Excess-CAPE-Yield](../../120-excess-cape-yield/)**
  — regime/valuation timers off a single yield level. Study 576's signal is a *ratio of two* yields
  and its target is a muni-minus-Treasury *excess* return.
- **[Study 247 — Bond-Seasonality](../../247-bond-seasonality/)** / **[Study 380 — Curve-Roll-Down](../../380-curve-roll-down/)**
  — other fixed-income edges; neither is a muni/Treasury relative-value ratio.

## Shared method

- **Newey & West (1987)** — the heteroskedasticity- and autocorrelation-consistent (HAC) standard
  error used for the predictive-slope *t*. Essential here: overlapping forward-horizon windows are
  strongly autocorrelated, so a naive OLS/two-sample *t* is badly inflated (the study demonstrates
  the naive quintile *t* = +4.12 collapsing to +0.51 on non-overlapping windows).
- **Welch (1947)** — the unequal-variance two-sample *t* for the Q5 − Q1 quintile-spread contrast.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  ratio-z labels against the forward excess and read the spread's tail probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a HAC/Lo *t* ≥ 2
  on the *real* tape for `REAL`; literature support alone reads `WEAK`), the overlap caveat, one
  execution lag, and costs one-way × NAV with the short leg paying borrow.
