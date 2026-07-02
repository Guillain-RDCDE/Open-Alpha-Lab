# References & literature map — Study 572 (Capex-Cycle)

## The claim, at full strength

- **Titman, Wei & Xie (2004)**, *"Capital Investments and Stock Returns."* *Journal of Financial
  and Quantitative Analysis* 39(4). Firms that increase capital investment (abnormal CapEx relative
  to their base) earn **lower** subsequent returns — the over-investment / empire-building channel.
  The direct source of the capex-*investment* anomaly this study puts in *growth* form.
- **Cooper, Gulen & Schill (2008)**, *"Asset Growth and the Cross-Section of Stock Returns."*
  *Journal of Finance* 63(4). Total-asset YoY growth is one of the strongest cross-sectional
  return predictors: high asset growth → low returns. Capex growth is the **capex-specific slice**
  of that same over-investment effect — the cousin this study isolates.
- **Xing (2008)**, *"Interpreting the Value Effect Through the Q-Theory: An Investment Growth
  Perspective."* *Review of Financial Studies* 21(4). Ties the **growth of investment** (not just
  its level) to low expected returns via q-theory — the theoretical anchor for a capex-*cycle*
  (change-of-intensity) signal.
- **Fama & French (2015)**, *"A Five-Factor Asset Pricing Model."* *Journal of Financial
  Economics* 116(1). Packages the investment channel as **CMA** (conservative-minus-aggressive):
  firms that invest conservatively out-earn aggressive investors. The factor-level statement of the
  effect this study tests at the firm level.

## The signal we build

- **capex_intensity_t = |CapEx_t| / TotalAssets_{t-1}**, and **capex_cycle = capex_intensity_t −
  capex_intensity_{t-1}** — the *change* in capex intensity (a binge if positive). This is the
  capex analogue of asset **growth**: not the level of investment (Titman-Wei-Xie / Study 523) but
  its acceleration. CapEx and Total Assets come from the yfinance annual cash-flow and balance-sheet
  statements; the ~4-5-year history means the real panel is a **snapshot cross-section** (named on
  the SIGNAL axis and the reason the study is capped below REAL).

## Neighbours on this bench (the dedup map)

- **[Study 244 — Asset-Growth](../../244-asset-growth/)** — the *total-asset* growth anomaly
  (Cooper-Gulen-Schill). Study 572 is the **capex-specific** growth slice: only the investment
  component, expressed as a change of capex intensity.
- **[Study 523 — Investment-To-Assets](../../523-investment-to-assets/)** — the capex **level**
  (Titman-Wei-Xie IA = CapEx_t / Assets_{t-1}), on a deep SEC-EDGAR panel. Study 572 tests the
  **change** of that intensity (the cycle / binge) on a yfinance snapshot — the growth cousin, not
  the level.
- **[Study 231 — Sloan-Accruals](../../231-sloan-accruals/)** /
  **[Study 522 — Percent-Operating-Accruals](../../522-percent-operating-accruals/)** — the
  accruals family. Capex is a *cash* investment, not an accrual; different channel of the same
  over-investment / earnings-quality literature.

## Shared method

- **Information coefficient (IC)** — the cross-sectional Pearson/Spearman correlation between a
  signal and forward return, the standard headline statistic for a cross-sectional factor; the
  Pearson IC carries a *t*-stat via *t* = *r*·√((n−2)/(1−*r*²)).
- **Welch (1947)** — the unequal-variance two-sample *t* for the harvest-minus-binge tercile hedge.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  capex_cycle labels against forward returns and read the hedge spread's tail probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust
  *t* ≥ 2 on a real tape plus a placebo null and seed-robustness), the explicit survivorship
  caveat, one execution lag, and costs one-way × NAV with shorts paying borrow.
