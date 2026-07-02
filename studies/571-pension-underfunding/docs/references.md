# References & literature map — Study 571 (Pension-Underfunding)

## The claim, at full strength

- **Franzoni, F. & Marin, J. M. (2006)**, *"Pension Plan Funding and Stock Market Efficiency."*
  *Journal of Finance* 61(2). The canonical statement of the **pension-underfunding anomaly**:
  firms with the most **underfunded** defined-benefit pension plans (the largest pension deficit
  scaled by market value) go on to earn *anomalously low* risk-adjusted returns — roughly −7%/yr
  — because the market **under-reacts** to the magnitude of this senior, off-balance-sheet
  liability. Investors correct the misvaluation only slowly, as the funding shortfall shows up in
  cash contributions and earnings. The measure this study proxies.
- **Coronado, J. & Sharpe, S. (2003)**, *"Did Pension Plan Accounting Contribute to a Stock Market
  Bubble?"* *Brookings Papers on Economic Activity*. Shows investors value the *smoothed pension
  accounting* number rather than the true economic funded status — the mispricing channel behind
  the Franzoni-Marin anomaly.
- **Jin, L., Merton, R. C. & Bodie, Z. (2006)**, *"Do a Firm's Equity Returns Reflect the Risk of
  Its Pension Plan?"* *Journal of Financial Economics* 81(1). Confirms pension risk is (eventually)
  priced in equity betas — the flip side: if it were *fully* and promptly priced there would be no
  under-reaction anomaly.
- **Picconi, M. (2006)**, *"The Perils of Pensions: Does Pension Accounting Lead Investors and
  Analysts Astray?"* *The Accounting Review* 81(4). Analysts and investors fail to fully impound
  pension-footnote information — the accounting-opacity mechanism of the anomaly.

## The funding measure we build

- Franzoni-Marin scale the **funded status** — plan assets minus the projected benefit obligation
  (PBO) — by market value of equity: `FR = (assets − PBO) / mktcap`. A negative `FR` is a hole; the
  most-negative decile is the "most underfunded." This study builds the identical measure
  (`funding_gap`) on a synthetic panel and z-scores the *depth* of the hole so that higher =
  more underfunded, then sorts and tests. The dropped realism (true point-in-time PBO/assets from
  Compustat pension items; the SFAS-158 balance-sheet net funded status post-2006) is named as the
  **data-availability limitation** on the SIGNAL axis — it is why this study is synthetic-only.

## Why synthetic-only

- The point-in-time pension footnote (PBO, plan assets, funded status) lives in Compustat pension
  items (`PPROBF`, `PBPRO`, and post-SFAS-158 the net funded status). No free no-key retail
  endpoint exposes it — yfinance `.balance_sheet` has no pension line and there is no free
  point-in-time pension-footnote panel. The desk's other synthetic-only studies
  ([273 Lego-Returns](../../273-lego-returns/), [275 Whisky-Cask](../../275-whisky-cask/),
  [276 Sneaker-Resale](../../276-sneaker-resale/)) take the same stance: when the free data does not
  exist, build synthetic-only and cap the Signal axis at `WEAK`/`NONE`, never `REAL`.

## Neighbours on this bench (the dedup map)

- **[Study 154 — Leverage-Anomaly](../../154-leverage-anomaly/)** — the *on*-balance-sheet
  financial-leverage anomaly (Penman-Richardson-Tuna). Study 571 is the *off*-balance-sheet
  cousin: pension underfunding as hidden leverage the market may miss.
- **[Study 231 — Sloan-Accruals](../../231-sloan-accruals/)** and
  **[Study 522 — Percent-Operating-Accruals](../../522-percent-operating-accruals/)** — accruals
  anomalies where investors over-weight low-quality accounting numbers. The pension-underfunding
  anomaly is the same *market-under-reacts-to-an-opaque-accounting-item* family, applied to the
  pension footnote.
- **[Study 540 — Distress-Risk-Anomaly](../../540-distress-risk-anomaly/)** — another
  "distressed/levered firms earn low returns" anomaly, sorted on failure probability. Study 571 is
  narrower: the *pension* hole specifically, as a distinct off-balance-sheet leverage signal.

## Shared method

- **Welch (1947)** — the unequal-variance two-sample *t* used for the well-funded-minus-underfunded
  bucket spread.
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  hole-depth labels against forward returns and read the spread's tail probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust *t* ≥
  2 **on a real tape** for `REAL`; synthetic-only is capped at `WEAK`/`NONE`), the explicit
  data-availability caveat on the SIGNAL axis, one execution lag (report → entry), and costs
  one-way × NAV with the short leg paying borrow.
