# Pre-registration — Study 71 (Ambush)

*Written and frozen **before** the first backtest run on the real tape (2026-06-12).
Every threshold below is inherited from a prior study on this bench or from the
literature — none is fitted to the confluence book's own results. If the run
contradicts the hope, the verdict says so; the goalposts do not move.*

## The hypothesis

Four short-horizon S&P 500 effects are individually **real gross but dead net** on this
bench, all for the same reason — they trade nearly every day, so the bid-ask eats them:

| Ingredient | Source study | Status there |
|---|---|---|
| Low-IBS close bounces next day | [19 — Rubber-Band](../../19-rubber-band/) | gross Sharpe +1.56 (t=8.6), net −0.37 |
| Turn-of-the-month drift | [42 — Last-Call](../../42-last-call/) | +11 bp/d pre-2008, faded since |
| Down day precedes bounce | [13 — Crimson-Hour](../../13-crimson-hour/) | weak intraday lift |
| VIX stress precedes recovery | [03 — Fear-Gauge](../../03-fear-gauge/) | +1.0% @ 1w (p≈0), Sharpe ≈ 0 traded daily |

**H₁ (Signal):** days when *k* of these fire together carry a next-day gross premium that
*increases monotonically in k*, and is statistically real at the k ≥ 3 confluence.
**H₂ (Tradability):** because k ≥ 3 fires only a handful of times a year, the premium
survives realistic CFD costs — rarity, not signal strength, is the cost defence.

## Frozen definitions (all known at the close of day *t*; they earn day *t+1*)

- **S1 · IBS low** — `IBS = (C−L)/(H−L) ≤ 0.20` (study 19's low bucket).
- **S2 · TOM** — day *t+1* falls in the classic [−1, +3] turn-of-month window
  (study 42's `tom_mask`, calendar-known, needs no lag).
- **S3 · Red day** — `Close_t < Close_{t−1}`.
- **S4 · VIX stress** — `VIX_t ≥ 1.15 × mean(VIX, 20d trailing)` (relative stress;
  study 03 showed absolute levels are regime-dependent).
- **Confluence** `C_t = S1+S2+S3+S4`; the book is **long-only**: long when `C_t ≥ K`,
  flat otherwise. Headline `K = 3`; K ∈ {1,2,3,4} all reported and the family is
  corrected with White's Reality Check (stationary bootstrap) — the grid is announced
  here, so the correction covers everything we looked at.

## Frozen risk & cost model (the CFD overlay)

- **Sizing** = `min( 12%/σ̂_ann , 1%/(2·σ̂_daily) , 2.0 )` — study 16's vol-target
  convention plus a hard **1%-of-NAV daily risk budget** (a 2σ down day loses ≤ 1%).
- **Stop** — intraday stop at −1% of NAV (`r_stop = −1%/w`); if `Low_{t+1}` breaches it,
  the day's return is `r_stop` (or the open, if it gaps through); flat until the next
  close. Daily-bar approximation stated, conservative on gap-throughs.
- **Costs** — spread+slippage **1.0 bp one-way × |Δposition|** (≈ a 0.5-pt round-trip
  on a US500 CFD at ~6000); **financing** `(rf + 250 bp)/252 × w` per night held
  (rf = 13-week T-bill); the account earns rf, so the book's return is quoted
  **excess-of-cash** and raced against SPY buy-and-hold **excess-of-cash**. Cost sweep
  0→10 bp and break-even reported.
- **Sample** — SPY split-only daily OHLC 1993→as-of **2026-06-01**; ^VIX raw closes.
  **IS = through 2014-12-31, OOS = 2015→as-of.** Signal is judged full-sample;
  tradability is judged **OOS**.

## Frozen verdict criteria

- **Signal `REAL`** ⟺ HAC *t* ≥ 2 on the K≥3 next-day gross premium (full sample)
  **and** the lift is monotone in K. Otherwise `WEAK`/`NONE`.
- **Tradability `INVESTABLE`** ⟺ OOS net excess Sharpe ≥ 0.3, block-bootstrap 95% CI
  above 0, and Reality-Check p < 0.10. **`FRAGILE`** if OOS net > 0 but the CI spans 0.
  **`MIRAGE`** if OOS net ≤ 0.
- **Positive control** — the harness must light up on a synthetic tape with a planted
  confluence premium and stay dark on a seeded random walk, or no real-tape claim is made.

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice.*
