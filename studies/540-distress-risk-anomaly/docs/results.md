# Results — Study 540 (Distress-Risk-Anomaly): the CHS distress puzzle on a survivor basket

*Generated from [`distress_risk_anomaly/`](../distress_risk_anomaly/) over this study's cached
yfinance tape: daily adjusted close for a fixed **39-name large-cap survivor basket** (fingerprint
`9bf6c3a922f0`, 2014-01-02 → 2026-06-25) plus a per-name fundamentals snapshot (leverage, ROA;
fingerprint `5f664306cf34`). Distress is scored as-of **2024-06-28** (trailing 252-day equity
vol + the accounting snapshot); the forward holding window is **2024-06-28 → 2026-06-25**.
Cross-section fingerprint `0835f4217788`. As-of **2026-06-26**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Distress puzzle on the tape?" `MIXED`

Campbell, Hilscher & Szilagyi (2008) document the **distress puzzle**: the firms a failure-prediction
model flags as *most likely to fail* go on to earn the *lowest* returns — the inverse of a risk
premium. We build a CHS-style distress score (high leverage − profitability + equity volatility),
sort a 39-name blue-chip survivor basket into terciles, and test whether the safe names beat the
distressed ones.

On the **headline 2024-06 → 2026-06 window the puzzle is not merely absent — it inverts.** The
*distressed* tercile earned **+82.7%** versus the safe tercile's **+34.1%**, a long-safe/short-distressed
spread of **−48.6%** (two-sample *t* = **−2.64**, placebo *p* = 0.033). The firm-level slope of
forward return on distress is **positive** (*t* +1.99): the high-leverage, high-vol names *ripped*
in the AI/high-beta melt-up. So `NONE` on the signal axis (the long-safe/short-distressed trade has
the wrong sign here and no stable *t* ≥ 2 for the puzzle), `MIRAGE` on tradability (a tiny survivor
basket, annual-rebalance, sign-unstable, with a punitive borrow on the distressed short), and
`MIXED` on the puzzle itself: it **does** appear (distressed underperform, slope-*t* −2.0 / −1.5) in
the 2021-23 and 2022-24 windows but **inverts** in 2023-25 and 2024-26.

## Data stamp

- **Prices**: 39 large-cap survivors + SPY, daily adjusted close, 2014-01-02 → 2026-06-25,
  fingerprint `9bf6c3a922f0`
- **Fundamentals snapshot**: leverage (total liabilities / total assets) and ROA (net income /
  total assets), 39 names, fingerprint `5f664306cf34`
- **Cross-section** (scored 2024-06-28, forward to 2026-06-25): 39 names, fingerprint `0835f4217788`

## The distress sort — the puzzle is the WRONG WAY ROUND here

| Tercile (12 names) | Forward return 2024-06 → 2026-06 |
|---|---|
| **Safe** (lowest distress: NVDA, GOOGL, MSFT, PG, JNJ, KO, AAPL, MRK, XOM, CVX, COST, WMT) | **+34.1%** |
| **Distressed** (highest distress: AVGO, JPM, ABBV, GE, NEE, GS, WFC, C, BAC, MS, ORCL, BA) | **+82.7%** |
| **Spread (safe − distressed)** | **−48.6%** (two-sample *t* −2.64) |

The puzzle predicts safe > distressed (a *positive* spread). The tape delivers the opposite: the
distressed bucket — banks (high leverage), Broadcom/Oracle/GE (high vol) — roughly *doubled* the
safe staples-and-mega-cap bucket. The label-shuffle placebo *p* = **0.033** says this inversion is
not noise on this window; it is a real, *anti*-puzzle outcome.

## The firm-level relation

| | value |
|---|---|
| Slope (forward_ret on distress) | **+11.0%** per distress unit |
| Slope *t* | **+1.99** (a *negative* slope would be the puzzle) |
| corr(distress, forward return) | **+0.31** |

A *positive* slope is the anti-puzzle: more distress, more return, over this window.

## Robustness — the sign is not stable

| Scoring window → hold | Safe − distressed spread | Firm slope-*t* | Reads as |
|---|---|---|---|
| 2021-06 → 2023-06 | **+22.0%** | **−2.02** | puzzle present |
| 2022-06 → 2024-06 | **+26.9%** | **−1.46** | puzzle present (weak) |
| 2023-06 → 2025-06 | **−45.6%** | +0.42 | inverted |
| 2024-06 → 2026-06 (headline) | **−48.6%** | +1.99 | inverted |

The puzzle appears in the two earlier windows (distressed underperform, slope-*t* −2.0 / −1.5) and
**flips sign** in the two recent ones. A signal whose *sign* depends on the window is not bankable
— `NONE` on the signal axis, `MIXED` on the puzzle.

## Costs

| | value |
|---|---|
| Gross spread (safe − distressed, headline window) | **−48.6%** |
| Net (5 bps/leg round-trip + 100 bps/yr borrow, 2y hold) | **−50.8%** |

Costs are almost a footnote here: the trade is the *wrong sign* before you pay for it, and the
distressed leg you would short is exactly the expensive-to-borrow tail.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `distress_alpha` | Mean slope-*t* (25 seeds) | |
|---|---|---|
| 0.00 (null) | **−0.04** | flat — no false signal |
| −0.06 | −0.87 | puzzle emerging |
| −0.12 | −1.69 | puzzle visible |
| −0.20 | **−2.78** | clears the bar |

At the null the slope-*t* is ≈ 0; planting a genuine puzzle (`distress_alpha < 0`) drives the slope
negative and past −2 as it grows. The detector works — so the real-tape result is a statement about
**this survivor basket on this window**, not a broken engine. (Control only; never cited for the
real-tape stamp.)

## Why the puzzle doesn't certify here

1. **Survivorship, the wrong way.** The basket is names *still trading in 2026*. The real CHS effect
   is driven by firms that *actually went bankrupt* (the extreme-distress tail that delivers the
   crushing losses). Strip those out and you keep only the survivors of distress — biasing the tape
   *against* the puzzle, and toward an anti-puzzle in a melt-up.
2. **A high-beta melt-up window.** 2023-26 rewarded leverage and volatility (the AI/banks rally), so
   the "distressed" blue chips *led*. The puzzle earns its keep over full cycles and in distress
   *events*, not a two-year bull run.
3. **Accounting snapshot, not point-in-time.** yfinance exposes only a shallow statement history, so
   the leverage/ROA legs are a recent snapshot rather than a true year-*t* panel — adequate for a
   blue-chip cross-section, but a real CHS replication needs Compustat-style point-in-time data.

## The honest takeaway

The distress puzzle is real in the literature and on full-cycle, full-universe data — but on a
39-name survivor basket over 2024-26 it **inverts**: the most-distressed names earned the most
(*t* −2.64 for the *anti*-puzzle), the firm-level slope is positive, and the sign flips across
windows. `NONE` × `MIRAGE`, with the puzzle itself `MIXED` (present in 2 of 4 windows, gone in the
other 2). The synthetic control confirms the engine would catch a real puzzle — so this is the tape
talking, not the code.
