# Results — Study 321 (Earnings-Season-Tide) on real daily tape

*The whole-market calendar contrast — mean daily return on the trading days inside the
four hard-coded **peak earnings windows** (mid-late January, April, July, October) versus
all other days — run on the **SPY total-return** daily tape. The in-window position is
calendar-known, so there is **no execution lag**. Inference is a Newey-West (HAC) *t* on
the in-window minus out-of-window mean difference, with a circular-block-bootstrap CI;
the tradable overlay is raced excess-of-cash vs buy-and-hold. As-of **2026-06-18**; match
the fingerprint to confirm you hold the same tape.*

## Data stamp

| Series | Mode | Window | Days | Fingerprint |
|---|---|---|--:|---|
| SPY | total-return | 1993-01-29 → 2025-12-31 | 8,288 | `ff2579c16a7d` |

*The in-progress 2026 year is dropped before any statistic is taken.*

## The headline — the calendar contrast (gross, log returns)

| Group | days | mean (bps/day) | annualised |
|---|--:|--:|--:|
| **In peak-earnings windows** | 1,737 | **+6.86** | +17.3%/yr |
| Out of windows | 6,550 | +3.28 | +8.3%/yr |
| **Difference (in − out)** | — | **+3.59** | — |

- The in-window days *do* drift faster (+6.86 vs +3.28 bps/day), exactly the kind of gap a
  believer points to. But the **HAC *t* on the difference is only +1.27** — comfortably
  **below** the inference bar (|t| ≥ 2).
- The **circular block-bootstrap 95% CI on the difference is [−1.67, +9.17] bps/day** — it
  straddles zero. The "tide" is statistically indistinguishable from the ordinary equity
  drift you were already being paid for.

## Per-window breakdown — no single quarter carries it either

| Window | days | mean (bps/day) | HAC *t* |
|---|--:|--:|--:|
| Jan (Q4) | 361 | +0.37 | −0.72 |
| Apr (Q1) | 436 | +9.15 | +1.23 |
| Jul (Q2) | 471 | +5.44 | +0.33 |
| Oct (Q3) | 469 | +11.16 | +1.39 |

Not one of the four windows clears *t* = 2. October looks the strongest (+11 bps/day) but
at *t* = +1.39 it is well inside the noise band — and cherry-picking the strongest of four
pre-chosen windows is exactly the selection trap the desk corrects for.

## Could you trade it? — the in-window overlay vs buy-and-hold (excess-of-cash)

A "long SPY only inside the four windows, in cash otherwise" overlay, costs at 1 bp one-way
× NAV on each window entry/exit:

| Strategy | ann. return | Sharpe (excess-of-cash) | time in market | turnover/yr |
|---|--:|--:|--:|--:|
| In-window overlay (net, 1 bp) | +3.54%/yr | **0.40** | 21% | 8.0 (one-way × NAV) |
| Buy-and-hold SPY | +10.15%/yr | **0.54** | 100% | 0 |

The overlay does **worse** than simply holding the index, on both return and risk-adjusted
return, while parking capital in cash 79% of the year. There is no edge to harvest — the
in-window days are merely *some* of the up-drift days, not a separable, tradable tide.

## Synthetic positive control — the harness is a faithful detector

On a deterministic synthetic tape the same calendar contrast recovers a planted in-window
tide and reads insignificant when none is planted:

| Planted tide | diff (bps/day) | HAC *t* | detected? |
|---|--:|--:|:--:|
| 0 bps/day (pure random walk) | −0.71 | −0.27 | no (correct null) |
| +12 bps/day (strong tide) | +11.29 | +4.24 | yes |

The machinery clears the bar when there really is an in-window tide. On the real SPY tape
it does not — so the *t* = +1.27 is a property of the market, not of a blind pipeline.

## Verdict

- **Signal — NONE.** In-window drift exceeds out-of-window drift by +3.59 bps/day, but the
  HAC *t* is only +1.27 and the block-bootstrap CI [−1.67, +9.17] straddles zero; no single
  window clears *t* = 2. Indistinguishable from the equity drift you were already paid for.
- **Tradability — MIRAGE.** The in-window overlay earns +3.5%/yr at Sharpe 0.40 — *below*
  buy-and-hold's +10.2%/yr at Sharpe 0.54 — while sitting in cash 79% of the year. Even if
  the tiny gap were real, isolating it under-performs the thing it's carved out of.
