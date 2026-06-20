# Results — Study 335 (Buzz-Sentiment-ETF) on the real tape

*The VanEck Social Sentiment ETF (**BUZZ**) — a 75-stock long-only US-equity basket
built from an AI read of social media — raced against **SPY**, both as total-return
close levels rebased to 100 at the first common date. The inferential heart is a
**CAPM alpha**: regress BUZZ's excess return on SPY's, the intercept is the
skill the wrapper is supposed to add, reported with a Newey-West (HAC) t and a
circular block-bootstrap CI. Sharpe is **excess-of-cash vs excess-of-cash**.
Generated offline from the cached daily parquets; re-fetch with
`data.load_real(fetch=True)` and match the fingerprint to confirm the same tape.*

## Data stamp

| Pair | Window | Days | Fingerprint |
|---|---|--:|---|
| BUZZ + SPY (common index) | 2021-03-04 → **2026-05-29** | 1,316 | `52101bc91bce` |

As-of **2026-06-18** (cache pull). The pinned run **drops the partial month** of
June 2026 (the live tape ran to 2026-06-18); the untrimmed tape's fingerprint is
`fc7a3191009b` and moves the numbers by <0.4%/yr — included here only so a
re-runner can tell the two apart.

## The headline — does the AI basket beat the market it listens to?

| Leg | Total return | CAGR | Vol | Sharpe (ex-cash) | Max drawdown |
|---|--:|--:|--:|--:|--:|
| **BUZZ** | +74.8% | **+11.3%/yr** | 32.7% | **0.49** | **−56.9%** |
| **SPY** | +115.7% | **+15.9%/yr** | 16.9% | **0.96** | **−24.5%** |

BUZZ does **not** beat the market. Over its full live history it returned ~5 pts/yr
**less** than SPY, with **double the volatility**, **half the Sharpe**, and a
drawdown **2.3× as deep**. The "AI that reads social media to pick winners" pitch
delivered a worse risk-adjusted outcome than a plain index fund.

## The inference — CAPM alpha (the skill the wrapper claims)

| Quantity | Value |
|---|--:|
| Beta on SPY | **1.59** |
| Alpha | **−9.7%/yr** |
| HAC (Newey-West) *t* on alpha | **−1.26** |
| Block-bootstrap 95% CI on alpha (%/yr) | **[−24.5, +4.8]** |
| Bootstrap P(alpha > 0) | **9.2%** |
| R² (BUZZ ~ SPY) | 0.68 |

- The point estimate of alpha is **negative** (−9.7%/yr), but the HAC *t* is
  **−1.26** — below |t| = 2, so we **cannot even certify it is significantly
  *negative***. What we *can* say with confidence: there is **no positive alpha**.
  The bootstrap puts only a **9%** chance on alpha > 0, and the CI's whole mass
  sits at or below zero.
- The high beta (**1.59**) is the entire story of BUZZ's return shape: it is a
  **leveraged, concentrated bet on the same market**, not a source of skill. In a
  bull tape that beta out-runs SPY on up days and gives it all back (and more) on
  down days — hence the −57% drawdown.

## Active return — the information ratio is ~zero

| Quantity | Value |
|---|--:|
| Mean active (BUZZ − SPY) return | −0.05 bps/day |
| HAC *t* on the active return | **−0.01** |
| Information ratio (ann.) | **−0.005** |

The BUZZ-minus-SPY active return is statistically indistinguishable from zero (a
high-beta basket that nets to no active edge). There is no alpha to harvest, in
either direction, once you account for the market exposure.

## A timing overlay can't rescue it either

Owning whichever leg (BUZZ or SPY) has the stronger trailing 20-day return,
re-checked daily, signal known at *t*'s close earning *t+1* (one lag), 5 bps
one-way switching cost:

| Strategy | Sharpe (net) |
|---|--:|
| Buy & hold SPY | **0.96** |
| 20-day momentum switch (net) | 0.52 |
| Buy & hold BUZZ | 0.49 |

The overlay (135 switches over the sample) does not beat simply holding SPY — it
just spends costs to land between the two legs.

## Synthetic controls — the harness is a faithful alpha detector

The offline engine recovers a planted alpha **only when one is present**, and
returns "no alpha" on a dressed-up-beta null — so the negative real result is a
statement about **BUZZ**, not an artefact of the test.

| Synthetic tape (n = 3000 days) | planted alpha | recovered alpha *t* | reads as |
|---|--:|--:|:--:|
| Null (alpha = 0, beta 1.15, fee 75 bps/yr) | 0%/yr | **−0.61** | no alpha ✓ |
| Positive control (alpha = 4 bps/day) | +10.1%/yr | **+2.99** | alpha ✓ |

The null tape's point estimate is below the bar (|t| = 0.61) — the harness does
not manufacture significance from a pure beta-plus-fee construction. The positive
control clears |t| = 2, so the engine *can* bank a real edge when one exists.
(A synthetic control is a machinery proof, never market evidence — the Signal
stamp is earned on the real tape alone.)

## Verdict

- **Signal — NONE.** Real-tape alpha is −9.7%/yr at HAC *t* = −1.26 (bootstrap
  P(alpha > 0) = 9%); the active return's IR is ≈ 0. There is no positive,
  statistically real alpha in the AI-sentiment wrapper — the inference bar
  (|t| ≥ 2 *for* an effect) is not cleared in the favourable direction.
- **Tradability — MIRAGE.** BUZZ underperformed SPY by ~5%/yr with double the
  vol, half the Sharpe and a −57% drawdown; it is dressed-up beta (β = 1.59) plus
  a fee, not an edge. A timing overlay can't fix it.
- **"AI social-sentiment beats the market" — BUSTED.** The packaged product
  delivered worse risk-adjusted returns than the index it implicitly races. The
  marketing sold the narrative (an AI reading the crowd); the tape sold a
  high-beta closet-index fund that lost to SPY.
