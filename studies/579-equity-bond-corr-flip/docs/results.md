# Results — Study 579 (Equity-Bond-Corr-Flip): the stock-bond correlation sign as a 60/40 timer

*Generated from [`equity_bond_corr_flip/`](../equity_bond_corr_flip/) over this study's cached
yfinance tape: daily adjusted close for **SPY** (equity leg) and **TLT** (20+yr Treasury leg),
2002-07-30 → 2026-06-26 (price fingerprint `884aad7ef4eb`). Reduced to **286 complete monthly
observations** (2002-08 → 2026-05; the partial June-2026 month is dropped), monthly-panel
fingerprint `55f8c175e893`. The regime signal is the sign of the trailing 6-month SPY/TLT return
correlation; the test compares the **forward one-month** 60/40 return conditional on that sign.
As-of **2026-06-30**.*

## The verdict, earned — Signal `WEAK` · Tradability `FRAGILE`

The folklore: the stock-bond correlation went **negative** for two decades (bonds hedged equity
drawdowns, so 60/40 diversification worked), then **flipped positive in 2022** (stocks and bonds
fell together, and 60/40 had its worst year in a generation). The claim under test: *the sign of
the trailing correlation is a timing signal* — when it flips positive, cut the 60/40 because the
hedge has stopped working.

On the real tape the effect has the **right sign but does not clear the bar.** Forward 60/40
returns average **+0.885%/mo** after negative-correlation months versus **+0.566%/mo** after
positive-correlation months — a spread of **+0.319%/mo** in the folklore's predicted direction, but
the two-sample *t* is only **+0.89** (placebo *p* = **0.36**), far below the *t* ≥ 2 inference bar,
and the sign **flips across sub-windows** (−0.30%/mo in 2002-09, +0.61%/mo in 2010-19). So `WEAK`
on the signal axis: the direction matches the story and the 2022 flip is a real macro event the
signal *did* flag, but this tape alone cannot certify a bankable edge.

`FRAGILE` on tradability: a rules timer that de-risks the 60/40 when the correlation is positive
**cuts risk but gives up return** — max drawdown falls **−28.5% → −23.6%** and annualised vol
**9.9% → 7.1%**, and the risk-adjusted Sharpe nudges up (**0.88 → 0.97**), but the net annualised
return *edge* is **−1.84%/yr** (a scaled-down book earns less in a rising market). It is a
defensible *risk overlay*, not an alpha source — and the whole "flip" it hangs on is **one** macro
regime change in 24 years (only 4 negative-correlation months in the entire 2022-onward window),
so the effective sample behind the headline claim is tiny.

## Data stamp

- **Prices**: SPY + TLT, daily adjusted close, 2002-07-30 → 2026-06-26, fingerprint `884aad7ef4eb`
- **Monthly panel** (equity/bond returns, trailing-6m correlation sign, forward 60/40 & equity
  returns): 286 complete months 2002-08 → 2026-05, fingerprint `55f8c175e893`

## The regime split — right sign, sub-threshold *t*

| Forward one-month 60/40 return, conditioned on the trailing-corr sign | mean | n months |
|---|---|---|
| After a **negative**-correlation month (hedge "working") | **+0.885%/mo** | 158 |
| After a **positive**-correlation month (hedge "broken") | **+0.566%/mo** | 123 |
| **Spread (neg − pos)** — the claim predicts > 0 | **+0.319%/mo** | — |

Two-sample (Welch) *t* = **+0.89**, label-shuffle placebo *p* = **0.36**. The sign is the one the
folklore predicts (60/40 does better when bonds are hedging), but the gap is inside the noise: at
*t* 0.89 you cannot reject "the correlation sign tells you nothing about next month's 60/40."

The **equity-leg** split is weaker still: +1.089%/mo (neg) vs +0.899%/mo (pos), spread +0.19%/mo,
*t* **+0.37**, placebo *p* = 0.72 — the regime signal says even less about forward *equity* returns
than about the 60/40 blend.

## Robustness — the sign is not stable across the sample

| Window | Spread (neg − pos) %/mo | *t* | n_neg / n_pos | Reads as |
|---|---|---|---|---|
| 2002-2009 | **−0.300** | −0.42 | 55 / 29 | wrong sign |
| 2010-2019 | **+0.613** | +1.49 | 87 / 33 | folklore sign (weak) |
| 2020-2026 | **+0.636** | +0.60 | 16 / 61 | folklore sign (weak) |
| pre-2022 flip (→ 2021-12) | **+0.174** | +0.47 | 154 / 74 | folklore sign (flat) |
| 2022-onward | **+1.257** | +1.20 | 4 / 49 | folklore sign — but only **4** neg months |

The spread is *negative* in the 2002-09 sub-period and positive afterward, and no window clears
*t* = 2. The 2022-onward window has the largest spread but rests on **4** negative-correlation
months against 49 positive ones — you cannot estimate a regime *contrast* from 4 observations of
one side. A signal whose sign depends on the sub-period, none of which is significant, is `WEAK`.

## The timing overlay — a risk-reducer, not an alpha source

A rules timer holds the static 60/40 when the trailing correlation is negative and **de-risks**
when it is positive (scaling the book to 0.4×, long-only, no borrow), with a 5 bps one-way
switching cost (43 switches over 24 years).

| | Static 60/40 | Corr-timer (de-risk to 0.4×) |
|---|---|---|
| Annualised return | **8.75%** | **6.90%** (net) |
| Annualised vol | **9.9%** | **7.1%** |
| Sharpe | **0.88** | **0.97** |
| Max drawdown | **−28.5%** | **−23.6%** |
| Net annualised return edge | — | **−1.84%/yr** |

The timer does exactly what a de-risking overlay does: it trims the left tail (drawdown −28.5% →
−23.6%, vol 9.9% → 7.1%) and *marginally* improves the risk-adjusted return (Sharpe +0.09), but it
**gives up 1.84%/yr of return** by sitting in a smaller book through a mostly-rising market. A
cash-de-risk variant is worse (net edge −3.01%/yr, Sharpe unchanged). This is `FRAGILE`: a
defensible risk-management tilt, not a source of excess return, and its Sharpe gain is well inside
the noise the *t* = 0.89 split already flagged.

## Did the signal flag 2022?

Yes — and this is the honest steel-man. The trailing correlation first printed **positive in
early 2021** and stayed positive through **all 12 months of 2022** (the year static 60/40 returned
**−14.0%**). A timer following the sign would have been de-risked through the entire 60/40
drawdown — which is exactly why it cut the max drawdown. The catch: the *same* de-risking rule
also sat out much of the 2023-2025 recovery, and across the full 24-year tape the return given up
outweighs the drawdown saved. One correct call on one regime change is not a *t* ≥ 2 edge.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `regime_edge` (fwd-return penalty in positive-corr months) | Mean split-*t* (25 seeds) | |
|---|---|---|
| 0.000 (null) | **−0.12** | flat — no false signal |
| 0.005 | +1.34 | effect emerging |
| 0.010 | **+2.80** | clears the bar |
| 0.020 | +5.73 | unmistakable |

At the null the split-*t* is ≈ 0; planting a genuine regime effect (forward 60/40 worse after
positive-correlation months) drives the *t* positive and past +2 as it grows. The detector works —
so the sub-2 real-tape *t* is a statement about **the data**, not a broken engine. (Control only;
never cited for the real-tape stamp.)

## The honest takeaway

The stock-bond correlation regime is *economically real* — the 2022 flip genuinely broke the 60/40
hedge, and the signal would have flagged it. But as a **timing rule** on the real tape the edge is
`WEAK` (right sign, *t* = 0.89, placebo *p* = 0.36, sign flips across sub-periods) and its tradable
expression is `FRAGILE` (a de-risking overlay that cuts drawdown −28.5% → −23.6% and vol 9.9% →
7.1% but *costs* −1.84%/yr of return, with the whole claim resting on a single 24-year regime
change and just 4 negative-correlation months post-2022). The synthetic control confirms the
engine would catch a real regime effect — so this is the tape talking, not the code.
