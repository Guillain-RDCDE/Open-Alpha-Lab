# Results — Study 572 (Capex-Cycle): the capex-*growth* anomaly on a survivor basket

*Generated from [`capex_cycle/`](../capex_cycle/) over this study's cached yfinance tape: daily
adjusted close for a fixed **large-cap survivor basket** (prices fingerprint `a01c8032c6fc`,
2018-01-02 → 2026-06-26) plus a per-name **cash-flow snapshot** of Capital Expenditure and Total
Assets (fingerprint `8a66058175f0`, fiscal years 2022-2025). The capex-cycle signal is scored
as-of fiscal year **2024** (public by mid-2025); the forward holding window is
**2025-06-27 → 2026-06-26**. Cross-section: **42 names**, fingerprint `5a4b0aebc102`. As-of
**2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Capex cousin of asset growth?" `BUSTED`

The claim (the capex-*growth* cousin of Cooper-Gulen-Schill asset growth and Titman-Wei-Xie
capex level): a firm in a **capex-spending binge** — one *ramping* its capital expenditure
relative to its asset base — should earn lower future returns than one *harvesting* cash. We build
the signal

    capex_intensity_t = |CapEx_t| / TotalAssets_{t-1}
    capex_cycle       = capex_intensity_t - capex_intensity_{t-1}   (a *binge* if > 0)

score it as-of FY2024, and test the cross-sectional information coefficient (IC) between
`capex_cycle` and forward return, plus a long-harvest / short-binge tercile hedge.

On the headline window the effect is a **flat zero — if anything the wrong sign.** The Pearson
**IC is +0.004** (*t* = **+0.03**) — indistinguishable from noise. The tercile hedge (long
harvest, short binge) earns **−10.2%** (two-sample *t* **−0.14**), buried in the shuffle null
(placebo *p* = **0.88**). The bingers *out*-earned the harvesters because the biggest capex bingers
on a survivor basket in 2024-25 are exactly the AI-hyperscaler winners (**AMZN, GOOGL, MSFT, META**)
that ripped. So `NONE` on the signal axis (no *t* ≥ 2, and the sign points the wrong way), `MIRAGE`
on tradability (a survivor basket, snapshot cross-section, sign-unstable, distressed-short borrow),
and `BUSTED` on the "capex cousin of asset growth" myth: like the total-asset-growth channel of
[Study 244](../../244-asset-growth/) and the capex-level channel of
[Study 523](../../523-investment-to-assets/), the capex-*growth* channel vanishes on a survivor
large-cap panel.

## Data stamp

- **Prices**: large-cap survivors + SPY, daily adjusted close, 2018-01-02 → 2026-06-26,
  fingerprint `a01c8032c6fc`
- **Cash-flow snapshot**: |Capital Expenditure| and Total Assets, fiscal years 2022-2025,
  fingerprint `8a66058175f0`
- **Cross-section** (scored FY2024, forward 2025-06-27 → 2026-06-26): 42 names, fingerprint
  `5a4b0aebc102`

## The capex-cycle sort — no edge, faint wrong sign

| Tercile (13 names) | Forward return 2025-06 → 2026-06 |
|---|---|
| **Harvest** (lowest capex_cycle: TXN, TSM, VZ, UPS, INTC, NEE, QCOM, PFE, PEP, AAPL, SLB, MRK, UNP) | **+67.0%** |
| **Binge** (highest capex_cycle: MCD, XOM, T, BA, COP, JNJ, SBUX, D, MU, META, MSFT, GOOGL, AMZN) | **+77.1%** |
| **Spread (harvest − binge)** | **−10.2%** (two-sample *t* −0.14) |

The anomaly predicts harvest > binge (a *positive* spread). The tape delivers a small *negative*
spread with a *t* of −0.14 — no signal, faint wrong sign. The label-shuffle placebo *p* = **0.88**
confirms this is pure noise on this window.

## The headline information coefficient

| | value |
|---|---|
| Pearson IC (capex_cycle, forward return) | **+0.004** |
| IC *t* | **+0.03** (a *negative* IC would be the anomaly) |
| Spearman IC | **−0.21** |

The Pearson IC is a dead zero. The *rank* (Spearman) IC is mildly negative (−0.21) — the only
whiff of the predicted direction — but it is not significant on 42 names and is entirely a
rank-order artefact of a handful of mega-cap bingers that were also mega-winners (Pearson, which
weights those large returns, washes it out to zero). No bankable signal either way.

## Robustness — no stable sign

| Signal (fiscal yr) → forward window | IC | IC *t* | Hedge spread | Reads as |
|---|---|---|---|---|
| FY2024 → 2025-06 → 2026-06 (headline) | **+0.004** | +0.03 | **−10.2%** | flat / faint wrong sign |
| FY2024 → 2025-06 → 2025-12 (6mo) | +0.175 | +1.12 | −2.7% | flat |
| FY2024 → 2025-01 → 2026-01 | +0.146 | +0.94 | −20.4% | flat / wrong sign |
| FY2025 → 2026-02 → 2026-06 (short) | −0.018 | −0.12 | −0.2% | flat |

Across every window the IC-*t* sits inside ±1.2 and the hedge spread flips between small negatives —
never a *t* ≥ 2 in the predicted (negative-IC) direction. A signal this unstable and this weak is
not bankable — `NONE` on the signal axis, `BUSTED` on the myth.

## Costs

| | value |
|---|---|
| Gross spread (harvest − binge, headline window) | **−10.2%** |
| Net (5 bps/leg round-trip + 100 bps/yr borrow, 1y hold) | **−11.4%** |

Costs are a footnote: the trade is the *wrong sign* before you pay for it, and the bingeing leg you
would short is exactly the crowded, expensive-to-borrow AI-capex tail.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `binge_alpha` | Mean IC-*t* (25 seeds) | |
|---|---|---|
| 0.00 (null) | **−0.15** | flat — no false signal |
| −0.10 | −0.94 | effect emerging |
| −0.20 | −1.72 | effect visible |
| −0.30 | **−2.49** | clears the bar |
| −0.50 | **−4.01** | unmistakable |

At the null the IC-*t* is ≈ 0; planting a genuine capex-binge penalty (`binge_alpha < 0`) drives the
IC negative and past −2 as it grows. The detector works — so the real-tape zero is a statement about
**this survivor snapshot on this window**, not a broken engine. (Control only; never cited for the
real-tape stamp.)

## Why the capex-cycle anomaly doesn't certify here

1. **Survivorship, the wrong way.** The basket is names *still large-cap in 2026*. The real
   over-investment effect is driven by aggressive expanders that *subsequently failed*; strip those
   out and the surviving bingers are the *successful* capex-heavy compounders — biasing the tape
   *against* the anomaly, toward an anti-anomaly in an AI-capex boom.
2. **A hyperscaler-capex melt-up window.** 2024-25 rewarded the biggest capex ramps (AMZN, GOOGL,
   MSFT, META building AI datacentres). The over-investment story earns its keep over full cycles,
   not a boom that pays the bingers.
3. **Cash-flow *snapshot*, not point-in-time.** yfinance exposes only ~4-5 annual statements, and a
   capex *cycle* (a change of a ratio) costs two of them — so this is a one-shot cross-section, not
   a deep panel. A true replication needs Compustat/EDGAR history (as
   [Study 523](../../523-investment-to-assets/) uses for the capex *level*). This is why the study is
   **capped below REAL** on the SIGNAL axis regardless of the number.

## The honest takeaway

The capex-cycle anomaly — high capex growth → low future returns, the investment cousin of asset
growth — is a flat zero on a 42-name survivor snapshot over 2025-26: Pearson IC **+0.004**
(*t* +0.03), hedge spread **−10.2%** (*t* −0.14), placebo *p* 0.88, and no stable sign across
windows. `NONE` × `MIRAGE`, myth `BUSTED`. The synthetic control confirms the engine would catch a
real effect — so this is the tape (survivorship + an AI-capex melt-up + a shallow yfinance snapshot)
talking, not the code.
