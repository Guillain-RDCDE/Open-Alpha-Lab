# Results — Study 561 (ETF-Flow-Momentum): does chasing ETF inflows pay, on a synthetic tape

*Generated from [`etf_flow_momentum/`](../etf_flow_momentum/). **Synthetic-only study** — there is
no free, no-key, point-in-time feed of ETF creation-unit flows (see the [data note](#why-synthetic-only)),
so this study builds a deterministic synthetic ETF-flow panel and never touches a real tape. The
panel is a 16-ETF × 120-month cross-section (seed 561). Two worlds share one schema: the **null**
(no flow→return relation, `flow_alpha = 0`, panel fp `9cd505b99a53`) and a **literature-modest
planted edge** (`flow_alpha = 0.005`, panel fp `dca0e2e0ba52`). As-of **2026-06-30**.*

## The verdict, earned — Signal `WEAK` · Tradability `MIRAGE`

The claim (flow-based momentum): sector/asset ETFs pulling in the most *new money* — net
creation-unit inflows — keep outperforming. The reversal/crowding view says the opposite: heavy
inflows mark over-extended sectors, so chasing flows is a trap. Both have academic support; the
*sign* is genuinely contested (Ben-Rephael-Kandel-Wohl flow-return persistence vs. Frazzini-Lamont
"dumb money" and Ben-David-Franzoni-Moussawi ETF-crowding reversal).

We cannot settle it on a real tape, because **the real data a retail stack can reach cannot
reconstruct honest ETF flows** — yfinance exposes a single stale `sharesOutstanding` scalar per
ETF, not the daily shares-outstanding history creation-unit flows require. So this is a
**synthetic-only** study: the machinery is proven on a deterministic panel, and per the desk's
rubric a synthetic-only study can never earn `REAL` (that needs a robust *t* ≥ 2 on a real tape).
The literature genuinely supports *a* flow-return relation, but with no real-tape certification and
a live sign dispute, that reads **`WEAK`** on the signal axis; and because you cannot cheaply
*measure* the signal at all, it is `MIRAGE` on tradability.

## The honest baseline — the null world (no planted edge)

When there is **no** flow→return relation (`flow_alpha = 0`), the engine correctly finds nothing:

| | value |
|---|---|
| Long-inflow / short-outflow spread (annualised) | **+2.88%** |
| One-sample *t* on the monthly spread (120 months) | **+0.78** |
| Monthly hit rate | **51.7%** (≈ coin) |
| Label-shuffle placebo *p* | **0.418** |
| Pooled within-month slope *t* (flow → next-month return) | **+0.32** |
| corr(flow, next-month return) | **+0.008** |

This is what "chasing flows is a trap / does nothing" looks like: a spread indistinguishable from
zero, a placebo *p* near ½, no firm-level slope. It is the null the real world *might* be — and we
have no real tape to rule it out.

## The literature-modest planted edge (illustration only)

If we plant a small, realistic flow-momentum edge (`flow_alpha = 0.005` — high-inflow ETFs earn a
modest premium next month), the engine recovers it:

| | value |
|---|---|
| Spread (annualised, gross) | **+15.99%** |
| One-sample *t* (120 months) | **+4.35** |
| Monthly hit rate | **64.2%** |
| Label-shuffle placebo *p* | **0.0005** |
| Pooled slope *t* | **+4.21** (corr +0.11) |
| Net (3 bps/leg monthly round-trip + 40 bps/yr borrow) | **+14.15%** |

This is an **illustration of a planted world, never evidence for the tape** — a synthetic panel
with an effect switched on will of course be significant. It shows only that the detector works and
that ETF frictions (cheap, liquid) would leave a *real* edge, if one existed, largely intact
(gross +16.0% → net +14.2%).

## Robustness — leg fraction (planted world)

| Leg fraction | Spread (ann.) | *t* | Hit |
|---|---|---|---|
| top/bottom 25% | +17.1% | +4.04 | 62.5% |
| top/bottom 33% | +16.0% | +4.35 | 64.2% |
| top/bottom 50% | +8.8% | +3.39 | 62.5% |

Sign and significance are stable across leg fractions in the planted world (they shrink toward 50%
as the legs dilute, as expected).

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

The house rule: average the spread *t* over ≥ 20 seeds so no single lucky seed manufactures a
result. The knob plants either sign of the flow effect:

| Planted `flow_alpha` | Mean spread-*t* (25 seeds) | Reads as |
|---|---|---|
| −0.008 | **−5.87** | strong reversal (the trap) |
| −0.005 | −3.57 | reversal |
| **0.000 (null)** | **+0.25** | flat — no false signal |
| +0.003 | +2.54 | weak momentum |
| +0.005 | +4.07 | momentum |
| +0.008 | +6.36 | strong momentum |

At the null the mean *t* is ≈ 0; planting momentum drives it positive and planting the reversal
trap drives it negative — **the engine catches both signs and stays flat at zero**. So the verdict
is a statement about the *evidence available* (synthetic-only, no real tape), not a broken detector.

## Why synthetic-only

Honest ETF creation-unit flow = **daily change in shares outstanding × NAV**. That needs a clean
*daily shares-outstanding history* per ETF, which is a paid vendor field (Bloomberg / FactSet /
issuer files). yfinance's `sharesOutstanding` for an ETF is a single stale scalar, not a daily
series; a `price × Δshares` proxy hacked from it would smuggle in staleness and look-ahead — exactly
what a flow study must avoid. Rather than publish a dishonest "real" number, this study is
synthetic-only and says so on the SIGNAL axis (like the desk's
[lego-returns](../../273-lego-returns/), [whisky-cask](../../275-whisky-cask/) and
[sneaker-resale](../../276-sneaker-resale/) studies).

## The honest takeaway

Flow momentum has real academic support, but it competes head-on with a well-documented reversal /
crowding effect, and **no free real tape can measure ETF flows cleanly enough to settle the sign.**
On the synthetic null the flow spread is a coin (*t* +0.78, placebo *p* 0.42); the engine provably
catches a planted edge of either sign (control: +0.25 at the null, up to ±6 as the effect grows).
`WEAK` (literature says maybe-real; no real-tape certification; sign disputed) × `MIRAGE` (you
cannot cheaply measure the signal to trade it).
