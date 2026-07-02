# Results — Study 575 (CDS-Equity-Basis): the cross-asset credit-vs-equity gap

*Generated from [`cds_equity_basis/`](../cds_equity_basis/). **There is no real tape**: single-name
CDS spreads live on an OTC, licensed market (Markit/IHS, Bloomberg) with no free retail feed, so
this study is **synthetic-only** and the effect **cannot earn a REAL stamp** (that needs a robust
*t* ≥ 2 on a real tape). Every number below is from the deterministic synthetic world — a
name-by-month panel of the CDS-equity **basis** and forward equity returns — seeded at 575. The
headline single-seed panel is 60 names × 95 months (5,700 name-months, fingerprint `884a4f7a09c4`);
the null panel fingerprint is `4b6557ac1c53`. As-of **2026-06-30**.*

## The verdict, earned — Signal `WEAK` · Tradability `MIRAGE`

The claim (Kapadia-Pu 2012; the credit-equity convergence folklore): the **CDS-equity basis** —
a name's CDS spread minus its equity-implied structural default spread — predicts the firm's
forward equity return. A *wide* basis (credit more worried than the stock) is supposed to forecast
a *falling* equity (it catches down to credit), i.e. a **negative** basis→return relation; the
tradable expression longs the low-basis (credit-complacent) names and shorts the high-basis
(credit-worried) ones each month.

The engine is a **faithful detector** of that effect on the synthetic tape, and the literature for
the effect is genuine — hence `WEAK`, not `NONE`. But there is **no real tape** on a no-key stack
to certify it (single-name CDS is licensed OTC data), so it can go no higher than `WEAK`. And on
tradability it is `MIRAGE`: even the *inputs* are unreachable retail, the short leg is exactly the
distressed, expensive-to-borrow tail, and the documented convergence effect is known to be small,
noisy and largely arbitraged away in liquid names.

## Data stamp — synthetic-only (no real CDS feed)

- **Synthetic panel** (seed 575, planted `convergence_beta = -0.90`): 60 names × 95 months = 5,700
  name-months, columns `cds_bp`, `eq_impl_bp`, `basis_bp`, `forward_ret`; fingerprint `884a4f7a09c4`.
- **Null panel** (seed 575, `convergence_beta = 0`): fingerprint `4b6557ac1c53`.
- **Real panel**: **absent by design** — `fetch_panel()` returns an empty frame; there is no free
  single-name CDS tape.

## The pooled (panel) regression — the sign IS the claim

Forward equity return on the per-month z-scored basis, pooled across all 5,700 name-months, with a
standard error **clustered by month** (so the cross-sectional correlation within a month can't
inflate the *t*):

| | value |
|---|---|
| Slope (forward_ret on basis-z) | **−0.00461** per basis-z unit |
| Slope *t* (clustered by month) | **−5.79** (negative = the convergence sign) |
| corr(basis, forward return) | **−0.082** |
| n (name-months) | 5,700 |

A *negative* slope is the folklore convergence sign: a wider basis (credit worried) predicts a
lower forward equity return. The correlation is *tiny* (−0.08) even at a healthy planted effect —
the basis is a noisy signal swamped by idiosyncratic equity moves, which is exactly why the real
effect (when it exists) is faint and hard to harvest.

## The decile long-short

Each month, long the lowest-basis quintile, short the highest, hold one month:

| | value |
|---|---|
| Mean monthly spread (gross) | **+1.18%** / month |
| IID *t* on monthly spreads | **+4.78** (n = 95 months) |
| Month-label-shuffle placebo *p* | **0.0005** |

The placebo (shuffle basis labels vs forward returns within each month, 2,000 perms) puts the
observed spread deep in the tail — the signal is real *in this synthetic world where we planted it*.

## Robustness — the sign holds across sub-samples (engine sanity)

| Sub-sample | Pooled slope-*t* (clustered) |
|---|---|
| Early third (months 0-31) | **−3.12** |
| Mid third (months 32-63) | **−3.45** |
| Late third (months 64-94) | **−3.36** |

The planted convergence sign is stable across the panel — as it must be for a faithful engine.

## Costs — the friction the folklore ignores

| | value |
|---|---|
| Gross mean monthly | **+1.18%** |
| Net (10 bps/leg round-trip + 150 bps/yr borrow, monthly rebalance) | **+0.65%** / month |
| Net annualised | **+7.8%** / yr |

Monthly rebalancing on both legs plus a punitive borrow on the credit-worried short leg eats
roughly *half* the gross spread — and this is on a *frictionless synthetic tape*. On the real
OTC/illiquid instruments the wedge would be far worse.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `convergence_beta` | Mean pooled slope-*t* | Mean long-short *t* | |
|---|---|---|---|
| 0.00 (null) | **−0.23** | **+0.19** | flat — no false signal |
| −0.30 | −2.95 | +2.60 | effect emerging |
| −0.60 | −5.64 | +4.98 | effect visible |
| −0.90 (headline) | **−8.27** | **+7.31** | clears the bar |

At the null both statistics sit at ≈ 0 (averaged over 25 seeds — a single lucky seed can print a
spurious *t* ≈ ±1.9, which is *precisely* why the house rule averages ≥ 20 seeds). Planting a
genuine convergence effect drives the slope negative and the long-short positive, monotonically.
The detector works — so the ceiling here is set by **data availability, not by the engine**. (This
is a control; it can never be cited as evidence for a real-world stamp.)

## Why this can never certify REAL here

1. **No real tape.** Single-name 5-year CDS is an OTC, licensed market (Markit/IHS, Bloomberg
   CDSW). There is no free, no-key retail feed; a structural equity-implied spread additionally
   needs point-in-time liabilities. A synthetic-only study **cannot** clear the REAL bar (a robust
   *t* ≥ 2 on a *real* tape) — this is the SIGNAL-axis data limitation, stated openly.
2. **A faint, arbitraged effect even when real.** The literature (Kapadia-Pu 2012) finds
   CDS-equity convergence is *statistically* present but *economically small* and concentrated in
   frictional, hard-to-arbitrage names — the corr here (−0.08) is deliberately faithful to that
   faintness. In liquid names the basis is largely closed by capital-structure arbitrageurs.
3. **Unreachable, expensive frictions.** The short leg is the distressed, wide-CDS tail — the
   costliest to borrow — and CDS itself carries bid/ask, funding and roll costs the folklore
   pitch never charges. `MIRAGE` on tradability.

## The honest takeaway

The CDS-equity basis convergence effect is **real in the literature** and the engine here is a
**faithful detector** of it — so `WEAK`, not `NONE`. But it can go no higher: there is **no free
real CDS tape** to certify it, the true effect is faint and largely arbitraged, and the trade needs
data and a short-borrow a retail desk simply cannot reach — `MIRAGE`. The synthetic control proves
the machinery would catch the effect the day someone hands it a real, licensed CDS panel.
