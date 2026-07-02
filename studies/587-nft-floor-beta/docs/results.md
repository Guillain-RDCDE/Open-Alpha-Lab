# Results — Study 587 (NFT-Floor-Beta): does NFT floor momentum *lead* ETH, or just *follow* it?

*Generated from [`nft_floor_beta/`](../nft_floor_beta/) on the **deterministic synthetic world** in
[`nft_floor_beta/data.py`](../nft_floor_beta/data.py) (seed **587**, `lead_alpha = 0` — the null:
NFT floors are lagged high-beta ETH noise). Daily panel of **1500 days**, series fingerprint
`01ef0bf578c2`. Signal = 14-day NFT floor-index momentum; target = forward ETH return over a 5-day
window. As-of **2026-06-30**.*

> **DATA-AVAILABILITY LIMITATION (named on the SIGNAL axis).** No clean, free, no-key retail tape
> exists for blue-chip NFT floor prices: floors live behind rate-limited, key-gated marketplace /
> aggregator APIs (OpenSea, Blur, Reservoir, NFTGo), are ETH-denominated (so they inherit ETH beta
> mechanically), wash-traded, survivorship-ridden, and only a few years long. This study is
> therefore **synthetic-only**. Per the desk rule a synthetic-only study can **never** earn a REAL
> signal (that needs a robust *t* ≥ 2 on a REAL tape) — it is capped at `WEAK`/`NONE`. The engine
> here is validated as a faithful *detector* on a planted lead; it makes no claim about the real
> NFT tape.

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "NFT floor leads crypto?" `BUSTED`

The claim: blue-chip NFT floor-price **momentum** is a crypto risk-appetite signal that **leads**
ETH. The sceptical prior (the study's expected lean): floors are a **lagged, high-beta echo** of
ETH — they *follow* the market and amplify it with heavy idiosyncratic noise, so floor momentum has
**no predictive edge** on forward ETH returns. We build that null world explicitly and test whether
floor momentum forecasts forward ETH.

On the synthetic null world, floor momentum has **no predictive power** over forward ETH returns:
the predictive slope's Newey-West HAC *t* is **−0.47** (plain OLS *t* −0.87), R² = **0.0005**,
placebo *p* = **0.73**. Controlling for ETH's own trailing momentum leaves it at HAC *t* **+0.41**.
Across five forward horizons the |HAC *t*| never clears **1.5** raw. Meanwhile the **follow**
relationship is overwhelming: the floor index regressed on *lagged* ETH has beta **1.61** (*t*
**49.7**, corr **0.79**), and the lead-lag cross-correlation peaks at lag **−1** (floors correlate
0.79 with *yesterday's* ETH) and is ~0 at every positive lag. Floors **follow ETH by a day and
amplify it** — the textbook high-beta-noise story, with zero lead. The engine is not blind: planting
a genuine lead (`lead_alpha > 0`) drives the mean HAC *t* past **+2** at `lead_alpha = 0.10` and to
**+8.8** at `0.30`, while the null stays flat at **−0.17** (25 seeds each).

So `NONE` on the signal axis (no lead in the null world, and synthetic-only can't earn REAL),
`MIRAGE` on tradability (a floor-momentum long/flat ETH overlay is **gross +1.4%/yr → net −3.1%/yr**
after 10 bps costs, Sharpe **−0.05**), and `BUSTED` on the myth-check ("NFT floor leads crypto?" —
it follows, it does not lead).

## Data stamp

| Field | Value |
|---|---|
| World | synthetic, deterministic, offline (seed 587, `lead_alpha = 0`) |
| Panel | 1500 daily rows (ETH return, NFT floor return, 14-day floor momentum) |
| Series fingerprint | `01ef0bf578c2` |
| Signal | 14-day trailing NFT floor-index momentum |
| Target | forward ETH return over (t, t+5] |
| As-of | 2026-06-30 |

## Floors FOLLOW ETH — the high-beta-lagged echo

| Measure | Value | Reads as |
|---|--:|---|
| Beta of floor on **lagged** ETH (t−1) | **1.61** (*t* 49.7) | floors amplify *yesterday's* ETH 1.6× |
| corr(floor, lagged ETH) | **0.79** | strong follow relationship |
| Lead-lag corr peak | at lag **−1** (0.79) | floors correlate with *past* ETH |
| Lead-lag corr at positive lags (+1..+5) | ≈ 0.00–0.05 | **no** lead of floor over future ETH |

The whole signal in floors is *lagged ETH*: they move a day *after* ETH and swing harder. That is
exactly "high-beta noise", not a leading indicator.

## The predictive test — floor momentum → forward ETH (the LEAD)

| Measure | Value |
|---|--:|
| Predictive slope (floor mom → forward ETH) | **−0.095** |
| Newey-West HAC *t* | **−0.47** |
| Plain OLS *t* | **−0.87** |
| R² | **0.0005** |
| HAC *t* controlling for ETH's own momentum | **+0.41** |
| Block-shuffle placebo *p* | **0.73** |

The bar for a signal — a predictive HAC *t* ≥ 2 the right way — is missed by a mile, in both the raw
and the ETH-momentum-controlled regressions, and the placebo confirms it is noise (*p* = 0.73).

## Robustness — the horizon sweep

| Forward horizon (days) | HAC *t* (raw) | HAC *t* (control ETH mom) |
|---|--:|--:|
| 1 | +0.22 | +0.54 |
| 3 | +0.04 | +0.92 |
| 5 (headline) | **−0.47** | +0.41 |
| 10 | −0.58 | +0.89 |
| 21 | +0.11 | +1.42 |

No horizon delivers a predictive |HAC *t*| ≥ 1.5, and the sign wanders around zero. There is no lead
to harvest at any horizon.

## Tradability — a floor-momentum ETH overlay

| Measure | Value |
|---|--:|
| Rule | long ETH when 14-day floor momentum > 0, else flat (enter next day) |
| Fraction of days long | 49.4% |
| Turnover (avg |Δposition|/day) | 0.124 |
| **Gross** annualised return | **+1.4%** (1.45%/yr) |
| **Net** (10 bps one-way × turnover) | **−3.1%** |
| Net Sharpe | **−0.05** |

The overlay is barely positive gross (it is just partial ETH beta, timed by noise) and **negative
net** once you pay 10 bps per position change. Nothing to trade. `MIRAGE`.

## Synthetic positive control — the engine is a faithful detector (25 seeds each)

| Planted `lead_alpha` | Mean predictive HAC *t* (25 seeds) | Reads as |
|---|--:|---|
| 0.00 (null) | **−0.17** | flat — no false signal |
| 0.05 | +1.03 | lead emerging |
| 0.10 | **+2.30** | clears the *t* = 2 bar |
| 0.20 | +5.19 | strong |
| 0.30 | +8.84 | very strong |

At the null the HAC *t* is ≈ 0; planting a genuine lead (floor momentum informing *future* ETH)
drives it monotonically past +2. So the flat null result is the tape talking, not a broken engine.
*(Control only; a synthetic detector can never certify a REAL signal — the real NFT tape is not
reachable on a free stack.)*

## Why "NFT floor leads crypto" does not certify here

1. **Floors follow, by construction and in the data.** The floor index is ETH-denominated and
   liquidity-driven; it moves *after* ETH (lag −1 corr 0.79) and amplifies it (beta 1.61). A lagged
   high-beta echo cannot *lead* the thing it echoes.
2. **The predictive edge is zero at the null.** With no planted lead, floor momentum forecasts
   forward ETH at HAC *t* −0.47 (placebo *p* 0.73) — indistinguishable from noise, before and after
   controlling for ETH's own momentum.
3. **Synthetic-only, by data availability.** The real blue-chip-NFT floor tape is key-gated,
   wash-traded and survivorship-ridden; no free reconstruction is trustworthy. The signal axis is
   capped at `NONE`/`WEAK` regardless of how clean the synthetic result looks.

## The honest takeaway

NFT floor prices are a **lagged, high-beta echo of ETH**, not a leading risk-appetite signal:
they follow ETH by a day (beta 1.61, corr 0.79 at lag −1) and their momentum forecasts forward ETH
at HAC *t* −0.47 (placebo *p* 0.73), with a floor-momentum ETH overlay that is **gross +1.4%/yr →
net −3.1%/yr**. `NONE` × `MIRAGE`, myth **BUSTED**. The synthetic control proves the engine would
catch a real lead if one existed (HAC *t* past +2 once planted) — so this is a statement about the
economics of a floor index, not a broken detector. And because no clean free NFT floor tape exists,
the signal axis is capped below REAL on data availability alone.
