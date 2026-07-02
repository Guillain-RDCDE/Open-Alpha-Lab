# Results — Study 585 (Perp-Funding-Rate): the funding-rate contrarian signal, on a synthetic tape

*Generated from [`perp_funding_rate/`](../perp_funding_rate/). The reproducible core is
**synthetic** — a deterministic, seeded funding-rate + forward-return panel (seed 585, 2000
funding periods, planted `contrarian_beta = -0.012`, panel fingerprint `8121108453d2`). Free
retail data (yfinance) reaches BTC/ETH **prices** (cached tape `db49481bb5f1`, 2018-01-01 →
2026-06-29, 3102 daily closes) but **not** the historical funding-rate tape, so there is no real
funding series to test — the data-availability limitation is on the SIGNAL axis. As-of
**2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE`

The folklore: a crypto perpetual swap is tethered to spot by a periodic **funding rate** paid
between longs and shorts; when funding runs **hot** (large positive, longs paying shorts) the crowd
is long-and-levered and a mean-reversion flush is due — so extreme funding is a **contrarian**
signal for the forward return. It is a real, plausible *positioning* story with desk support.

But this desk's bar for `REAL` is a robust *t* ≥ 2 **on a real tape**, and the real tape here does
not exist for a no-key stack: **funding history is paid/rate-limited exchange data** (Binance/Bybit
`fapi`, Coinglass, Amberdata, Kaiko). We can build and validate the *machinery* on a deterministic
synthetic world — and we do, cleanly — but a synthetic-only study can never earn `REAL`. So the
Signal axis is `NONE` (no real-tape confirmation; literature/folklore support alone would be `WEAK`,
and the sign is *not* horizon-fragile in the synthetic control, but there is simply no real number
to certify). Tradability is `MIRAGE`: even if the signal were real, a per-funding-period rebalance
(every 8h) with the short leg *paying* the very funding it keys on, plus taker fees on the most
turnover-heavy trade on the desk, eats the edge. On the synthetic tape the +3.01% gross
cold-minus-hot spread nets **+2.47%** — but that is a *planted* effect, not a market fact.

## Data stamp

- **Synthetic panel** (the reproducible core): seed 585, 2000 funding periods, AR(1) funding
  (ρ = 0.85), planted `contrarian_beta = -0.012`, horizon = 3 funding periods (~1 day),
  fingerprint `8121108453d2`.
- **BTC/ETH price tape** (illustrative only; funding is *not* free): daily adjusted close,
  2018-01-01 → 2026-06-29, 3102 rows, fingerprint `db49481bb5f1`. Used only to show the price tape
  the funding rate would tether to — it yields **no** funding series and cannot lift the Signal.

## The contrarian sort — on the synthetic (planted-effect) tape

| Quintile tail (400 periods) | Forward return (~1 day, 3 funding periods) |
|---|---|
| **Cold funding** (lowest funding-z: capitulated shorts) | **+1.40%** |
| **Hot funding** (highest funding-z: crowded longs) | **−1.60%** |
| **Spread (cold − hot)** | **+3.01%** (two-sample *t* **+8.08**) |

The contrarian claim predicts cold > hot (a *positive* spread): the panic-short lows bounce, the
crowded-long highs get flushed. On the planted synthetic tape the spread is a clean **+3.01%** with
a label-shuffle placebo *p* = **0.0005** — this is what a *real* funding contrarian signal would
look like. It is a control, not a market result.

## The period-level relation

| | value |
|---|---|
| Slope (fwd_ret on standardized funding) | **−0.0111** per funding-z unit |
| Slope *t* | **−9.44** (a *negative* slope IS the contrarian effect) |
| corr(funding, forward return) | **−0.21** |

A negative slope is the contrarian effect: hotter funding, lower forward return — by construction on
the synthetic tape.

## Robustness — the sign is stable across horizons (synthetic control)

| Forward horizon | Cold − hot spread | Slope *t* | Reads as |
|---|---|---|---|
| 1 funding period (~8h) | **+2.88%** | **−15.58** | contrarian present |
| 3 funding periods (~1 day, headline) | **+3.01%** | **−9.44** | contrarian present |
| 6 funding periods (~2 days) | **+3.61%** | **−8.07** | contrarian present |
| 9 funding periods (~3 days) | **+3.10%** | **−6.26** | contrarian present |

On the synthetic tape the planted effect survives every horizon (sign stable, *t* well past 2) —
so the engine is not horizon-fragile. This proves the *detector*, not the *market*.

## Costs — the tradability wall

| | value |
|---|---|
| Gross cold − hot spread (synthetic, per rebalance) | **+3.01%** |
| Net (6 bps/leg round-trip + 30 bps short-side funding drag) | **+2.47%** |

Costs are charged per funding-period rebalance. The honest problem is structural, not the size of
this synthetic wedge: the contrarian short leg is the *hot-funding* leg — i.e. you **pay** the
positive funding you are keying on to hold the short — and an 8-hourly reform is the highest-turnover
trade on the desk. On a real (much noisier, non-planted) funding tape the net edge would be a small
difference of large, expensive numbers.

## Null / placebo — the engine stays flat when nothing is planted

| Planted `contrarian_beta` | Single-seed slope *t* | Single-seed cold−hot *t* | Placebo *p* |
|---|---|---|---|
| 0.00 (null) | **+0.75** | −0.98 | **0.344** |

At the null the slope *t* ≈ 0, the tail spread *t* is inside ±2, and the placebo *p* is a coin-flip
0.34 — no false signal. (One seed shown; the seed-robust table below averages 25.)

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `contrarian_beta` | Mean slope *t* (25 seeds) | |
|---|---|---|
| 0.000 (null) | **−0.17** | flat — no false signal |
| −0.006 | −5.34 | contrarian emerging |
| −0.012 (headline) | −10.51 | contrarian visible |
| −0.020 | −17.40 | strong |

At the null the mean slope *t* is ≈ 0 across 25 seeds; planting a genuine contrarian effect
(`contrarian_beta < 0`) drives the slope negative and past −2 as it grows. The detector works — so
the study's limitation is the **missing real funding tape**, not a broken engine. (Control only;
never cited for a real-tape stamp — there is no real-tape stamp.)

## Why this can never certify `REAL` here

1. **No free funding tape.** BTC/ETH *prices* are free (yfinance); the historical **funding rate**
   — the entire signal — is paid or rate-limited exchange/derivatives data. Without it there is no
   real number, and the desk's `REAL` requires a robust *t* ≥ 2 on a **real** tape.
2. **Highest-turnover trade on the desk.** An 8-hourly rebalance means costs and slippage dominate;
   the short leg literally pays the funding it trades on.
3. **Regime and reflexivity.** Real funding is reflexive (everyone watches the same Coinglass
   heatmap) and regime-dependent — a synthetic AR(1) with a clean planted effect cannot speak to
   whether the edge survives crowding and structural breaks on the live tape.

## The honest takeaway

The perp-funding contrarian idea is *plausible and well-liked on crypto desks* — and our synthetic
control shows that **if** the effect were real, this engine would catch it (spread +3.01%, slope *t*
−9.44, stable across horizons, flat at the null). But the reproducible core is synthetic **by
necessity**: the funding-rate tape that would make it a real test is not reachable from free data.
`NONE` × `MIRAGE`, with the data-availability limitation named openly — a synthetic-only study,
capped below `REAL` by the desk's own rules until someone pipes in a paid funding feed.
