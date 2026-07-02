# Results — Study 556 (Electricity-Demand): the real-economy power pulse vs equity returns

*Generated from [`electricity_demand/`](../electricity_demand/) over this study's tape: a
hardcoded monthly snapshot of **U.S. total electricity net generation, all sectors (TWh)** —
EIA *Electric Power Monthly* series `ELEC.GEN.ALL-US-99.M` (fingerprint `625867832b4d`,
2007-01 → 2026-06, 234 months) — aligned to cached daily total-return SPY & XLU closes from
yfinance (fingerprint `dc0f80c14e47`). The aligned monthly panel (fingerprint `e7d78ee9af2d`)
runs 2007-01-31 → 2026-06-30. The signal carries a **2-month lag** (1 for the EIA publication
delay, 1 for the signal→return convention). As-of **2026-06-30**.*

## The verdict, earned — Signal `WEAK` · Tradability `MIRAGE`

The macro-alt-data folklore says aggregate **electricity demand is the pulse of the real
economy** — factories, data centres, offices, un-spinnable and printed monthly — so
demand-growth should lead equity (or utility) returns. We build demand-growth momentum
(year-over-year % change of net generation, which strips the huge summer/winter seasonal),
lag it two months so it is strictly public, and test whether hot-demand months precede
stronger forward returns.

The **broad market (SPY) shows nothing**: at 1/3/6-month horizons the hot-minus-cold forward
spread is a rounding error (Welch *t* +0.37, −0.56, −0.25; placebo *p* 0.71, 0.59, 0.80), and
the only "significant" reading — the 12-month predictive slope, HAC *t* **−2.77** — has the
**wrong sign** (hot demand → *weaker* SPY, the late-cycle-overheating story) and is an
overlapping-window artefact. The **utilities sector (XLU)** is where a whiff of signal lives:
the 1-month hot-minus-cold spread is **+1.19%/mo** (Welch *t* **+2.13**, placebo *p* **0.032**),
just clearing the bar. But it **fails robustness**: drop calendar-2020 and the spread collapses
to +0.54%/mo (Welch *t* **+0.96**, placebo *p* **0.34**) — the entire edge is the COVID demand
crash coinciding with the market crash, one macro event, not a repeatable pulse. And the HAC
predictive slope for XLU is sub-2 (*t* **+1.54**). So `WEAK` on signal (one borderline,
COVID-dependent reading on utilities; literature-plausible but not robust; SPY flat). And the
tradable overlay **loses to buy-and-hold** on both tapes → `MIRAGE`.

## Data stamp

- **Electricity demand**: U.S. total net generation, all sectors, monthly TWh, 2007-01 → 2026-06,
  234 months, fingerprint `625867832b4d` (EIA `ELEC.GEN.ALL-US-99.M`, settled print)
- **Equity**: SPY + XLU daily total-return adjusted close, month-end aligned, fingerprint `dc0f80c14e47`
- **Aligned panel** (demand + SPY + XLU, monthly): 234 months, fingerprint `e7d78ee9af2d`

## The predictive regression — forward return on lagged demand growth (HAC *t*)

| Tape | Horizon | Slope | HAC slope-*t* | R² | n |
|---|---|---|---|---|---|
| SPY | 1m | −0.018 | −0.23 | 0.00 | 219 |
| SPY | 6m | −0.498 | −1.43 | 0.03 | 214 |
| SPY | 12m | −1.250 | **−2.77** | 0.11 | 208 |
| XLU | 1m | +0.101 | +1.54 | 0.01 | 219 |
| XLU | 6m | +0.244 | +1.42 | 0.01 | 214 |
| XLU | 12m | +0.156 | +0.51 | 0.00 | 208 |

The only slope past |*t*| = 2 is SPY at 12 months — and it is **negative** (the *opposite* of
the believers' sign: fast power-demand growth precedes *weaker* broad-market returns, the
classic late-cycle overheating read), on the most overlap-inflated horizon. XLU leans the
believers' way but never clears the HAC bar.

## The conditional split — hot- vs cold-demand-growth months (Welch two-sample *t*)

| Tape | Horizon | Hot mean | Cold mean | Spread | Welch *t* | Placebo *p* |
|---|---|---|---|---|---|---|
| SPY | 1m | +1.0% | +0.8% | +0.21% | +0.37 | 0.71 |
| XLU | 1m | +1.4% | +0.3% | **+1.19%** | **+2.13** | **0.032** |
| XLU | 3m | +3.5% | +1.2% | +2.30% | +2.72† | 0.009† |
| XLU | 6m | +7.3% | +2.2% | +5.11% | +4.49† | 0.0003† |

† The 3/6/12-month rows use **overlapping** forward windows, so their Welch *t* and placebo *p*
are badly over-stated (autocorrelated observations) — they are shown for shape, not inference.
The clean, non-overlapping test is **XLU at 1 month**: Welch *t* **+2.13**, placebo *p* **0.032**.

## Robustness — the XLU 1-month edge is a COVID artefact

| XLU, 1m | Spread | Welch *t* | Placebo *p* |
|---|---|---|---|
| Full sample (2007-2026) | **+1.19%** | **+2.13** | **0.032** |
| Ex-2020 (drop COVID year) | +0.54% | **+0.96** | 0.34 |
| HAC predictive slope (full) | — | +1.54 | — |
| HAC predictive slope (ex-2020) | — | +0.88 | — |

Remove a single year — 2020, when power demand cratered as the economy locked down and the
market cratered with it — and the only real-looking reading evaporates. A signal that depends
on one macro event is not a repeatable pulse.

## Tradability — the overlay loses to buy-and-hold

| Overlay: hold when demand hot, else cash | SPY | XLU |
|---|---|---|
| Buy-and-hold ann. return | **+10.8%** | **+8.5%** |
| Overlay gross ann. return | +6.3% | +3.6% |
| Overlay net ann. return (5 bps/switch, 56 switches) | +6.1% | +3.5% |
| Buy-and-hold Sharpe | 0.74 | 0.63 |
| Overlay Sharpe (net) | 0.64 | 0.38 |

On both tapes, acting on the demand-growth signal **destroys** return — you sit in cash through
too much of the rally. Even where the conditional spread looked real (XLU), it does not survive
being turned into a position. `MIRAGE`.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `edge` | Mean HAC slope-*t* (25 seeds) | |
|---|---|---|
| +0.00 (null) | **+0.37** | flat — no false signal |
| +0.01 | +1.87 | edge emerging |
| +0.02 | +3.36 | clears the bar |
| +0.03 | +4.81 | strong |
| +0.04 | +6.21 | unmistakable |

At the null the mean slope-*t* is ≈ 0; planting a genuine demand→returns link drives it
monotonically past +2. The detector works — so the real-tape result (SPY flat, XLU
COVID-dependent) is the tape talking, not a broken engine. (Control only; never cited for the
real-tape stamp.)

## Why the pulse doesn't certify here

1. **Electricity demand is coincident, not leading — and heavily seasonal.** The informative
   part (business-cycle growth) is a small residual once you strip the summer/winter seasonal;
   what remains moves *with* GDP, not ahead of it, and the market has usually already priced it.
2. **The one real-looking reading is a single event.** XLU's 1-month edge is the COVID-2020
   demand-and-market crash; ex-2020 it is *t* +0.96. One macro shock is not a strategy.
3. **Settled print, not real-time vintage.** EIA revises net generation ~1-2% for a few months;
   the 2-month lag protects against look-ahead but a strictly point-in-time backtest would see
   noisier early prints, biasing *against* any edge — so this is an optimistic reading already.

## The honest takeaway

The electricity the economy burns is a genuine hard-data pulse — but on the tape it is a
**coincident, seasonal, mostly-priced** one. The broad market shows nothing (its only
|*t*| ≥ 2 is the *wrong* sign at 12 months); utilities carry a single borderline reading
(*t* +2.13) that is entirely a COVID artefact and dies ex-2020; and turning any of it into a
position loses to buy-and-hold. `WEAK` × `MIRAGE`. The synthetic control confirms the engine
would bank a real lead — so this is the grid talking, not the code.
