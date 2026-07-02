# Results — Study 580 (Gold-Lease-Rate): does the bullion borrow cost lead gold?

*Generated from [`gold_lease_rate/`](../gold_lease_rate/). **This is a synthetic-only study**: the
LBMA discontinued the daily GOFO benchmark on **2015-01-30** and no free, continuous, long-history
gold-lease-rate tape is reachable on a no-key retail stack, so there is **no real tape** and the
headline runs on the deterministic offline generator (seed 580, 300 monthly periods, 2000-01 →
2024-12). The **null world** (`lead_beta = 0`, fingerprint `756d06b1b283`) is the honest headline —
the folklore's lead-lag is *withheld*; the **positive-control world** (`lead_beta = 0.020`,
fingerprint `e3ed82fc53fe`) proves the engine catches a planted lead-lag. As-of **2026-06-30**.*

## The verdict, earned — Signal `WEAK` · Tradability `MIRAGE`

Bullion-market folklore holds that the **gold lease rate** (the cost to borrow physical metal,
classically GOFO-implied as `LIBOR − GOFO`) *leads* the gold price — a borrow-cost spike
foreshadowing a rally. We build a predictive regression of forward gold returns on the lagged
standardised lease rate, a long-flat trading rule with costs, a label-shuffle placebo, a lag sweep,
and a seed-robust synthetic positive control.

There is **no real tape**, so the signal axis is **capped at WEAK** — a synthetic-only study can
never earn `REAL` (that needs a robust *t* ≥ 2 on a real tape). On the honest **null** world the
lead-lag is flat: forward-return slope on the lagged lease-*z* is **+0.0002** per sigma (*t*
**+0.08**), correlation **+0.005**, R² **0.000**, placebo *p* **0.936** — exactly the nothing you
should see when the effect is absent. The trading rule earns **+4.7%/yr net** versus **+9.2%/yr**
buy-&-hold — i.e. it *underperforms* passive gold, because sitting in cash ~56% of the time just
throws away drift. `MIRAGE` on tradability: no free tape, no lease-rate product, and even in the
generator the "signal" is a construct we planted or withheld by hand.

The **positive control** confirms the engine is faithful: plant the folklore (`lead_beta = 0.020`)
and the slope-*t* jumps to **+7.73** (placebo *p* **0.0005**, R² **0.167**), and across **25 seeds**
the mean *t* rises monotonically from **−0.43** at the null to **+11.3** at `lead_beta = 0.030`. So
*if* a real lease-rate lead-lag of this size existed, this machinery would catch it — but the study
cannot make that claim without the discontinued data.

## Data stamp

- **Synthetic panel (null)**: 300 monthly periods, `lead_beta = 0`, seed 580, fingerprint `756d06b1b283`
- **Synthetic panel (positive control)**: same, `lead_beta = 0.020`, fingerprint `e3ed82fc53fe`
- **Real tape**: *none* — GOFO discontinued 2015-01-30; no free continuous series (the cap on the SIGNAL axis)

## The predictive regression — null world (headline)

| | Null (`lead_beta = 0`) | Positive control (`lead_beta = 0.020`) |
|---|---|---|
| Slope *b* (fwd gold ret per 1σ of lagged lease-*z*) | **+0.0002** | **+0.0202** |
| Slope *t* | **+0.08** | **+7.73** |
| corr(lagged lease-*z*, fwd gold ret) | **+0.005** | **+0.409** |
| R² | **0.000** | **0.167** |
| Placebo *p* (label-shuffle, 2000 perms) | **0.936** | **0.0005** |
| n | 299 | 299 |

Folklore says *b* > 0 at *t* ≥ 2. The null world delivers ~0 (as it should); the planted world
clears the bar with room to spare. There is no real tape to adjudicate which the world actually is.

## The trading rule — long gold when lease-*z* > 0, else cash (null world)

| | Gross | Net (5 bps/switch × NAV) | Buy-&-hold gold |
|---|---|---|---|
| Annualised return | **+4.8%** | **+4.7%** | **+9.2%** |
| Net Sharpe | | **0.44** | |
| Switches | 34 | | |
| Fraction of months long | 44% | | |

On the null world the rule *loses to passive gold* — it is in cash 56% of the time and forgoes
drift for a signal that carries no information. (Costs are almost a footnote: 34 switches over
25 years at 5 bps each barely dents the return.) On the positive-control world the same rule earns
**+14.5% net** at Sharpe **1.20** — the machinery converts a real lead-lag into a tradable edge, but
only when one is actually present.

## Robustness — the lag sweep

| lag (months) | Null slope-*t* | Positive-control slope-*t* |
|---|---|---|
| 0 | −0.15 | +6.03 |
| 1 | **+0.08** | **+7.73** |
| 2 | −0.54 | +5.60 |
| 3 | −0.28 | +4.66 |
| 6 | −0.85 | +1.70 |
| 12 | −0.44 | −0.25 |

The null world scatters around zero at every lag (no lucky lag manufactures significance); the
positive-control world peaks at the *true* planted lag (1) and decays as the lag moves away — the
signature of a genuine, correctly-specified lead-lag.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `lead_beta` | Mean slope-*t* (25 seeds) | |
|---|---|---|
| 0.000 (null) | **−0.43** | flat — no false signal |
| 0.005 | +1.52 | emerging |
| 0.010 | +3.47 | clears the bar |
| 0.020 | +7.37 | strong |
| 0.030 | +11.27 | very strong |

At the null the mean *t* is ≈ 0; a genuine lead-lag drives it monotonically positive and past +2.
The detector works — so the study's inability to certify the folklore is a **data** limitation
(no GOFO tape), not a broken engine. (Control only; never cited for a real-tape stamp — there is
no real tape.)

## Why this can only be `WEAK`

1. **No real data.** The daily GOFO benchmark — the only clean, continuous public source of an
   implied lease rate — was discontinued by the LBMA on 2015-01-30. What remains is either
   short, paywalled, or reconstructed with judgement calls. A `REAL` stamp requires a robust
   *t* ≥ 2 on a **real** tape; a synthetic-only study cannot supply one.
2. **The folklore is untested here, not confirmed.** The null world shows the *absence* of the
   effect; the positive control shows the engine would *catch* it. Neither is evidence the effect
   exists on the actual market — only that the question is well-posed and the machinery honest.
3. **Even granting the signal, tradability is a mirage.** There is no lease-rate ETF or clean
   product to trade, the implied rate is quoted in arrears with wide bid/ask, and the operational
   long-flat rule underperformed passive gold on the null world.

## The honest takeaway

The gold-lease-rate lead-lag is a clean, seductive commodity-microstructure story — and this study
builds the exact machinery to test it: a lagged predictive regression, a costed long-flat rule, a
label-shuffle placebo, a lag sweep, and a seed-robust control that recovers a planted effect and
stays flat at the null. But the one thing it cannot supply is the **data**: GOFO is gone, and no
free continuous lease-rate tape exists. So the desk's verdict is `WEAK × MIRAGE` — the honest cap
for a well-built engine with nothing real to point it at.
