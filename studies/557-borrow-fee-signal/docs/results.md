# Results — Study 557 (Borrow-Fee-Signal): does the securities-lending fee predict returns?

*Generated from [`borrow_fee_signal/`](../borrow_fee_signal/) on the **deterministic synthetic
cross-section** (seed **557**, 120 names, planted `fee_alpha = -0.035`), fingerprint
`f48c7a1477b7`. There is **no free historical stock-borrow-fee tape** — the data is private
(Markit / IHS DataExplorers, S3 Partners), so this study is **synthetic-only** and its verdict is
capped at WEAK/NONE by construction (REAL needs a robust t ≥ 2 on a **real** tape). As-of
**2026-06-30**.*

## The verdict, earned — Signal `WEAK` · Tradability `FRAGILE`

Cohen, Diether & Malloy (2007) and D'Avolio (2002) document that the **cost to borrow** a stock —
the securities-lending fee — is the market-clearing price of shorting demand against lendable
supply, and that *special* / hard-to-borrow names (high fee) go on to earn **negative** forward
returns, with information *beyond* raw short interest. We build a synthetic borrow-fee
cross-section that plants exactly that effect, sort into quintiles by fee, and test the
long-cheap / short-expensive spread.

On the synthetic tape the effect is **real but modest, and the engine catches it**: the cheap
quintile earned **+13.2%** vs the expensive quintile's **+0.5%**, a long-cheap/short-expensive
spread of **+12.8%** (two-sample *t* **+2.53**, placebo *p* **0.009**). The firm-level slope of
forward return on the standardized fee is **−4.2%/σ** (*t* **−2.58**). Crucially the fee's signal
**survives controlling for short interest**: in a joint regression the fee slope-*t* is **−2.15**
while the short-interest slope-*t* is **+0.15** — the fee carries the information, not raw SI. So
`WEAK` on the signal axis (the effect is present and literature-backed, but **synthetic-only** —
no real tape can certify REAL, and even here the seed-robust quintile *t* averages only ~1.5),
and `FRAGILE` on tradability (the alpha lives on the *expensive-to-borrow short leg* — the fee is
both the signal **and** the bill; it survives at modest quintile fees but the real hard-to-borrow
tail can swamp it).

## Data stamp

- **Panel**: 120-name synthetic cross-section, seed 557, planted `fee_alpha = -0.035`
  (`idio_vol = 0.18`), fingerprint `f48c7a1477b7`
- **Fee distribution**: annualised borrow fee, median **0.61%**, 90th pct **2.6%**, max **8.5%**
  (a low general-collateral floor with a right tail of *special* / hard-to-borrow names)
- **corr(borrow_fee, short_interest)** = **0.60** — correlated but distinct (the fee prices the
  *supply* side that short-% of float cannot see)

## The fee sort — cheap beats expensive (the claim's sign)

| Quintile (24 names) | Forward return |
|---|---|
| **Cheap-to-borrow** (lowest fee) | **+13.2%** |
| **Expensive / hard-to-borrow** (highest fee) | **+0.5%** |
| **Spread (cheap − expensive)** | **+12.8%** (two-sample *t* **+2.53**) |

The claim predicts cheap > expensive (a *positive* spread). The synthetic tape delivers it: the
expensive-to-borrow tail underperforms. The label-shuffle placebo *p* = **0.009** says this is not
noise on this panel. Mean borrow fee of the short (expensive) leg: **3.1%** annualised.

## The firm-level relation

| | value |
|---|---|
| Slope (forward_ret on z(fee)) | **−4.2%** per fee-σ |
| Slope *t* | **−2.58** (a *negative* slope is the claim) |
| corr(fee, forward return) | **−0.23** |

## Incrementality — does the fee add signal *beyond* short interest? (dedup from Study 262)

| Regressor (joint OLS) | slope | *t* |
|---|---|---|
| z(borrow fee) | **−4.3%** | **−2.15** |
| z(short interest) | +0.3% | **+0.15** |

The fee's slope-*t* stays significant while short interest's collapses to noise: on this panel the
**fee carries the information, short interest does not**. This is the study's distinction from
[Study 262 — Short-Interest](../../262-short-interest/), which sorts on short-% of float (the
*quantity* demanded); here we sort on the *price* of that demand.

## Robustness — the sign holds across bucket widths

| Tail fraction | Spread (cheap − expensive) | Welch *t* |
|---|---|---|
| 0.10 (deciles) | **+11.3%** | +1.32 |
| 0.20 (quintiles, headline) | **+12.8%** | +2.53 |
| 0.30 | **+9.2%** | +2.11 |
| 0.40 | **+9.1%** | +2.51 |

The spread is **positive and same-signed at every cut** — a real, if modest, cross-sectional
signal, strongest at the quintile/decile tails as the literature predicts.

## Costs — the borrow fee is BOTH the alpha AND the bill

| | value |
|---|---|
| Gross spread (quintile) | **+12.8%** |
| Net (5 bps/leg round-trip + real 3.1% short-leg borrow, 3-month hold) | **+11.8%** |
| Net (same, 1-year hold) | **+9.4%** |
| Top-decile short-leg fee | **4.1%** → net (1y) **+7.0%** |

The honest twist: the short leg you want to hold is *exactly* the expensive-to-borrow tail, so the
borrow you pay is the observed fee, not a flat placeholder. At the modest synthetic fees it
survives; but in reality the hard-to-borrow tail routinely carries **20–100%+** annualised fees
(GameStop-2021 territory), which would **swamp** a 12% gross spread — the `FRAGILE` core.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `fee_alpha` | Mean firm slope-*t* (25 seeds) | Mean LS *t* |
|---|---|---|
| 0.00 (null) | **−0.00** | −0.08 |
| −0.01 | −0.61 | +0.39 |
| −0.02 | −1.22 | +0.86 |
| −0.035 (headline) | **−2.13** | +1.55 |
| −0.05 | **−3.05** | +2.21 |

At the null the slope-*t* is ≈ 0 (no false signal); planting a genuine fee effect drives the slope
negative and past −2 as it grows. At the headline `fee_alpha = −0.035` the seed-robust firm
slope-*t* is **−2.13** (clears the bar) but the quintile LS-*t* averages only **+1.55** — the
effect is genuinely *modest*, which is why detection is marginal and the signal is `WEAK`, not
strong. **Control only; never cited for a REAL stamp.**

## Short-interest-only null — the fee-sort does not fire on a pure-SI world

Plant the effect in **short interest instead** (`fee_alpha = 0`, `si_alpha = −0.03`) and re-run
the *joint* regression: the fee slope-*t* falls to **−0.42** (dead) while the short-interest
slope-*t* is **−1.34**. The partial regression correctly attributes the signal to SI — so the
fee-sort's headline is *not* a laundered short-interest effect. (The raw fee-sort slope-*t* does
leak to ≈ −1.2 on that world because fee and SI correlate 0.60, which is *why* the incremental
regression, not the raw sort, is the honest test.)

## Why this can only be WEAK

1. **No real tape.** A clean historical panel of daily borrow fees is private, paid data. yfinance
   exposes short *interest* but not the borrow *fee*. Without a real tape, a robust *t* ≥ 2 on real
   data is unreachable — so REAL is off the table (house rule). This limitation is named on the
   SIGNAL axis, exactly as for the desk's other synthetic-only studies (lego-returns, whisky-cask,
   sneaker-resale).
2. **The effect is modest.** Even on the synthetic world the seed-robust quintile *t* is ~1.5; the
   headline seed (557) is a touch luckier than the average. Real published fee premia are real but
   small and concentrated in the hardest-to-borrow micro-caps.
3. **The signal and the cost share a variable.** The alpha sits on the short leg you can least
   afford to hold — a structural tradability tax, not a footnote.

## The honest takeaway

The borrow-fee signal is real in the literature and reproducible on a planted synthetic tape: the
engine banks it, it survives controlling for short interest, and its sign is stable across bucket
widths. But it is `WEAK` because there is **no free real tape to certify it** and even the
synthetic effect is modest, and `FRAGILE` because the alpha lives on the expensive-to-borrow short
leg — the fee is both the signal and the bill. The synthetic control confirms the harness would
catch a real effect, so this is a data-availability verdict, not a broken engine.
