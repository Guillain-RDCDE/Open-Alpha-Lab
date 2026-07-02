# Results — Study 573 (Customer-Concentration): supply-chain fragility as a risk / return factor

*Generated from [`customer_concentration/`](../customer_concentration/) on the **deterministic
synthetic panel** (seed 573; 400 firms; concentration ∈ [0.004, 0.848], mean 0.313; panel
fingerprint `bc1d5db4bfa5`). **This study is synthetic-only**: there is no free, point-in-time
customer-concentration tape (the measure lives in paywalled 10-K "major customer" / Compustat
segment disclosures), so it can never earn a `REAL` signal — that needs a robust t ≥ 2 on a real
tape — and is capped at `WEAK`. The headline panel plants both effects at literature-plausible
strengths (`vol_beta = 0.35`, `ret_alpha = 0.04`); a matching null panel (both knobs 0) is the
honesty check. As-of **2026-06-30**.*

## The verdict, earned — Signal `WEAK` · Tradability `MIRAGE`

The customer-concentration claim (Patatoukas 2012; Dhaliwal, Judd, Serfling & Shaikh 2016; Hertzel,
Li, Officer & Rodgers 2008) has two legs: concentrated firms should have **higher forward risk**
(the robust part) and, if that risk is priced, a **return premium** (or, in the behavioural telling,
a discount). We build a Herfindahl-style concentration score, sort a 400-firm synthetic
cross-section into terciles, and test both legs.

On the headline (effect-planted) panel the **risk leg is unambiguous**: the concentrated tercile
has forward vol **24.5%** vs the diversified tercile's **21.2%**, a spread of **+3.3pp**
(two-sample *t* **+29.4**), and the firm-level slope of forward vol on concentration is **+0.076**
per unit (*t* **+53.5**, corr **+0.94**). But the **return leg is weak even where the premium is
planted**: the concentrated tercile earned **+12.1%** vs diversified **+7.2%**, a long-short spread
of **+4.9%** — the right sign, but two-sample *t* only **+1.61**, placebo *p* **0.102**, and the
firm-level return slope-*t* is just **+1.38** (corr **+0.07**). So `WEAK` on signal (the *risk* half
is real-in-synthetic and robust, but no real tape exists and the *return* half fails to clear t = 2
even in the planted world) and `MIRAGE` on tradability (no investable concentration index, and the
return spread you would trade does not survive its own noise).

## Data stamp

- **Tape**: synthetic only — no real customer-concentration data is free/point-in-time
- **Headline panel** (effect planted, `vol_beta = 0.35`, `ret_alpha = 0.04`, seed 573): 400 firms,
  fingerprint `bc1d5db4bfa5`; truth fingerprint `0f4a05e82a22`
- **Null panel** (both knobs 0, seed 573): the honesty check

## The concentration sort — risk leg strong, return leg weak

| Tercile (120 names) | Forward VOL | Forward RETURN |
|---|---|---|
| **Diversified** (lowest concentration) | **21.2%** | **+7.2%** |
| **Concentrated** (highest concentration) | **24.5%** | **+12.1%** |
| **Spread (concentrated − diversified)** | **+3.3pp** (two-sample *t* **+29.4**) | **+4.9%** (two-sample *t* **+1.61**) |

The risk story is emphatic; the return story is directionally there (a premium) but statistically
soft — the label-shuffle placebo *p* on the return spread is **0.102**, i.e. not distinguishable
from noise at conventional thresholds.

## The firm-level relations — the sign IS the claim

| Regression | Slope | Slope *t* | corr | Reads as |
|---|---|---|---|---|
| forward VOL on concentration | **+0.076** | **+53.5** | **+0.94** | risk story confirmed |
| forward RETURN on concentration | **+0.087** | **+1.38** | **+0.07** | premium sign, but not significant |

The return correlation is **+0.07** — the premium is *swamped by the very fragility volatility it
plants*: a concentrated firm is genuinely noisier, so the extra return sits inside a much wider
dispersion and the cross-sectional slope barely registers. That is the study's core lesson: even
if concentration IS a priced risk, its return premium is nearly undetectable in a single
cross-section.

## Robustness — sweep the tail fraction

| Tail fraction | VOL spread | VOL *t* | RETURN spread | RETURN *t* |
|---|---|---|---|---|
| 0.10 (decile) | **+4.7pp** | **+30.1** | **−1.3%** | −0.23 |
| 0.20 (quintile) | **+3.9pp** | **+30.0** | **+3.1%** | +0.85 |
| 0.30 (tercile) | **+3.3pp** | **+29.4** | **+4.9%** | +1.61 |
| 0.40 (broad) | **+2.8pp** | **+27.6** | **+5.3%** | +2.05 |

The **risk** *t* is enormous and stable at every cut. The **return** *t* wanders from **−0.23** at
the deciles to **+2.05** at the broad cut — sign-unstable and only marginally significant at the
widest bucket, which is exactly the fragile-signal fingerprint.

## Costs

| | value |
|---|---|
| Gross return spread (concentrated − diversified, terciles) | **+4.9%** |
| Net (5 bps/leg round-trip + 100 bps/yr borrow, 1y hold) | **+3.7%** |

Costs shave the already-soft return spread, but the binding problem is upstream: the gross spread
is not statistically distinguishable from zero (placebo *p* 0.102).

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

**Risk leg** (mean forward-vol-on-concentration slope-*t*):

| Planted `vol_beta` | Mean vol slope-*t* (25 seeds) | |
|---|---|---|
| 0.00 (null) | **+0.21** | flat — no false risk signal |
| 0.15 | +22.3 | risk visible |
| 0.35 (headline) | +51.8 | emphatic |
| 0.60 | +88.7 | scales cleanly |

**Return leg** (mean forward-return-on-concentration slope-*t*):

| Planted `ret_alpha` | Mean ret slope-*t* (25 seeds) | |
|---|---|---|
| 0.00 (null) | **+0.15** | flat — no false premium |
| 0.02 | +0.46 | barely moves |
| 0.04 (headline) | +0.78 | weak |
| 0.08 | +1.41 | still below the bar |
| 0.12 | **+2.05** | clears t = 2 only at a large planted premium |

At the null both legs sit at *t* ≈ 0 (no false signal). The risk detector lights up immediately;
the return detector needs a **large** planted premium (`ret_alpha ≈ 0.12`) to clear *t* = 2 under
realistic dispersion — and if we shrink the noise (`base_vol = 0.08`) a modest `ret_alpha = 0.08`
returns *t* ≈ 3.4, proving the detector is faithful, not broken. The return premium is hard to see
precisely *because* concentration also inflates the return's own variance.

## Why this is synthetic-only (the data-availability limit, named on SIGNAL)

Customer concentration is disclosed in 10-K "major customer" footnotes under SFAS 131 (any customer
> 10% of revenue) and aggregated in Compustat's segment files — paywalled and/or hand-collected. A
no-key retail stack cannot assemble a point-in-time concentration panel, so there is **no real tape
to certify against**. Every number above is from the synthetic world; a synthetic-only study is
capped at `WEAK` by house rule (a `REAL` stamp requires a robust t ≥ 2 on a real tape).

## The honest takeaway

The literature's *risk* leg — concentrated firms are more volatile — is strong and robust in the
synthetic world (*t* +29 at the sort, +53 firm-level, stable across cuts). The *return* leg is the
fragile one: even with a premium deliberately planted, the cross-sectional return *t* is only ~1.6
and the placebo *p* is 0.10, because the same fragility that raises risk also buries the premium in
noise. Combined with the absence of any free real tape, that earns `WEAK` × `MIRAGE`: depending on
a few big customers plausibly *does* make a stock riskier, but the market does not visibly, tradably
pay you a premium for bearing that risk.
