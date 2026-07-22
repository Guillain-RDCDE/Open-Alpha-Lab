# References & literature map — Study 795 (Corporate-Bond-Momentum)

## The claim under test

- **The source paper.** Jostova, Nikolova, Philipov & Stahel (2013), *"Momentum in
  Corporate Bond Returns"*, **Review of Financial Studies** 26(7). Using 1973-2011 US
  corporate-bond data, they document significant price momentum in corporate bonds:
  past-6-month winners out-perform past losers over the next 6 months. Crucially, the
  effect is **concentrated in *non-investment-grade* (high-yield) bonds** and is *weak to
  absent in investment-grade* — the credit-quality dimension is central to their finding.
- **The wider family.** Cross-sectional momentum — past winners keep winning relative to
  past losers — is one of finance's most robust anomalies in equities (Jegadeesh & Titman
  1993) and across asset classes (Asness, Moskowitz & Pedersen 2013, *"Value and Momentum
  Everywhere"*, JF). Whether it survives in the *bond* cross-section, and in which credit
  segment, is the open empirical question Jostova et al. answer for single names.
- **What we test, and its honest limit.** We take the claim to the **ETF panel**: rank an
  11-name credit + Treasury bond-ETF basket on trailing total return and trade
  winners-minus-losers monthly. This is the *tradable, retail-accessible* version of the
  claim — but it is a **coarser instrument** than the single-name cross-section Jostova
  studied: one ETF per credit sleeve cannot reproduce the within-high-yield dispersion the
  effect lives in. A null on this panel is therefore consistent with — not a refutation of —
  the paper's own emphasis that momentum is a *high-yield single-name* phenomenon.

## What we measure, and the honesty rails

- **The winners-minus-losers spread.** Equal-weight top-third-long / bottom-third-short,
  dollar-neutral, formed on the month-*t* close and earning month *t+1* (one documented
  execution `shift`, no same-bar fill). The inference-bar number is a **Newey-West (HAC)
  one-sample *t*** on the monthly WML mean; we also report the plain *t*, Sharpe, a Wilson
  interval on the hit rate, and the max drawdown.
- **The rank-shuffle placebo.** Keep each month's realised return cross-section exactly, but
  randomly permute which asset gets which momentum rank, 2,000 times — destroying the
  past→future link while preserving the return distribution and leg sizes. The one-sided
  share of placebos beating the real mean is the permutation *p*.
- **Robustness rails.** The formation window is swept across the **6-12 month** range the
  claim itself names (6m headline, 12m, 12-1, 3m); the sample is split into the Jostova-era
  (≤ 2013) and post-publication (2014-2026) slices; a cost sweep charges one-way turnover ×
  NAV with the short leg paying borrow, and reports the break-even cost.
- **The "is it just credit beta?" check.** We contrast the dollar-neutral WML spread against
  a naive long-HYG position — a real credit premium (HAC *t* ≈ 2.5) that the momentum tilt
  is designed to strip out (corr ≈ −0.10) and, once stripped, adds nothing to.
- **Survivorship** is named on the **Signal** axis: the basket is the current-membership ETF
  list projected backwards. Because a falling ETF is shorted rather than deleted, the bias is
  milder than a single-name sort, and a *null* result (what we find) is not manufactured by
  it.

## Data sources

- **Bond-ETF total-return prices** — yfinance (no key), auto-adjusted close (coupons folded
  in), cached under `_cache/bondmom_prices.parquet`, 2007-01-03 → 2026-06-30.
- **The synthetic positive control** — a deterministic, seeded total-return panel with a
  planted cross-sectional-momentum knob (null at 0); no network. It proves the WML engine
  recovers a real planted effect and scores the null at zero.
- All headline numbers are pinned in [`docs/results.md`](results.md) (fingerprint
  `1f2efa58efab`) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [518-time-series-momentum](../518-time-series-momentum/) — **time-series** (trend)
  momentum: each asset traded long/short on the sign of **its own** trailing 12-month
  return, then vol-scaled and diversified (Moskowitz-Ooi-Pedersen). This study is
  **cross-sectional**: assets are ranked **against each other** and the book is a
  winners-*minus*-losers spread, not an own-sign trend book.
- [247-bond-seasonality](../247-bond-seasonality/) — a **calendar** (month-of-year) claim on
  bonds. No ranking, no cross-section: it asks whether specific months are systematically
  strong/weak, an entirely different signal from a trailing-return rank sort.
- [611-mreit-carry](../611-mreit-carry/) and [612-em-debt-carry](../612-em-debt-carry/) —
  **carry** claims (a fat yield you collect by *holding* mortgage-REITs / EM sovereign
  debt), graded on whether the coupon survives its left tail. Carry is a *level* signal
  (own the high-yielder); momentum is a *change* signal (own the recent out-performer). Same
  fixed-income neighbourhood, orthogonal signal.

None of the siblings test a **cross-sectional trailing-return rank** on a bond universe —
that is this study's own axis.
