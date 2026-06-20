# References & literature map — Study 327 (Disposition-Effect)

## The claim under test

- **Grinblatt, M. & Han, B. (2005).** *Prospect Theory, Mental Accounting, and Momentum*
  (Journal of Financial Economics 78). The canonical cross-sectional statement of the
  disposition effect: disposition-prone investors create excess *selling* pressure in stocks
  with large **unrealised capital gains**, holding their prices below fundamentals; those
  stocks subsequently **out**perform, while deep-underwater names under-perform. The proxy is
  **capital-gains overhang** ``g = (P − R)/P`` with ``R`` a turnover-weighted reference price
  (the volume-weighted cost basis of current holders). This is the testable hypothesis we run:
  an overhang quintile sort earns a positive Q5 − Q1 hedge that survives a momentum control.

## The behavioural foundation — the disposition effect

- **Shefrin, H. & Statman, M. (1985).** *The Disposition to Sell Winners Too Early and Ride
  Losers Too Long* (Journal of Finance 40) — the original naming of the bias, grounded in
  Kahneman & Tversky's prospect theory.
- **Odean, T. (1998).** *Are Investors Reluctant to Realize Their Losses?* (Journal of Finance
  53) — the decisive empirical confirmation in a large discount-brokerage account dataset:
  investors realise gains at a markedly higher rate than losses.
- **Frazzini, A. (2006).** *The Disposition Effect and Underreaction to News* (Journal of
  Finance 61) — links the bias to post-earnings-announcement drift: the overhang predicts the
  sign and size of under-reaction, the mechanism this study's factor is meant to harvest.

## The central confound — overhang vs. momentum

- **Jegadeesh, N. & Titman, S. (1993).** *Returns to Buying Winners and Selling Losers*
  (Journal of Finance 48) — the 12-1 momentum premium. A deep-in-the-money stock is, almost by
  construction, a recent winner, so the overhang factor is **mechanically correlated with
  momentum**. Grinblatt & Han's contribution is precisely the claim that overhang *subsumes*
  momentum; the skeptical reading is that overhang is momentum wearing a behavioural costume.
  This study cross-sectionally **orthogonalises** overhang to 12-1 momentum to separate them.
- **Novy-Marx, R. (2012).** *Is Momentum Really Momentum?* (Journal of Financial Economics 103)
  — a cautionary note on how many "distinct" anomalies are repackaged momentum; directly
  relevant to whether the disposition premium is a free-standing effect.

## Replication, fragility & multiple testing

- **Hou, K., Xue, C. & Zhang, L. (2020).** *Replicating Anomalies* (Review of Financial Studies
  33) — a large fraction of published cross-sectional anomalies fail to replicate out of sample
  or shrink to insignificance with proper standard errors; the overhang/disposition factor is in
  the contested zone, especially among large, liquid names.
- **Harvey, C., Liu, Y. & Zhu, H. (2016).** *…and the Cross-Section of Expected Returns* (Review
  of Financial Studies 29) — the multiple-testing problem. A small, hand-picked large-cap
  cross-section (as used here for the offline-reproducible real tape) is exactly where a real
  broad-universe premium can vanish — which is what we observe.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey, W. & West, K. (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../disposition_effect/strategy.py), lag rule ``floor(4·(n/100)^(2/9))``.
- **Circular block bootstrap.** Politis, D. & Romano, J. (1992/1994) — block resampling to
  preserve autocorrelation in the monthly hedge series; [`strategy.block_bootstrap_ci`].
- **Information coefficient.** Spearman rank-correlation of signal vs forward return — the
  standard cross-sectional alpha diagnostic ([`strategy.information_coefficient`]).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), adjusted close + volume, for a small named
  large-cap cross-section (Dow-30-style), 2010–2026. The overhang reference price uses a
  turnover-weighted 1260-day (≈5-year) window. All headline numbers are pinned with an as-of
  date and content fingerprints (see [`docs/results.md`](results.md)). The offline reproducible
  core and the test-suite run on the deterministic
  [`data.synthetic_panel`](../disposition_effect/data.py) generator, never the network.

## Related desk studies

- **[Study 124 — Cash-Flow-Yield](../../124-cash-flow-yield/)**: the cross-sectional
  quintile-sort + survivorship-opt-in + momentum-control apparatus this study reuses.
- The desk's broader momentum and factor-zoo work (12-1 momentum, residual momentum, BAB, QMJ)
  supplies the comparison set for "is this just a repackaged known factor?".
