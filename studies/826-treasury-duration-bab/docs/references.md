# References & literature map — Study 826 (Treasury Duration BAB)

## The claim under test

- **The source paper.** Andrea **Frazzini & Lasse Heje Pedersen**, *"Betting Against Beta"*
  (Journal of Financial Economics, 2014). Leverage-constrained investors bid up high-beta
  assets, so beta is priced too flat: low-beta assets earn higher **risk-adjusted** returns.
  Their **BAB factor** goes long a portfolio of low-beta assets **levered to a beta of one** and
  short a portfolio of high-beta assets **de-levered to a beta of one**, making the book
  beta-neutral; it earns a large, robust positive alpha. Crucially they show the effect is *not*
  an equity curiosity — it appears **across asset classes, including US Treasury bonds sorted by
  maturity**, credit, and international equities.
- **The Treasury-curve version.** Sorting Treasuries by maturity is sorting by **duration**, and
  duration is (nearly) the bond's beta to a common rates/duration factor. Betting against beta
  inside the curve therefore means **levering up the short-duration (low-beta) end and shorting
  the long-duration (high-beta) end**, beta-neutral — the low-risk tilt applied to governments.
- **The specific test here.** We rebuild that Treasury-curve BAB from five liquid iShares ETFs
  that ladder the curve (SHY → IEI → IEF → TLH → TLT), estimate each ETF's trailing-252-day beta
  to an equal-weight duration factor, and form the classic Frazzini-Pedersen rank-weighted book,
  with a Newey-West *t*, a factor-regression alpha, a permutation placebo, a two-era robustness
  cut, a costed leveraged timer, and a seeded synthetic positive control.

## What we measure, and the honesty rails

- **Beta to a duration factor, no free model.** The factor is the equal-weight mean of the five
  daily returns; each ETF's beta is the trailing-window `Cov(r_i,f)/Var(f)`, computed vectorised.
- **Frazzini-Pedersen rank weights, levered to unit beta.** Long weights ∝ `(z̄−z_i)` on the
  below-median (low-beta) legs, short weights ∝ `(z_i−z̄)` on the above-median (high-beta) legs,
  each side normalised to sum to one; the long leg is levered `1/β_L`, the short `1/β_H`, so the
  net factor beta is ~zero. Any surviving return is the claimed low-risk alpha.
- **Point-in-time, one documented lag.** The ranking beta is **known at the close of `t−1`**
  (`.shift(1)`); the book is held on day `t`. Zero look-ahead.
- **The risk-free simplification, named openly.** BAB is defined on returns **in excess of the
  risk-free rate**; the `1/β` leverage otherwise amplifies any common drift asymmetrically. We
  run the real total-return ETFs with `rf ≈ 0` (a daily bill is ~0 and not in the five-ticker
  fetch) — a simplification that *flatters* the levered low-vol leg's carry, which is exactly the
  artefact the placebo and the costed timer are built to expose.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily BAB — an
  overlapping-formation beta signal is serially correlated, so a plain *t* overstates
  significance. A **1,000-permutation placebo** breaks the beta → return link (permuting which
  ETF feeds each leg of the same leverage cage) to test whether the *beta signal*, not the
  leverage mechanics, produces the spread.
- **The timer is graded separately.** One-way cost × turnover of the *levered* weights, plus
  borrow on the short leg — the honest test of whether the (already placebo-condemned) spread
  survives friction.

## Shared method citations

- **Frazzini, A. & Pedersen, L. H. (2014)** — betting against beta (the claim under test).
- **Black, F. (1972)** — the flat security market line / restricted-borrowing CAPM that
  underpins why low-beta assets should earn positive alpha.
- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the BAB series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.

## Data sources

- **yfinance daily total-return closes** (`auto_adjust=True`), five iShares Treasury ETFs
  (SHY, IEI, IEF, TLH, TLT), 2010-01-04 → 2026-06-30, cached under `_cache/`.
- All headline numbers are pinned in [`results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [796-corporate-bond-low-risk](../../796-corporate-bond-low-risk/) — the low-risk / BAB tilt in
  **corporate & credit** bonds (spread/credit risk). This study is **duration risk inside the
  government curve** — no credit component; the factor is the Treasury duration factor, the legs
  are maturity buckets of the *same* riskless issuer.
- [238-betting-against-beta](../../238-betting-against-beta/) — the original Frazzini-Pedersen
  BAB across the **equity** cross-section (market beta). This study runs the identical
  construction on **Treasury ETFs by maturity**, where "beta" is duration, not equity market
  beta.
- [581-term-premium](../../581-term-premium/) — an ACM-style **term-premium timing** signal on
  long duration (TLT): a time-series *when to own duration* call. This study is a
  **cross-sectional, beta-neutral** long-short across the five maturity buckets — no directional
  duration bet.

None of the siblings run a **beta-neutral betting-against-beta book across Treasury maturity
buckets** — the duration-BAB signal — which is this study's own axis.
