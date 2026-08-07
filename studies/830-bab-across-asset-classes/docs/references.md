# References & literature map — Study 830 (BAB Across Asset Classes)

## The claim under test

- **The source paper.** Andrea **Frazzini & Lasse Heje Pedersen**, *"Betting Against Beta"*
  (Journal of Financial Economics, 2014). Leverage-constrained investors overweight
  high-beta assets, flattening the security-market line: high-beta assets earn *too little*
  and low-beta assets *too much* per unit of risk. Their **BAB factor** — long low-beta
  (levered to unit beta), short high-beta (de-levered to unit beta) — earns a large,
  significant alpha. Crucially, Section 4 of the paper argues the effect is **not** confined
  to US stocks: they document a flat SML and positive BAB returns **across 20 international
  equity markets, Treasury bonds, credit indices, and commodities** — a *betting-against-beta
  everywhere* claim. This study builds the self-contained **cross-asset-class** version.
- **The frictional mechanism.** Because many investors cannot (or will not) use leverage, they
  buy high-beta securities to reach their target risk, bidding those prices up and their future
  returns down; the arbitrageur who *can* lever should prefer low-beta assets and lever them.
  Whether that mechanism operates *across whole asset classes* — where beta is measured to a
  blended multi-asset market and the low-beta bucket is dominated by bonds and gold — is the
  open question this study grades.
- **The specific test here.** Nine liquid asset-class ETFs; each asset's Frazzini-Pedersen
  rolling beta to their equal-weight portfolio; a beta-neutral long-low / short-high BAB factor;
  a Newey-West *t* on the daily factor return; a HAC CAPM alpha; a permutation placebo; a two-era
  robustness cut; a costed levered timer; and a seeded synthetic positive control.

## What we measure, and the honesty rails

- **Frazzini-Pedersen betas, no free model.** `beta_i = 0.6·(ρ_i·σ_i/σ_m) + 0.4`, with the
  correlation ρ over a long (252-day) window and the volatilities over a short (63-day) window,
  shrunk toward the cross-sectional prior of 1 — the asymmetric estimator from the paper's
  Appendix. Fully causal.
- **Ex-ante beta neutrality, one documented lag.** Each leg is scaled to unit beta so the netted
  book is *ex-ante* market-neutral; the ranking beta is **known at the close of `t−1`**
  (`.shift(1)`) and the book is held on day `t`. Zero look-ahead. We also report the book's
  *realized* market beta — which turns out far from zero, the study's central finding.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily factor return; a HAC
  CAPM regression separates alpha from any residual market exposure; a **1,000-permutation
  placebo** shuffles which asset each leg holds to confirm the beta sort — not the mechanical
  leverage tilt — is what (would) drive the return.
- **Survivorship is named on the Signal axis.** The nine ETFs are current-membership, still-listed
  proxies; magnitudes are an upper bound.
- **The timer is graded separately.** Realized turnover × one-way cost × NAV on the levered book,
  plus borrow on the short leg — the honest test of whether a small cross-asset alpha survives the
  friction of running 2.6× gross.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the factor series and the CAPM alpha).
- **Black, F., Jensen, M. & Scholes, M. (1972)** — the original "flat security-market line"
  evidence that BAB formalises and monetises.
- **Wilson, E. B. (1927)** — score interval for a binomial share.

## Data sources

- **yfinance daily total-return closes** (`auto_adjust=True`), nine asset-class ETFs
  (SPY, EFA, EEM, TLT, LQD, HYG, GLD, DBC, VNQ), 2007-04-11 → 2026-06-30, cached under `_cache/`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [238-betting-against-beta](../../238-betting-against-beta/) — BAB in the **US single-stock
  cross-section** (the original Frazzini-Pedersen test on thousands of names). This study lifts
  the same construction to **whole asset classes** (nine ETFs), where the low-beta bucket is
  bonds and gold, not defensive stocks — a different universe and a different economic bet.
- [660-carry-everywhere](../../660-carry-everywhere/) — carry (yield / roll-down) harvested
  **across asset classes**. Same "everywhere" spirit, but the sort variable is **carry**, not
  **beta to a common market** — an orthogonal signal.
- [68-all-weather-risk-parity](../../68-all-weather-risk-parity/) — inverse-volatility /
  risk-parity **allocation** across asset classes (long-only, a portfolio-construction rule).
  BAB is a **long-short, beta-neutral factor**, not an allocation, and it *sorts* on beta rather
  than *weighting* by inverse vol.

None of the siblings build a **beta-neutral long-low-beta / short-high-beta factor across asset
classes** — the Frazzini-Pedersen "BAB everywhere" claim — which is this study's own axis.
