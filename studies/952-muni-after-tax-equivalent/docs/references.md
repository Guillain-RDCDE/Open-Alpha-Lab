# References & literature map — Study 952 (After-Tax Equivalent)

## The claim under test

- **The tax-equivalent-yield rule of thumb.** Every retail muni brochure prints the same
  identity: a tax-exempt yield *y* is worth `y / (1 − tau)` to an investor in marginal
  bracket *tau*, so above some bracket munis "obviously" beat taxable bonds. The rule is
  arithmetically unimpeachable and empirically incomplete: it compares *yields*, not
  *realised after-tax total returns*, and it silently assumes the two bonds are otherwise
  identical in duration, credit and liquidity. This study replaces the brochure identity
  with the measured thing — reconstructed monthly income legs, the price leg kept separate,
  and a **break-even effective marginal rate** solved from the tape.
- **The steelman.** The muni market is dominated by taxable US retail investors, so in an
  efficient market the tax break should be *priced in*: muni yields should sit below taxable
  yields by roughly the marginal bracket of the price-setting investor, leaving no free
  after-tax lunch. The empirical question is whether the priced-in rate is above or below
  the bracket a given investor actually faces. A break-even that lands inside the top two
  brackets is exactly what an efficient market predicts.

## Why the break-even bracket is the right statistic

- **Miller (1977), *Debt and Taxes*, Journal of Finance.** The classic argument that the
  muni-vs-taxable yield ratio reveals the marginal tax rate of the *marginal* investor, not
  the top statutory rate. The break-even bracket this study solves for is that implied rate,
  measured on realised total returns rather than quoted yields.
- **Trzcinka (1982), *The Pricing of Tax-Exempt Bonds and the Miller Hypothesis*, Journal of
  Finance**, and **Poterba (1989), *Tax Reform and the Market for Tax-Exempt Debt*, Regional
  Science and Urban Economics** — the empirical literature on the implied marginal rate in
  muni pricing, and how it moves with tax reform. Our estimate against IG corporates lands in
  the neighbourhood these papers keep landing in — but note *which* estimate: the **income-leg
  (tax-equivalent-yield)** break-even is 27–29% with a ±3 pp interval, while the
  **total-return** break-even's point estimate of 29–35% carries a 95% interval of roughly
  [−11%, +82%] and is not identified by this tape. Agreeing with the literature is not the
  same as measuring the thing, and this desk does not let the former stand in for the latter.
- **Green (1993), *A Simple Model of the Taxable and Tax-Exempt Yield Curves*, Review of
  Financial Studies** — why the muni curve is not a flat `(1 − tau)` scaling of the Treasury
  curve, which is why a duration-matched comparator (VCIT) and a duration-mismatched one
  (LQD, AGG) give materially different break-evens.
- **Ang, Bhansali & Xing (2010), *Taxes on Tax-Exempt Bonds*, Journal of Finance** — muni
  returns carry tax effects beyond the coupon exemption (the de-minimis rule, the market
  discount rule, capital-gains treatment). Our default "price leg untaxed" is an explicit
  simplification of exactly this, and the capital-gains sweep is its sensitivity check.

## Why the comparator choice decides the answer

- **Longstaff (2011), *Municipal Debt and Marginal Tax Rates: Is There a Tax Premium in
  Asset Prices?*, Journal of Finance** — muni yields relative to *risk-matched* taxable
  benchmarks imply implausibly high marginal rates, which is another way of saying the
  comparator does most of the work. Our tape shows the same thing, and shows *why*: the
  MUB-vs-AGG total-return break-even is **−6.9%** while MUB-vs-VCIT is **+35.0%** — a 42-point
  swing driven by comparator choice alone. Their bootstrap intervals ([−52%, +37%] and
  [−11%, +82%]) overlap over most of their length, which is the blunt version of Longstaff's
  point: on realised *total* returns the implied marginal rate is barely a measurement. On the
  *income* legs it is (MUB/VCIT 26.7% [23.4, 29.7]; MUB/AGG +7.6% [2.7, 12.0] — note the sign
  reversal against AGG once the noisy price legs are dropped).
- **Chalmers (1998), *Default Risk Cannot Explain the Muni Puzzle*, Review of Financial
  Studies**, and **Schwert (2017), *Municipal Bond Liquidity and Default Risk*, Journal of
  Finance** — the muni-Treasury spread is mostly liquidity and segmentation, not credit. A
  muni-vs-Aggregate comparison therefore mixes a liquidity premium into what looks like a
  tax result, which is why this study reports the pre-tax difference alongside every
  break-even and flags the pairings where munis already win at a 0% bracket.
- **Asset location.** Dammon, Spatt & Zhang (2004), *Optimal Asset Location and Allocation
  with Taxable and Tax-Deferred Investing*, Journal of Finance — the framing this study
  adopts for tradability: the decision is *which account holds which bond*, a one-round-trip
  choice, not a long-short spread. Reichenstein (2001), *Asset Allocation and Asset Location
  Decisions Revisited*, Journal of Wealth Management, for the practitioner version.

## Related desk studies (dedup)

- **[Study 576 — Muni-Treasury-Ratio](../../576-muni-treasury-ratio/)**: the muni/Treasury
  yield **ratio as a market-timing signal** (does a cheap ratio predict muni outperformance?)
  — a *valuation-timing* question on a distribution-yield proxy, verdict Weak/Mirage.
  Study 952 asks the orthogonal *level* question: at what marginal rate does the muni's
  after-tax total return actually beat taxable credit? No timing signal is involved; the tax
  wrapper, not the ratio's z-score, is the whole mechanism.
- **[Study 887 — High-Yield Muni Premium](../../887-high-yield-muni-premium/)**: a
  *within-muni credit* question (does HYD out-earn MUB?), with tax-equivalent yield used as
  a single-bracket side-check at 40.8%. Study 952 turns that side-check into the main event
  and generalises it: **seven** muni-vs-taxable pairings, a **sweep of five brackets** plus a
  state-tax and in-state-exemption knob, and — the part 887 never computes — the **break-even
  bracket** at which each pairing crosses. HYD vs LQD appears here as one row of that sweep.
- **[Study 616 — Muni-CEF-Tax-Loss](../../616-muni-cef-tax-loss/)**: a *seasonal* tax-loss
  harvesting effect in closed-end muni funds — a calendar anomaly, not an after-tax level
  comparison.
- **[Study 115 — Credit-Spreads](../../115-credit-spreads/)**: HY credit spreads as an
  *equity*-timing signal. Shares the taxable-credit tape, asks nothing about tax.
- **[Study 613 — Currency-Hedged-Carry](../../613-currency-hedged-etf-carry/)** and
  **[Study 889 — Dollar-Hedge-Overlay](../../889-dollar-hedge-overlay/)**: the desk's other
  *mechanical identity* studies, where the payoff is arithmetic rather than anomaly. Study
  952 belongs to that family — and, like them, the honest finding is that the identity is
  real while the excess return over a properly matched comparator is not.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../after_tax/strategy.py), 6 monthly lags.
- **Circular block bootstrap.** Politis & Romano (1992, 1994) — 6-month blocks preserve the
  autocorrelation of bond-fund returns; [`strategy.block_bootstrap_mean_ci`](../after_tax/strategy.py).
- **Bootstrapping a ratio of means.** The break-even is `τ* = −mean(d₀) / mean(i_taxable)`.
  Ratio estimators are notoriously badly behaved when the denominator's numerator is
  imprecise (Fieller, 1954, *Some Problems in Interval Estimation*, JRSS-B), which is exactly
  our case: the numerator never clears |*t*| ≥ 2. `strategy.breakeven_ci` block-bootstraps
  the whole ratio rather than propagating a delta-method standard error, so the asymmetric,
  very wide interval shows up honestly instead of being symmetrised away.
- **Why a *t*-stat can be manufactured by a constant.** `d(τ) = d(0) + τ·i_taxable`, and the
  second term is a coupon stream — large in the mean, near-zero in variance. Adding it raises
  the *t*-statistic mechanically. `strategy.tax_constant_decomposition` reports the tax term's
  share of the mean and of the variance for exactly this reason; on this tape every pairing
  that crosses |*t*| = 2 does so on a term carrying under 2% of the variance, and the same
  lift reproduces on a synthetic twin null with nothing planted. This is the study's main
  methodological warning, and it generalises to any comparison where one leg receives a
  deterministic subsidy (tax exemption, fee rebate, securities-lending credit — cf.
  **[914 — Securities-Lending Offset](../../914-sec-lending-offset/)**).
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — as-of slice
  plus content fingerprint, so a rerun either matches byte for byte or flags drift loudly.

## Data sources

- **MUB, VTEB, SUB, HYD** (municipal), **AGG, LQD, VCIT** (taxable credit), **BIL** (cash) —
  daily closes via `yfinance`, pulled **twice**: `auto_adjust=True` for the **total-return**
  tape and `auto_adjust=False` for the **price-only** tape. The monthly income leg is
  `total − price`; it is the only leg the tax code touches, so it is measured rather than
  assumed. Fund expense ratios are inside the total-return tape and are not charged again.
- **As-of 2026-06-30.** The partial current month is dropped so the monthly sample never
  creeps. Per-leg windows are set by ETF inception (VCIT 2009-11 gates the duration-matched
  headline pairing; VTEB 2015-08 gates its cross-check).
- **Every tax rate is a PROXY, none is tape.** Federal brackets 0 / 24 / 32 / 37%, the 3.8%
  net-investment-income surtax, state rates 0 / 5 / 9.3 / 13.3%, the in-state-exempt share of
  a national muni fund, and the capital-gains rate on the price leg are all assumptions
  imposed on the measured price and income legs. Each is swept in
  [`docs/results.md`](results.md); none of them changes the verdict. Real taxpayers also face
  AMT, bracket phase-outs and the muni de-minimis rule, none of which is modelled.
