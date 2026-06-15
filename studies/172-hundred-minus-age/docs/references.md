# References & literature map — Study 172 (Hundred-Minus-Age)

## The claim under test

**The folk rule.** A near-universal piece of financial-planning advice: invest
*(100 − age)%* of your portfolio in stocks and the rest in bonds. A 30-year-old
holds 70/30; a 60-year-old holds 40/60. The implicit logic: stocks are riskier and
have higher expected return; as you age, your human capital (future earnings) shrinks
so your financial capital should become the "safe" leg; declining equity exposure
over the lifecycle therefore *matches* the natural hedge provided by labour income
and reduces catastrophic sequence-of-returns risk near retirement. The rule is
so embedded that many target-date mutual funds use it as their glidepath template.

## Primary academic references

- **Bodie, Merton & Samuelson (1992)**, *Labor Supply Flexibility and Portfolio
  Choice in a Life Cycle Model*, Journal of Economic Dynamics and Control. The
  canonical human-capital argument: young investors with large labour income are
  implicitly holding a bond-like asset, so their financial portfolio *should* be
  equity-heavy — and as human capital is consumed the financial portfolio should
  de-risk. The 100-minus-age rule is a rough heuristic approximation of this idea.
- **Samuelson (1969)**, *Lifetime Portfolio Selection by Dynamic Stochastic
  Programming*, Review of Economics and Statistics. The Merton-Samuelson theorem:
  with constant relative risk aversion, a constant equity fraction is optimal
  regardless of age — the direct theoretical counter-argument to any declining-equity
  rule. The rule of thumb violates this result unless risk aversion *rises* with age.
- **Shiller (2005)**, *Irrational Exuberance* (2nd ed.), Princeton University Press.
  The dataset underlying this study — long-run real returns for US equities and
  interest rates since 1871 — is drawn directly from Robert Shiller's publicly
  available dataset.

## The rising-equity glidepath (the central academic challenge to HMA)

- **Pfau, W. & Kitces, M. (2014)**, *Reducing Retirement Risk with a Rising Equity
  Glidepath*, Journal of Financial Planning. The most-cited challenge to the
  rule of thumb. The argument: sequence-of-returns risk is most dangerous *at* and
  *just after* retirement, not at entry. Starting with low equity and rising it
  through the accumulation phase means the investor arrives at retirement with
  *growing* equity exposure just as the sequence risk is highest — the opposite of
  the HMA prescription. This study includes the Pfau-Kitces variant as an explicit
  comparator. On the Shiller tape, Rising-equity earns statistically more than HMA
  (HAC t = +2.6) but still trails 60/40 (HAC t = −3.4).

## Sequence-of-returns risk and the 4% rule

- **Bengen, W. (1994)**, *Determining Withdrawal Rates Using Historical Data*,
  Journal of Financial Planning. The original paper behind the "4% rule": the worst
  historical 30-year withdrawal sequence was survivable at a 4% initial withdrawal
  rate with a 50-75% equity portfolio. The sequence-of-returns issue HMA tries to
  address is real; the question is whether HMA addresses it optimally.
- **Kitces, M. (2008)**, *Resolving the Paradox: Is the Safe Withdrawal Rate
  Sometimes Too Safe?*, The Kitces Report. Sequence-of-returns risk is asymmetric:
  early bear markets are catastrophic for the retiree; early bull markets allow
  ratcheting withdrawals. A flexible glidepath should respond to this asymmetry.

## Why 60/40 beats HMA here

- **DeMiguel, V., Garlappi, L. & Uppal, R. (2009)**, *Optimal Versus Naive
  Diversification: How Inefficient is the 1/N Portfolio Strategy?*, Review of
  Financial Studies. The surprising finding that naive equal weighting beats
  optimized portfolios out-of-sample — the general lesson that simple constant-mix
  strategies are hard to beat. A constant 60/40 is in this spirit.
- **Arnott, R. & Bernstein, P. (2002)**, *What Risk Premium is "Normal"?*,
  Financial Analysts Journal. Long-run equity risk premiums are lower than commonly
  assumed — historically ~2-4% real over bonds (consistent with the Shiller data's
  ~3.1% real ERP in this study). At moderate premiums, the return cost of
  de-risking is always proportional to how much equity is trimmed.

## Method and inference

- **Newey, W. & West, K. (1987)**, *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*,
  Econometrica. The HAC correction used here is essential: overlapping 40-year
  cohorts share up to 479 months of data, inducing massive positive serial
  correlation in the terminal-wealth differences. The NW bandwidth selector
  L = 4*(n/100)^(2/9) adjusts automatically.
- **Lo, A. (2002)**, *The Statistics of Sharpe Ratios*, Financial Analysts Journal.
  Standard guidance on annualising and inferring from overlapping return windows —
  motivation for the cohort-based (rather than daily) inference used in this study.

## Related desk studies

- **[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/)**: the
  25/25/25/25 Browne portfolio — a different flavour of the "de-risk via asset class
  diversification" argument. PP holds more equity than the aged-down HMA investor
  while including gold and cash as non-bond buffers. Real → FRAGILE.
- **[Study 68 — All-Weather](../../68-all-weather/)**: Bridgewater's risk-parity
  cousin; similar diversification claim tested with the same Shiller / cross-asset
  data pipeline.
- **[Study 102 — Free-Rebalance](../../102-free-rebalance/)**: examines whether
  annual rebalancing itself adds value vs a drifting portfolio — directly relevant
  to whether the HMA cost estimate is biased.
- **[Study 97 — Balancing-Act](../../97-balancing-act/)**: optimal rebalancing
  frequency — the sensitivity test that complements HMA's annual-rebalance
  assumption.
