# References — Study 1022 (Behind the Curve)

## The claim and the rule

- **Taylor, J. B. (1993).** "Discretion versus policy rules in practice." *Carnegie-Rochester
  Conference Series on Public Policy*, 39, 195–214. — The rule tested here, verbatim:
  `i* = r* + π + 0.5(π − π*) + 0.5·gap`, `r* = π* = 2`. Taylor fitted it, loosely, to 1987–92.
- **Clarida, R., Galí, J. and Gertler, M. (2000).** "Monetary Policy Rules and Macroeconomic
  Stability: Evidence and Some Theory." *Quarterly Journal of Economics*, 115(1), 147–180. — The
  pre-1979 Fed responded too weakly to inflation: the canonical "behind the curve" era, and the
  reason the regime split in §5 of the results is at least plausible ex ante.

## The real-time critique (the study's main caveat)

- **Orphanides, A. (2001).** "Monetary Policy Rules Based on Real-Time Data." *American Economic
  Review*, 91(4), 964–985. — Rules evaluated on revised data describe a Fed that never existed:
  real-time output-gap estimates differed greatly from the final ones. Our macro is final
  vintage, so every result is an upper bound on what an investor could have known.
- **Orphanides, A. and van Norden, S. (2002).** "The Unreliability of Output-Gap Estimates in
  Real Time." *Review of Economics and Statistics*, 84(4), 569–583. — The end-point problem of
  one-sided filters (which this study reproduces) and the larger data-revision problem (which it
  cannot).
- **Croushore, D. and Stark, T. (2001).** "A real-time data set for macroeconomists." *Journal of
  Econometrics*, 105(1), 111–130. — The vintage data that would turn this study's upper bound
  into a real-time test (beat 7).

## Output gaps

- **Hodrick, R. J. and Prescott, E. C. (1997).** "Postwar U.S. Business Cycles: An Empirical
  Investigation." *Journal of Money, Credit and Banking*, 29(1), 1–16. — The HP filter, λ = 1600
  for quarterly data.
- **Hamilton, J. D. (2018).** "Why You Should Never Use the Hodrick-Prescott Filter." *Review of
  Economics and Statistics*, 100(5), 831–843. — The case against HP (including its end-point
  behaviour); the unemployment gap is our filter-free cross-check.
- **Okun, A. M. (1962).** "Potential GNP: Its Measurement and Significance." *Proceedings of the
  Business and Economic Statistics Section, American Statistical Association*. — Okun's law; the
  coefficient of 2 used for the unemployment gap is the conventional modern rounding.

## Predictive-regression inference

- **Stambaugh, R. F. (1999).** "Predictive regressions." *Journal of Financial Economics*, 54(3),
  375–421. — Small-sample bias of the slope on a persistent regressor whose innovations correlate
  with returns; the first-order correction used in §3.
- **Kendall, M. G. (1954).** "Note on bias in the estimation of autocorrelation." *Biometrika*,
  41, 403–404. — The `−(1 + 3ρ)/T` bias of the AR(1) coefficient inside the Stambaugh formula.
- **Newey, W. K. and West, K. D. (1987).** "A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703–708.
- **Hodrick, R. J. (1992).** "Dividend Yields and Expected Stock Returns: Alternative Procedures
  for Inference and Measurement." *Review of Financial Studies*, 5(3), 357–386. — Why overlapping
  long-horizon regressions over-reject; we answer it with a simulation null instead.
- **Ang, A. and Bekaert, G. (2007).** "Stock Return Predictability: Is It There?" *Review of
  Financial Studies*, 20(3), 651–707. — Long-horizon predictability largely disappears under
  correct inference; the template for the 4- and 8-quarter tests.
- **Welch, I. and Goyal, A. (2008).** "A Comprehensive Look at the Empirical Performance of
  Equity Premium Prediction." *Review of Financial Studies*, 21(4), 1455–1508. — Macro predictors
  of the equity premium rarely survive out of sample; the prior this study confirms.

## Data provenance

- **US quarterly macro** — `statsmodels.datasets.macrodata` (1959Q1–2009Q3; real GDP, CPI-U,
  3-month T-bill quarterly average, unemployment), **final vintage**, SHA-256 pinned in
  `taylorgap/data.py` (`MACRODATA_SHA256`).
- **Equity total return and T-bill** — Fama-French monthly factors (Kenneth R. French Data
  Library) as shipped in the `arch` package (`frenchdata`), via `quantlab.bundled`
  (SHA-256 pinned); `Mkt-RF + RF` is a total return. Fama, E. F. and French, K. R. (1993),
  "Common risk factors in the returns on stocks and bonds," *Journal of Financial Economics*,
  33(1), 3–56.
- **Moody's AAA corporate yield** — monthly, as shipped in `arch` (`default`), via
  `quantlab.bundled` (SHA-256 pinned). The long-bond return is an **approximation** from yield
  changes (duration + convexity of a 20-year par bond); see `data.long_bond_return`.

## Neighbours on this desk

- [118-fed-model](../../118-fed-model/) — earnings yield minus bond yield as a stock/bond timer.
- [119-real-rate-regime](../../119-real-rate-regime/) — stepping aside when real long rates are high.
- [925-short-rate-momentum-switch](../../925-short-rate-momentum-switch/) — the bill yield's own trend as a duration switch.
- [985-last-hike-timing](../../985-last-hike-timing/) — buying the end of a tightening cycle (and the recognition lag).
- [924-cut-cycle-duration-extension](../../924-cut-cycle-duration-extension/) — buying duration at the first cut.
- [1014-macro-clairvoyance](../../1014-macro-clairvoyance/) — what a perfect macro forecast would be worth.
