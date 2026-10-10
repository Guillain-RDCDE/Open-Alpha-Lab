# Sources & literature map — Study 1026 (Gibson's Paradox)

## The claim and its history

- **Gibson, A. H. (1923), "The Future Course of High-Class Investment Values", *Bankers',
  Insurance Managers', and Agents' Magazine*.** The original observation: British consol yields
  moved with wholesale prices.
- **Keynes, J. M. (1930), *A Treatise on Money*, Macmillan.** Names "Gibson's paradox" and calls
  it one of the best-established empirical facts in quantitative economics.
- **Fisher, I. (1930), *The Theory of Interest*, Macmillan.** The Fisher effect — nominal rates
  compensate for expected *inflation* — and Fisher's own reconciliation of Gibson's correlation:
  inflation expectations formed over a long distributed lag of past inflation. §5 of this study is
  that idea, tested.
- **Sargent, T. J. (1973), "Interest Rates and Prices in the Long Run: A Study of the Gibson
  Paradox", *Journal of Money, Credit and Banking* 5(1), Part 2, 385-449.** A rational-expectations
  analysis of the Gibson correlation and of Fisher's distributed-lag explanation of it.
- **Shiller, R. J. & Siegel, J. J. (1977), "The Gibson Paradox and Historical Movements in Real
  Interest Rates", *Journal of Political Economy* 85(5).**
- **Barsky, R. B. & Summers, L. H. (1988), "Gibson's Paradox and the Gold Standard", *Journal of
  Political Economy* 96(3).** The paradox as a gold-standard phenomenon: real rates set the
  relative price of gold, hence the price level. The reason this study frames its fiat-era test
  as a test of the *modern* claim, not of the gold-standard regularity.

## Method

- **Granger, C. W. J. & Newbold, P. (1974), "Spurious Regressions in Econometrics", *Journal of
  Econometrics* 2(2).** Why level correlations of trending series are not evidence (§2, and the
  random-walk demonstration in the curious notebook).
- **Dickey, D. A. & Fuller, W. A. (1979), "Distribution of the Estimators for Autoregressive Time
  Series with a Unit Root", *Journal of the American Statistical Association* 74.**
- **Kwiatkowski, D., Phillips, P. C. B., Schmidt, P. & Shin, Y. (1992), "Testing the Null
  Hypothesis of Stationarity against the Alternative of a Unit Root", *Journal of Econometrics*
  54.** KPSS — paired with ADF so that both nulls are tested.
- **Engle, R. F. & Granger, C. W. J. (1987), "Co-integration and Error Correction:
  Representation, Estimation, and Testing", *Econometrica* 55(2).** The two-step cointegration
  test behind leg L1.
- **Newey, W. K. & West, K. D. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix", *Econometrica* 55(3).**
- **Clark, T. E. & West, K. D. (2007), "Approximately Normal Tests for Equal Predictive Accuracy
  in Nested Models", *Journal of Econometrics* 138(1).** The out-of-sample test behind leg L3.
- **Campbell, J. Y. & Thompson, S. B. (2008), "Predicting Excess Stock Returns Out of Sample: Can
  Anything Beat the Historical Average?", *Review of Financial Studies* 21(4).** The
  prevailing-mean benchmark used to demean the timing signal.

## Data provenance

All series come from tapes shipped inside Python packages, read through
[`quantlab/bundled.py`](../../../quantlab/bundled.py); the `arch` files are SHA-256 pinned and a
changed byte fails loudly.

- `arch` · `default` — Moody's seasoned AAA and BAA corporate bond yields, monthly, percent,
  1919-01 → 2018-12 (originally from FRED).
- `arch` · `core_cpi` — US CPI less food and energy (`CPILFESL`, seasonally adjusted), monthly,
  1957-01 → 2018-11 (FRED). **The earliest price index on any bundled tape — the reason the
  gold-standard era cannot be tested here.**
- `arch` · `frenchdata` — Fama-French monthly `RF` (one-month T-bill), used as the cash leg.
- `statsmodels` · `macrodata` — US quarterly macro, final-vintage, 1959Q1 → 2009Q3 (headline CPI).
- `statsmodels` · `interest_inflation` — German quarterly `Dp` (Δ log GDP deflator, not
  seasonally adjusted) and `R` (long-term rate), 1972Q2 → 1998Q4.

Fingerprints of the exact frames used are printed at the top of [results.md](results.md).

## Neighbours on this desk

**152-inflation-hedge** (do stocks hedge inflation — Fama-Schwert), **119-real-rate-regime**
(equity timing on real long rates), **118-fed-model** (earnings yield vs bond yield),
**625-starting-yield-bond-decade** (the yield as the forecast of bond returns),
**581-term-premium** (term-premium timing of long bonds), **644-cpi-day-drift** (bond moves on
CPI release days), **924-cut-cycle-duration-extension** (duration timing on Fed cuts).
