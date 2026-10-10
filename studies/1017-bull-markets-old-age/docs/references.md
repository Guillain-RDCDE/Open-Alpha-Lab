# Sources & literature map — Study 1017 (Do Bull Markets Die of Old Age?)

## The claim and its business-cycle ancestor

- **"Long in the tooth" / "late cycle"** — market-commentary shorthand for the belief that an old
  bull is a fragile bull. Its academic twin is the question of whether *economic expansions* die of
  old age, which macroeconomists have tested with duration models for decades.
- **Diebold, F. X. & Rudebusch, G. D. (1990), "A Nonparametric Investigation of Duration Dependence
  in the American Business Cycle", *Journal of Political Economy* 98(3), 596-616.** The template for
  this study: test whether the hazard of an expansion ending rises with its age, and treat the
  answer as an empirical question rather than folklore.
- **Rudebusch, G. D. (2016), "Will the Economic Recovery Die of Old Age?", *FRBSF Economic Letter*
  (Federal Reserve Bank of San Francisco).** A short, readable statement of the same question for
  the post-2009 expansion.

## Bull and bear markets: dating and duration dependence

- **Bry, G. & Boschan, C. (1971), *Cyclical Analysis of Time Series: Selected Procedures and
  Computer Programs*, NBER Technical Paper 20.** The turning-point algorithm behind business-cycle
  dating, with minimum-phase censoring rules.
- **Pagan, A. R. & Sossounov, K. A. (2003), "A Simple Framework for Analysing Bull and Bear
  Markets", *Journal of Applied Econometrics* 18(1), 23-46.** Adapts Bry-Boschan to stock prices
  and compares the features of dated bull and bear markets with those that simple
  return-generating models produce under the same dating algorithm. Our random-walk and GARCH
  nulls apply that idea to the hazard. We did *not*
  implement their censoring rules (a fork; see the notebooks' beat 7).
- **Lunde, A. & Timmermann, A. (2004), "Duration Dependence in Stock Prices: An Analysis of Bull and
  Bear Markets", *Journal of Business & Economic Statistics* 22(3), 253-273.** Dates bulls and bears
  with percentage-move filters (possibly asymmetric between up and down moves) and models how the
  hazard of a phase ending depends on its age. Our "+20% / −15%" row is a filter in that spirit.
- **Maheu, J. M. & McCurdy, T. H. (2000), "Identifying Bull and Bear Markets in Stock Returns",
  *Journal of Business & Economic Statistics* 18(1), 100-112.** Duration-dependent Markov switching:
  the hazard modelled inside the return process rather than on dated cycles.
- **Gonzalez, L., Powell, J. G., Shi, J. & Wilson, A. (2005), "Two Centuries of Bull and Bear
  Market Cycles", *International Review of Economics & Finance*.** Long-sample dating of US cycles.

## Duration models and inference

- **Kiefer, N. M. (1988), "Economic Duration Data and Hazard Functions", *Journal of Economic
  Literature* 26(2), 646-679.** Hazards, censoring and the Weibull as the workhorse
  duration-dependence model (shape k = 1 memoryless, k > 1 positive dependence).
- **Bollerslev, T. (1986), "Generalized Autoregressive Conditional Heteroskedasticity", *Journal of
  Econometrics* 31(3), 307-327.** The GARCH(1,1) used for the second null (Student-t shocks, fitted
  with the `arch` package).
- **Newey, W. K. & West, K. D. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix", *Econometrica* 55(3), 703-708.** HAC errors for the
  overlapping 12-month predictive regression.
- **Stambaugh, R. F. (1999), "Predictive Regressions", *Journal of Financial Economics* 54(3),
  375-421.** Why a persistent regressor (bull age is a ramp) biases predictive regressions — the
  reason we also report a simulated null for the HAC t-stat.
- **Künsch, H. R. (1989), "The Jackknife and the Bootstrap for General Stationary Observations",
  *Annals of Statistics* 17(3), 1217-1241.** Block bootstrap; we use the circular variant for the
  Sharpe-difference test.

## Data provenance

- **Fama-French monthly market factor** (`Mkt-RF` + `RF`, i.e. the value-weighted CRSP market
  **total return** and the one-month T-bill), 1926-07 → 2018-11, as shipped inside the `arch`
  Python package (`arch/data/frenchdata/frenchdata.csv.gz`, SHA-256 pinned in
  [`quantlab/bundled.py`](../../../quantlab/bundled.py)). Original source: Kenneth R. French's data
  library; factor construction per **Fama, E. F. & French, K. R. (1993), "Common Risk Factors in the
  Returns on Stocks and Bonds", *Journal of Financial Economics* 33(1), 3-56.**
- **S&P 500 index, daily close — price index, no dividends**, 1990-01-02 → 2022-12-28, as shipped
  inside the `skfolio` 1.8.1 wheel (`sp500_index.csv.gz`, SHA-256 pinned). Stamped run stops at the
  last full month, 2022-11-30.
- Both are nominal USD. Fingerprints of the exact frames used are printed in
  [`results.md`](results.md).

## Neighbours on this desk

**81-four-year-itch** (a calendar cycle in returns), **816-drawdown-duration** (time underwater in
the cross-section), **333-recovery-speed** (who leads out of a drawdown), and the desk's valuation
and regime studies (e.g. **349-regime-dependence**), which ask the better-posed version of the
"late cycle" question: is it *price*, not *age*, that predicts the end?
