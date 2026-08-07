# References & literature map — Study 835 (Spurious Regression)

## The claim under test

- **The source paper.** Clive W. J. **Granger & Paul Newbold** (1974), *"Spurious Regressions in
  Econometrics."* *Journal of Econometrics* 2(2), 111–120. The founding demonstration: regress two
  **independent random walks** on one another and OLS routinely returns a large, "significant"
  *t*-statistic, a high R², and a low Durbin-Watson — a textbook relation that does not exist. Their
  Monte Carlo found the null of "no relation" rejected the great majority of the time at the nominal
  5% level. Their prescription: be suspicious when R² > DW, and **difference** the series (or model in
  changes) before trusting the regression.
- **The theory that explains it.** Peter C. B. **Phillips** (1986), *"Understanding Spurious
  Regressions in Econometrics."* *Journal of Econometrics* 33(3), 311–340. The asymptotic theory: with
  `I(1)` regressors the usual *t*-statistic **diverges** (it does *not* converge to a fixed
  distribution), so the rejection rate → 1 as the sample grows — the "more data makes it worse" result
  reproduced in this study's sample-size sweep. The R² converges to a non-degenerate random variable
  rather than to zero.
- **The specific test here.** We simulate many pairs of independent random walks, run the level OLS,
  and record the slope *t*-stat and R²; we then apply the two textbook cures — **first-differencing**
  and a **cointegration test** — and a **stationary-series control** to prove the inflation is a
  property of nonstationarity, not of OLS. Every claim averages over thousands of pairs; the tradable
  angle is checked with a costed, look-ahead-free pairs timer.

## The fixes, and why they work

- **First-differencing.** The difference of a random walk is (by construction) its `I(0)` white-noise
  increment; regressing `Δy` on `Δx` for two independent walks is a regression of one white-noise
  series on another, which is correctly sized (~5% rejection, R² ≈ 0). The cost is that differencing
  throws away any genuine *long-run* (cointegrating) relationship — which is why cointegration testing
  exists.
- **Cointegration.** Robert F. **Engle & Clive W. J. Granger** (1987), *"Co-integration and Error
  Correction: Representation, Estimation, and Testing."* *Econometrica* 55(2), 251–276. Two `I(1)`
  series are *cointegrated* if some linear combination `y − βx` is stationary — a genuine long-run
  relation that a levels regression can legitimately estimate. The Engle-Granger two-step test (regress
  in levels, then test the residual for a unit root) distinguishes a real long-run relation from a
  spurious one; this study uses `statsmodels.tsa.stattools.coint`, which implements it.
- **Unit-root testing.** David A. **Dickey & Wayne A. Fuller** (1979), *"Distribution of the Estimators
  for Autoregressive Time Series with a Unit Root."* *Journal of the American Statistical Association*
  74(366), 427–431. The augmented Dickey-Fuller test underneath the cointegration residual check — the
  standard way to ask "is this series stationary, or does it have a unit root?".

## What we measure, and the honesty rails

- **The classical OLS *t* is exactly the culprit.** We compute the textbook slope *t* =
  `β / sqrt( SSE/(n−2) / Sxx )` — the same statistic Granger & Newbold's critique targets — and the
  regression R², both vectorised across thousands of pairs.
- **Correct size is the benchmark.** A valid 5% test rejects ~5% of a true null; the *distance* of the
  observed rejection rate from 0.05 (a 17× gap on driftless walks, a 20× gap with a trend) is the
  measure of the pitfall. A Wilson score interval bands the rejection rate as the binomial proportion
  it is.
- **The control is a machinery proof, never market evidence.** The stationary-series size control and
  the cointegration positive control show the estimators are unbiased (correctly sized on the null,
  powered on a planted relation). Per house methodology, a synthetic control can never earn `REAL`,
  which requires a robust *t* ≥ 2 on a **real tape** — which a synthetic-only method demo does not
  have.
- **Tradability is graded separately and costed.** The pairs timer uses a trailing (out-of-sample)
  hedge ratio and z-score, charges one-way cost × NAV on turnover plus short borrow, and reports the
  net *t* and Sharpe.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent covariance (the
  HAC *t* primitive shared with the desk's template).
- **Wilson, E. B. (1927)** — score interval for a binomial share (used on the rejection rate).

## Neighbours on this bench (the dedup map — what this study is NOT)

- **[346-multiple-testing](../../346-multiple-testing/)** — inflated significance from running **many
  hypotheses** and keeping the winners (a multiple-comparisons / family-wise-error problem). Study 835
  inflates significance from a **single** regression on **nonstationary** data — the *t*-stat is
  oversized even with *one* test, before any selection.
- **[348-curve-fitting](../../348-curve-fitting/)** — over-flexible models that **memorise** in-sample
  noise (an over-parameterisation / bias-variance problem). Study 835's regression is a *simple*,
  correctly specified two-variable OLS; the failure is in the **data's unit root**, not model
  complexity.
- **[343-data-mining-roulette](../../343-data-mining-roulette/)** — false "edges" from **searching** a
  large space of signals until one looks good (selection over trials). Study 835 does **no search**: a
  single level regression on two independent walks is already grossly over-sized — the pitfall is the
  nonstationarity, not the mining.
- Sibling *method demos* on this desk: **[344-backtest-overfitting](../../344-backtest-overfitting/)**
  and **[590-sharpe-hacking](../../590-sharpe-hacking/)** — the same synthetic-only, three-axis
  (Signal `NONE` / Tradability `MIRAGE` / myth `CONFIRMED`) template, applied to overfitting-by-search
  and metric-gaming respectively rather than to nonstationary inference.

## Data sources

- **None external.** All worlds are deterministic seeded simulations (base seed 835) built in
  [`spurious_regression/data.py`](../spurious_regression/data.py). No network, no real market data.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a synthetic
  control is a machinery proof, never market evidence; `REAL` needs a robust *t* ≥ 2 on a real tape),
  costed gross/net labelling, and the reproducibility fingerprint stamp.
