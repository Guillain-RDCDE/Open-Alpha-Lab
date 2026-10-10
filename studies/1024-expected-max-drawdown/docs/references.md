# Sources & literature map — Study 1024 (The Drawdown You Were Promised)

## The claim: drawdown from mean and volatility

- **Magdon-Ismail, M., Atiya, A. F., Pratap, A. & Abu-Mostafa, Y. S. (2004), "On the Maximum
  Drawdown of a Brownian Motion", *Journal of Applied Probability* 41(1).** The distribution
  and expectation of the maximum drawdown of a Brownian motion with drift over a finite
  horizon; for zero drift, E[MDD] = √(π/2)·σ·√T. This is the promise the study writes down
  before every window, and the anchor §1 checks the simulation against.
- **Magdon-Ismail, M. & Atiya, A. F. (2004), "Maximum Drawdown", *Risk* (October 2004).** The
  practitioner version: expected drawdown as a function of (μ, σ, T) and its use to normalise
  the Calmar ratio across track-record lengths.
- **Young, T. W. (1991), "Calmar Ratio: A Smoother Tool", *Futures* magazine.** The origin of
  the return-over-maximum-drawdown ratio that the normalisation above corrects.
- **Grossman, S. J. & Zhou, Z. (1993), "Optimal Investment Strategies for Controlling
  Drawdowns", *Mathematical Finance* 3(3), 241-276.** Drawdown constraints as a portfolio
  objective — why the budget matters.
- **Chekhlov, A., Uryasev, S. & Zabarankin, M. (2005), "Drawdown Measure in Portfolio
  Optimization", *International Journal of Theoretical and Applied Finance* 8(1).** Conditional
  Drawdown-at-Risk; a beat-7 fork.

## Why the promise fails: the stylised facts it ignores

- **Cont, R. (2001), "Empirical Properties of Asset Returns: Stylized Facts and Statistical
  Issues", *Quantitative Finance* 1(2), 223-236.** Volatility clustering and heavy tails — the
  two things the synthetic `signal_strength` knob plants.
- **Bollerslev, T. (1986), "Generalized Autoregressive Conditional Heteroskedasticity",
  *Journal of Econometrics* 31(3), 307-327.** The GARCH(1,1) challenger.
- **Bollerslev, T. (1987), "A Conditionally Heteroskedastic Time Series Model for Speculative
  Prices and Rates of Return", *Review of Economics and Statistics* 69(3), 542-547.** GARCH with
  Student-*t* innovations — the exact challenger fitted here.

## Method

- **Broadie, M., Glasserman, P. & Kou, S. (1997), "A Continuity Correction for Discrete Barrier
  Options", *Mathematical Finance* 7(4), 325-349.** Why a discretely monitored extreme of a
  Brownian path sits systematically inside the continuous one — the §1 discretisation effect.
- **Politis, D. N. & Romano, J. P. (1994), "The Stationary Bootstrap", *Journal of the American
  Statistical Association* 89(428), 1303-1313.** The block-bootstrap challenger.
- **Künsch, H. R. (1989), "The Jackknife and the Bootstrap for General Stationary
  Observations", *Annals of Statistics* 17(3), 1217-1241.** The moving-block bootstrap used for
  overlapping windows.
- **Kupiec, P. H. (1995), "Techniques for Verifying the Accuracy of Risk Measurement Models",
  *Journal of Derivatives* 3(2), 73-84.** Coverage testing by counting breaches — the same logic
  applied here to a drawdown band instead of a one-day VaR.
- **Christoffersen, P. F. (1998), "Evaluating Interval Forecasts", *International Economic
  Review* 39(4), 841-862.**
- **Diebold, F. X., Gunther, T. A. & Tay, A. S. (1998), "Evaluating Density Forecasts with
  Applications to Financial Risk Management", *International Economic Review* 39(4), 863-883.**
  The probability-integral transform (PIT) check in §2 and the quants notebook.

## Data provenance

All four tapes are frozen inside Python packages and read through
[`quantlab/bundled.py`](../../../quantlab/bundled.py), which refuses any file whose SHA-256
differs from its pin:

| Tape | Source | Label |
|---|---|---|
| S&P 500 index | `skfolio` 1.8.1 wheel, `sp500_index.csv.gz` | **price index**, daily, 1990-01-02 → 2022-12-28 (cut at 2022-11-30) |
| Nasdaq Composite | `arch`, `data/nasdaq/nasdaq.csv.gz` (`Adj Close`) | **price index**, daily, 1999-01-04 → 2018-12-31 |
| US market | `arch`, `data/frenchdata/frenchdata.csv.gz` (Ken French's library, `Mkt-RF + RF`) | **total return**, monthly, 1926-07 → 2018-11 |
| 20 US stocks | `skfolio` 1.8.1 wheel, `sp500_dataset.csv.gz` | adjusted closes, daily — **survivor sample** |

Pins and content fingerprints are printed at the top of [`results.md`](results.md).

## Neighbours on this desk

**990-var-breach-count** (the same coverage logic for one-day VaR; this study does it for a
path-dependent, multi-year quantity), **991-aggregational-gaussianity** (how slowly returns
become Gaussian — the reason the Gaussian drawdown fails at one year),
**992-vol-clustering-halflife**, **813-maximum-drawdown-anomaly** (drawdown depth as a
cross-sectional *signal*), **816-drawdown-duration** (time underwater as a signal).
