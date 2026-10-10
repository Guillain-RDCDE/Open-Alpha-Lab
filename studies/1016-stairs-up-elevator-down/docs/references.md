# Sources & literature map — Study 1016 (Stairs Up, Elevator Down)

## The claim

*"Stocks take the stairs up and the elevator down"* is trading-floor folklore with no single
author. Its formal cousins are the "gain/loss asymmetry" stylised fact and the time-irreversibility
literature below. This study tests the folklore in its strongest form — as a statement about the
order of returns, not their size.

- **Cont, R. (2001), "Empirical Properties of Asset Returns: Stylized Facts and Statistical
  Issues", *Quantitative Finance* 1(2), 223-236.** Lists gain/loss asymmetry ("large drawdowns
  in stock prices and stock index values but not equally large upward movements") among the
  stylised facts, alongside the leverage effect.
- **Jensen, M. H., Johansen, A. & Simonsen, I. (2003), "Inverse Statistics in Economics: The
  Gain-Loss Asymmetry", *Physica A* 324.** The closest antecedent of section 1: the waiting time
  for a given gain or loss from an arbitrary starting day, finding losses reached sooner on the
  Dow. Our measure differs on purpose — first passage from a *confirmed turning point* with a
  log-symmetric threshold and a sign-flip null that keeps the drift — and on these tapes it gives
  the opposite answer.

## Time irreversibility

- **Ramsey, J. B. & Rothman, P. (1996), "Time Irreversibility and Business Cycle Asymmetry",
  *Journal of Money, Credit and Banking* 28(1), 1-21.** The `E[x_t² x_(t−k)] − E[x_t x_(t−k)²]`
  statistic of section 2.

## The mechanism: asymmetric volatility

- **Black, F. (1976), "Studies of Stock Price Volatility Changes", *Proceedings of the 1976
  Meetings of the American Statistical Association, Business and Economic Statistics Section*.**
  The leverage effect.
- **Bollerslev, T. (1986), "Generalized Autoregressive Conditional Heteroskedasticity",
  *Journal of Econometrics* 31(3), 307-327.** The symmetric GARCH null of section 4.
- **Nelson, D. B. (1991), "Conditional Heteroskedasticity in Asset Returns: A New Approach",
  *Econometrica* 59(2), 347-370.** EGARCH.
- **Glosten, L. R., Jagannathan, R. & Runkle, D. E. (1993), "On the Relation between the Expected
  Value and the Volatility of the Nominal Excess Return on Stocks", *Journal of Finance* 48(5),
  1779-1801.** GJR-GARCH — the generator whose leverage term `signal_strength` scales.
- **Engle, R. F. & Ng, V. K. (1993), "Measuring and Testing the Impact of News on Volatility",
  *Journal of Finance* 48(5), 1749-1778.** The news-impact curve.
- **Campbell, J. Y. & Hentschel, L. (1992), "No News Is Good News: An Asymmetric Model of
  Changing Volatility in Stock Returns", *Journal of Financial Economics* 31(3), 281-318.**
  Volatility feedback — the competing explanation for the same asymmetry.
- **Bekaert, G. & Wu, G. (2000), "Asymmetric Volatility and Risk in Equity Markets", *Review of
  Financial Studies* 13(1), 1-42.** Leverage versus volatility feedback, jointly.

## Skewness and crashes

- **Chen, J., Hong, H. & Stein, J. C. (2001), "Forecasting Crashes: Trading Volume, Past Returns,
  and Conditional Skewness in Stock Prices", *Journal of Financial Economics* 61(3), 345-381.**
- **Hong, H. & Stein, J. C. (2003), "Differences of Opinion, Short-Sales Constraints, and Market
  Crashes", *Review of Financial Studies* 16(2), 487-525.** A theory in which bad news is
  revealed abruptly — an elevator by construction.
- **Kim, T.-H. & White, H. (2004), "On More Robust Estimation of Skewness and Kurtosis", *Finance
  Research Letters* 1(1), 56-73.** Why section 3 carries a quantile skewness next to the moment
  one.

## Where the asymmetry is priced

- **Bates, D. S. (2000), "Post-'87 Crash Fears in the S&P 500 Futures Option Market", *Journal of
  Econometrics* 94(1-2), 181-238.**
- **Bakshi, G., Kapadia, N. & Madan, D. (2003), "Stock Return Characteristics, Skew Laws, and the
  Differential Pricing of Individual Equity Options", *Review of Financial Studies* 16(1),
  101-143.** Links the return asymmetry to the slope of the implied-volatility smile. Not tested
  here (no options tape).

## Method

- **Newey, W. K. & West, K. D. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix", *Econometrica* 55(3), 703-708.**
- **Politis, D. N. & Romano, J. P. (1992), "A Circular Block-Resampling Procedure for Stationary
  Data", in LePage, R. & Billard, L. (eds.), *Exploring the Limits of Bootstrap*, Wiley.** The
  circular block bootstrap used for every interval and Sharpe difference here.
- **Sheppard, K., `arch` — ARCH models and financial econometrics in Python (software).** Used for
  the GARCH / GJR / EGARCH fits.

## Data provenance

- **S&P 500 price index**, daily 1990-01-02 → 2022-12-28 (used to 2022-11-30), shipped in the
  `skfolio` 1.8.1 wheel (`sp500_index.csv.gz`), fetched once by exact PyPI URL and SHA-256-pinned
  in [`quantlab/bundled.py`](../../../quantlab/bundled.py). Price-only.
- **Nasdaq Composite**, daily 1999-01-04 → 2018-12-31, shipped in `arch` (`nasdaq.csv.gz`,
  `Adj Close`), SHA-256-pinned. Price-only.
- **Fama-French US market**, monthly 1926-07 → 2018-11, `Mkt-RF + RF` (total return) and `RF`,
  shipped in `arch` (`frenchdata.csv.gz`), SHA-256-pinned; originally from the Kenneth R. French
  Data Library. `RF` is also the cash rate for every trading race.

## Neighbours on this desk

**993-leverage-effect-asymmetry** (the volatility response this study finds to be the mechanism),
**991-aggregational-gaussianity** (tails by horizon), **992-vol-clustering-halflife**,
**816-drawdown-duration** (time underwater as a predictor), **867-currency-crash-risk** (carry
skewness), **371-skew-index** and **617-crash-insurance-cost** (where the asymmetry is priced).
