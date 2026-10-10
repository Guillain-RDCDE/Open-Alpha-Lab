# Sources & literature map — Study 1013 (It's Official)

## The chronology under test

- **NBER Business Cycle Dating Committee — US business cycle expansions and contractions, and
  the committee's announcements index** (https://www.nber.org/research/business-cycle-dating).
  The source of every hard-coded date in `nberclock/data.py`: the sixteen monthly peak → trough
  pairs since 1926 and the twelve announcement dates since the committee began announcing
  turning points in 1980. The committee states explicitly that it dates turning points after
  the fact and does not forecast; the lag the folk claim exploits is a design choice, not a
  failing.
- **Burns, A. F. & Mitchell, W. C. (1946), *Measuring Business Cycles*, NBER.** The
  methodological origin of NBER reference-cycle dating.

## Turning points in markets and in the economy

- **Bry, G. & Boschan, C. (1971), *Cyclical Analysis of Time Series: Selected Procedures and
  Computer Programs*, NBER.** The classic algorithm for locating peaks and troughs in a series.
- **Pagan, A. R. & Sossounov, K. A. (2003), "A Simple Framework for Analysing Bull and Bear
  Markets", *Journal of Applied Econometrics* 18(1), 23-46.** Adapts Bry-Boschan to stock
  prices. This study's turning-point rule is deliberately simpler (a fixed window around each
  NBER date) and its bias is measured, not assumed away: see `lead_rotation_test`.
- **Siegel, J. J. (1991), "Does It Pay Stock Investors to Forecast the Business Cycle?",
  *Journal of Portfolio Management* 18(1), 27-34.** The direct antecedent: stock peaks and troughs lead the
  NBER's, and the gain from timing them requires knowing the turning point in real time — which
  the official announcement, by construction, does not provide.
- **Fama, E. F. & French, K. R. (1989), "Business Conditions and Expected Returns on Stocks and
  Bonds", *Journal of Financial Economics* 25(1), 23-49.** Expected returns are higher in bad
  times — the economic reason one might expect post-recession-announcement returns to be high,
  and the effect a 12-event test cannot resolve (section 5 of the results).

## Real-time recession signals (what moves *before* the committee)

- **Hamilton, J. D. (1989), "A New Approach to the Economic Analysis of Nonstationary Time
  Series and the Business Cycle", *Econometrica* 57(2), 357-384.** Regime-switching recession
  probabilities.
- **Stock, J. H. & Watson, M. W. (1989), "New Indexes of Coincident and Leading Economic
  Indicators", *NBER Macroeconomics Annual* 4.**
- **Estrella, A. & Mishkin, F. S. (1998), "Predicting U.S. Recessions: Financial Variables as
  Leading Indicators", *Review of Economics and Statistics* 80(1), 45-61.** The yield curve and
  the stock market itself as leading indicators of the cycle the NBER later dates.
- **Chauvet, M. & Piger, J. (2008), "A Comparison of the Real-Time Performance of Business
  Cycle Dating Methods", *Journal of Business & Economic Statistics* 26(1).** Statistical
  models can call turning points well before the committee does — the benchmark the
  announcement is slow against.

## Method

- **Newey, W. K. & West, K. D. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix", *Econometrica* 55(3), 703-708.** Active-return
  t-statistics on the timing rules.
- **Good, P. I. (2005), *Permutation, Parametric, and Bootstrap Tests of Hypotheses*, 3rd ed.,
  Springer.** Randomisation inference. The rotation (circular-shift) test used here shifts the
  whole set of event dates as a block, preserving their spacing and the market's own serial
  dependence; with every shift enumerated it is exact and needs no seed.

## Data provenance

- **Monthly:** Kenneth R. French Data Library, Fama-French research factors (Mkt-RF, RF),
  1926-07 → 2018-11, as frozen inside the `arch` package (`arch/data/frenchdata/frenchdata.csv.gz`,
  SHA-256 `f23436727b01d879e1372e3ee277514b0962a23d2b158f1aee6a402a90994f52`). `Mkt-RF + RF` is
  the value-weighted US market **total return**.
- **Daily:** S&P 500 index level, 1990-01-02 → 2022-12-28, as frozen inside `skfolio` 1.8.1
  (`skfolio/datasets/data/sp500_index.csv.gz`, SHA-256
  `894c34431c284f86aa64a467e176e2a3c9b697c857eb9ab35006c2dd55d3fdf2`). A **price index**: no
  dividends. December 2022 is a partial month and is dropped (as-of 2022-11-30).
- Both are loaded through [`quantlab/bundled.py`](../../../quantlab/bundled.py), which refuses a
  file whose bytes differ from the pin. Fingerprints of the frames actually used are printed in
  [`results.md`](results.md).

## Neighbours on this desk

- **268-sahm-rule** — a real-time *unemployment* recession trigger as a sell button; this study
  tests the official, after-the-fact *dating* announcement, and as a buy signal.
- **626-unemployment-trend-timing** — unemployment as a veto on a trend-following rule.
- **881-jobless-claims-nowcast** — claims as a nowcast for sector rotation.
