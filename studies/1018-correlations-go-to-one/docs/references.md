# Sources & literature map — Study 1018 (Correlations Go to One)

## The claim

- **Page, S. & Panariello, R. A. (2018), "When Diversification Fails", *Financial Analysts
  Journal* 74(3).** The practitioner statement of the claim at full strength: tail correlations
  rise and diversification weakens in sell-offs. This is the version the study steelmans.
- **Longin, F. & Solnik, B. (1995), "Is the Correlation in International Equity Returns Constant:
  1960–1990?", *Journal of International Money and Finance* 14(1), 3-26.** An early formal
  rejection of constant correlation, with correlation rising in high-volatility periods.

## The conditioning artefact

- **Boyer, B. H., Gibson, M. S. & Loretan, M. (1997, revised 1999), "Pitfalls in Tests for
  Changes in Correlations", Board of Governors of the Federal Reserve System, International
  Finance Discussion Paper 597.** Conditioning on a subsample selected by its realisations
  changes the measured correlation even when the true one is constant. Section 2a of the results
  and `strategy.bgl_conditional_corr` implement their closed form.
- **Forbes, K. J. & Rigobon, R. (2002), "No Contagion, Only Interdependence: Measuring Stock
  Market Comovements", *Journal of Finance* 57(5), 2223-2261.** The heteroskedasticity-adjusted
  correlation (`strategy.fr_adjust`). Section 2c applies it and explains why "no contagion" is not
  the same thing as "no harm to a diversified holder".

## Tail and asymmetric dependence

- **Longin, F. & Solnik, B. (2001), "Extreme Correlation of International Equity Markets",
  *Journal of Finance* 56(2), 649-676.** Exceedance correlation and the observation that
  correlation rises in bear markets but not in bull markets.
- **Ang, A. & Chen, J. (2002), "Asymmetric Correlations of Equity Portfolios", *Journal of
  Financial Economics* 63(3), 443-494.** The exceedance correlation compared against its Gaussian
  benchmark, and the H statistic used in section 3.

## Volatility, correlation models and the holder

- **Bollerslev, T. (1986), "Generalized Autoregressive Conditional Heteroskedasticity", *Journal
  of Econometrics* 31(3), 307-327.** The univariate GARCH(1,1) fitted to each series.
- **Bollerslev, T. (1990), "Modelling the Coherence in Short-Run Nominal Exchange Rates: A
  Multivariate Generalized ARCH Model", *Review of Economics and Statistics* 72(3), 498-505.**
  The constant-conditional-correlation model that serves as the exact null.
- **Engle, R. (2002), "Dynamic Conditional Correlation: A Simple Class of Multivariate
  Generalized Autoregressive Conditional Heteroskedasticity Models", *Journal of Business &
  Economic Statistics* 20(3), 339-350.** The natural next step (beat 7): a continuous
  conditional-correlation path in place of regimes.
- **Campbell, J. Y., Lettau, M., Malkiel, B. G. & Xu, Y. (2001), "Have Individual Stocks Become
  More Volatile? An Empirical Exploration of Idiosyncratic Risk", *Journal of Finance* 56(1),
  1-43.** Idiosyncratic and market volatility move separately. Their ratio is what correlation
  responds to, and why the constant-beta benchmark over-predicts here.
- **Choueifaty, Y. & Coignard, Y. (2008), "Toward Maximum Diversification", *Journal of
  Portfolio Management* 35(1), 40-51.** The diversification ratio used in section 4.
- **Pollet, J. M. & Wilson, M. (2010), "Average Correlation and Stock Market Returns", *Journal
  of Financial Economics* 96(3), 364-380.** Average pairwise correlation as an economic state
  variable. Their question (expected returns) belongs to neighbour 578, not this study.

## Method

- **Politis, D. N. & Romano, J. P. (1992), "A Circular Block-Resampling Procedure for Stationary
  Data", in R. LePage & L. Billard (eds.), *Exploring the Limits of Bootstrap*, Wiley.** The
  circular block bootstrap behind every interval (63-day blocks, regime labels carried with the
  days).

## Data provenance

- **skfolio 1.8.1** wheel from PyPI (SHA-256 `5ad89bcd52513b1e400ed1fe840db99d0a86af0ca9842c0e790e3618fe234852`),
  read through [`quantlab/bundled.py`](../../../quantlab/bundled.py), which pins each file:
  - `sp500_dataset.csv.gz`: daily adjusted closes of 20 large US stocks, 1990-01-02 → 2022-12-28.
    It is a **survivor sample**. The results caveats reason out the direction of the bias for
    correlation measurement.
  - `sp500_index.csv.gz`: the daily S&P 500 **price** index (no dividends). It is used only as
    the market state variable and as the market leg of the exceedance correlations.
- As-of **2022-11-30**, the last full month (the partial December 2022 is dropped). Fingerprints
  are printed in [results.md](results.md).

## Neighbours on this desk

**578-cross-asset-correlation-regime** (correlation regime as a forward-return predictor),
**1010-correlation-matrix-stability** (how much of a correlation matrix is noise),
**502-betting-against-correlation** (correlation as a cross-sectional anomaly),
**974-diversification-saturation** (how many assets before diversification stops paying),
**1004-how-many-stocks**.
