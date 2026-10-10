# Sources & literature map — Study 1021 (Stability Breeds Instability)

## The claim

- **Minsky, H. P. (1986), *Stabilizing an Unstable Economy*, Yale University Press.** The
  financial-instability hypothesis at book length: tranquil periods breed the speculative and
  Ponzi financing that makes the system fragile. "Stability is destabilising" is the shorthand.
- **Danielsson, J., Valenzuela, M. & Zer, I. (2018), "Learning from History: Volatility and
  Financial Crises", *Review of Financial Studies* 31.** The empirical version this study
  steelmans. Across many countries and two centuries, long periods of stock-market volatility
  *below* its slow-moving trend predict banking crises, and high volatility does not. Two
  differences: they predict **crises** across countries, while we test the practitioner's
  translation (US equity **drawdowns**), and our trend is a trailing mean so the signal is
  strictly past-only.
- **Brunnermeier, M. K. & Sannikov, Y. (2014), "A Macroeconomic Model with a Financial
  Sector", *American Economic Review* 104(2), 379-421.** The "volatility paradox" in a model:
  lower fundamental risk leads to higher leverage and *more* endogenous risk. This is the
  theoretical reason to expect a positive calm → crash slope.
- **Adrian, T. & Shin, H. S. (2010), "Liquidity and Leverage", *Journal of Financial
  Intermediation* 19(3), 418-437.** Balance sheets expand when measured risk is low (value-at-risk
  constraints loosen). This is the mechanism behind "agents take more risk when risk looks low".
- **Whaley, R. E. (2000), "The Investor Fear Gauge", *Journal of Portfolio Management* 26(3),
  12-17.** Where the VIX-as-sentiment framing, and so "low VIX = complacency", comes from.

## The confound — volatility models

- **Bollerslev, T. (1986), "Generalized Autoregressive Conditional Heteroskedasticity",
  *Journal of Econometrics* 31(3), 307-327.** GARCH(1,1): the first no-feedback null.
- **Baillie, R. T., Bollerslev, T. & Mikkelsen, H. O. (1996), "Fractionally Integrated
  Generalized Autoregressive Conditional Heteroskedasticity", *Journal of Econometrics* 74(1),
  3-30.** FIGARCH: long-memory volatility, the second and wider null. Its slow component
  wanders on decade scales, as the calm score does.
- **Moreira, A. & Muir, T. (2017), "Volatility-Managed Portfolios", *Journal of Finance* 72(4),
  1611-1644.** The vol-target comparator. Their effect runs through *current* vol at short
  horizons, the clustering leg that §6 of the results confirms.

## Method

- **Newey, W. K. & West, K. D. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix", *Econometrica* 55(3), 703-708.** HAC errors for
  overlapping forward windows (2H lags).
- **Politis, D. N. & Romano, J. P. (1992), "A Circular Block-Resampling Procedure for Stationary
  Data", in LePage, R. & Billard, L. (eds), *Exploring the Limits of Bootstrap*, Wiley.** The
  circular block bootstrap behind the Sharpe-difference test.
- **Wilson, E. B. (1927), "Probable Inference, the Law of Succession, and Statistical
  Inference", *Journal of the American Statistical Association* 22(158), 209-212.** The
  interval on crash shares (too narrow here because months overlap, as the results say).

## Data provenance

All tapes are read offline through [`quantlab/bundled.py`](../../../quantlab/bundled.py) and
checked against a SHA-256 pin before use (pins and fingerprints are in
[results.md §0](results.md)).

- **Fama-French monthly factors**, 1926-07 → 2018-11, as shipped in the `arch` package
  (`arch/data/frenchdata`). They originate in Kenneth R. French's Data Library. `Mkt-RF + RF` is the
  CRSP value-weighted market **total return**; `RF` is the one-month T-bill. Nominal.
- **S&P 500 index**, daily, 1990-01-02 → 2022-11-30 (the partial final month is dropped), as
  shipped in the `skfolio` 1.8.1 wheel (`sp500_index`). It is a **price index**: no dividends.
- **CBOE VIX close**, daily, 2014-01-03 → 2018-12-31, as shipped in `arch` (`arch/data/vix`).
  Five years and one episode: illustration only.
- **Software:** the `arch` package (K. Sheppard) for the GARCH and FIGARCH fits and FIGARCH
  simulation. The GARCH paths are simulated by our own vectorised recursion from the fitted
  parameters.

## Neighbours on this desk

**[03-fear-gauge](../../03-fear-gauge/)** (buying high-VIX spikes),
**[16-storm-shy](../../16-storm-shy/)** and **[591-vol-managed-portfolio](../../591-vol-managed-portfolio/)**
(scaling by current vol), **[898-managed-vol-equity](../../898-managed-vol-equity/)**,
**[111-vix-term-structure](../../111-vix-term-structure/)**,
**[578-cross-asset-correlation-regime](../../578-cross-asset-correlation-regime/)**,
**[992-vol-clustering-halflife](../../992-vol-clustering-halflife/)** (how fast vol mean-reverts,
the confound here), **[966-har-vs-garch](../../966-har-vs-garch/)**.
