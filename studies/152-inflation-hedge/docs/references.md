# References — Study 152 (Inflation-Hedge)

## Primary Papers

**Fama, E. F. & Schwert, G. W. (1977).** "Asset returns and inflation."
*Journal of Financial Economics*, 5(2), 115-146.
The foundational paper: using 1953-1971 US data, Fama and Schwert show that
nominal stock returns are *negatively* correlated with both expected and unexpected
inflation — the exact opposite of a hedge. Treasury bills, not stocks, hedge
short-term inflation. This paper motivates our Fisher beta test.

**Modigliani, F. & Cohn, R. A. (1979).** "Inflation, rational valuation and the
market." *Financial Analysts Journal*, 35(2), 24-44.
The "inflation illusion" explanation: investors discount real cash flows using
nominal interest rates during high inflation, causing equities to be systematically
undervalued. Explains *why* stocks underperform in high-inflation environments —
investors confuse nominal and real quantities.

**Fisher, I. (1930).** *The Theory of Interest.* Macmillan, New York.
The original Fisher hypothesis: in efficient markets, nominal interest rates should
fully reflect expected inflation. Our "Fisher hypothesis for stocks" tests whether
the same logic applies to equity returns (it doesn't, in the short run).

## Key Extensions and Replications

**Bodie, Z. (1976).** "Common stocks as a hedge against inflation."
*Journal of Finance*, 31(2), 459-470.
Extends the Fama-Schwert result, confirms stocks fail as a short-run inflation
hedge, and attributes part of the failure to a negative real output shock that
typically accompanies high inflation (the "proxy hypothesis").

**Geske, R. & Roll, R. (1983).** "The fiscal and monetary linkage between stock
returns and inflation." *Journal of Finance*, 38(1), 1-33.
The "reverse causation" (proxy) hypothesis: a negative stock market signal triggers
a monetary policy response that causes inflation. The negative correlation between
stocks and inflation is therefore partly causal in the wrong direction.

**Boudoukh, J. & Richardson, M. (1993).** "Stock returns and inflation: A long-horizon
perspective." *American Economic Review*, 83(5), 1346-1355.
Crucial long-horizon result: at 5-year horizons, the negative correlation weakens
and nominal returns do partially compensate for inflation. Stocks become better
(though still imperfect) hedges at longer investment horizons — consistent with
our 5-year regime results (-1.9pp, t=-2.1, less severe than the 1-year gap).

**Ritter, J. R. & Warr, R. S. (2002).** "The decline of inflation and the bull market
of 1982-1999." *Journal of Financial and Quantitative Analysis*, 37(1), 29-61.
Directly tests the Modigliani-Cohn inflation illusion hypothesis over the
disinflation period: falling inflation caused systematic equity undervaluation
correction, explaining a large portion of the 1980s-1990s bull market.

## Data Sources

**Shiller, R. J.** *Irrational Exuberance* (2000/2015), updated monthly data.
Staged parquet at `_cache/shiller_sp500.parquet`; columns include SP500 price,
Dividends, Earnings, CPI, Long Interest Rate, Real Price, Real Dividend, Real
Earnings, PE10 (CAPE). 1871-present. Window used: 1872-01 to 2023-06 (n=1,818
months after dropping zero-placeholder rows at the end of the file).

## Methodology

**Newey, W. K. & West, K. D. (1987).** "A simple, positive semi-definite,
heteroskedasticity and autocorrelation consistent covariance matrix."
*Econometrica*, 55(3), 703-708.
HAC estimator used for all t-statistics in this study.  We use 12 lags for
overlapping 12-month return windows (Hodrick 1992 conservative approach).

**Hodrick, R. J. (1992).** "Dividend yields and expected stock returns: Alternative
procedures for inference and measurement." *Review of Financial Studies*, 5(3),
357-386.
Guidance on inference with overlapping long-horizon return regressions — motivates
using 12 Newey-West lags for annual-window regressions.
