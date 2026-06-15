# References — Study 208 (Gold-Miners)

## Primary data

- **GLD (SPDR Gold Shares ETF)** — Yahoo Finance daily adjusted close, ticker `GLD`,
  inception 2004-11-18. The first ETF to give retail investors direct exposure to gold
  bullion; total-return adjusted (dividends minimal for commodity ETF).
- **GDX (VanEck Gold Miners ETF)** — Yahoo Finance daily adjusted close, ticker `GDX`,
  inception 2006-05-22. Market-cap-weighted index of global gold and silver mining
  companies (Newmont, Barrick, Agnico Eagle etc.).

## Literature on gold miners as "leveraged gold"

1. **Baur, D.G. & McDermott, T.K. (2010).** "Is gold a safe haven? International evidence."
   *Journal of Banking & Finance* 34(8), 1886-1898. Establishes gold's portfolio role and
   notes that miners are *not* equivalent to physical gold exposure.

2. **Blose, L.E. & Shieh, J.C.P. (1995).** "The impact of gold price on the value of gold
   mining stocks." *Review of Financial Economics* 4(2), 125-139. Early empirical work on
   the gold-price beta of miners; estimates betas substantially above 1.0 but with large
   idiosyncratic noise.

3. **Tufano, P. (1998).** "The determinants of stock price exposure: financial engineering
   and the gold mining industry." *Journal of Finance* 53(3), 1015-1052. Seminal work on
   hedging by gold miners and how production hedges modify the gold-beta. Documents that
   unhedged miners have betas ~2× and hedged miners closer to 1×.

4. **Conover, C.M., Jensen, G.R., Johnson, R.R. & Mercer, J.M. (2009).** "Can precious
   metals make your portfolio shine?" *Journal of Investing* 18(1), 75-86. Compares
   gold, silver and miner allocations; finds miners add idiosyncratic risk without
   proportional return compensation.

## On asymmetric beta / call-option structure of miners

5. **McDonald, R. & Siegel, D. (1986).** "The value of waiting to invest." *Quarterly
   Journal of Economics* 101(4), 707-728. Real options theory underpinning why miners
   have convex payoffs to commodity prices: the mine is a call option on gold prices
   above operating cost, implying higher sensitivity to gold upside than downside.

6. **Brennan, M.J. & Schwartz, E.S. (1985).** "Evaluating natural resource investments."
   *Journal of Business* 58(2), 135-157. Foundational paper on natural resource project
   valuation as real options; explains asymmetric price sensitivity.

## On operational drag and gold equity underperformance

7. **World Gold Council.** "Gold mining share performance vs gold price." Various annual
   reports 2015-2024. Consistently documents that GDX and similar indices have
   substantially underperformed the gold bullion price over 10-year periods, largely
   attributable to rising all-in sustaining costs (AISC) and capital allocation.

8. **Erb, C.B. & Harvey, C.R. (2013).** "The golden dilemma." *Financial Analysts Journal*
   69(4), 10-42. Comprehensive study of gold as an investment; distinguishes between
   physical gold and mining equity returns and their different exposures.

## Methodology and inference

9. **Newey, W.K. & West, K.D. (1987).** "A simple, positive semi-definite, heteroskedasticity
   and autocorrelation consistent covariance matrix." *Econometrica* 55(3), 703-708.
   The HAC estimator used for all t-statistics in this study.

10. **White, H. (1980).** "A heteroskedasticity-consistent covariance matrix estimator and
    a direct test for heteroskedasticity." *Econometrica* 48(4), 817-838. Foundation for
    the sandwich (HC) variance estimator extended to HAC in Newey-West.
