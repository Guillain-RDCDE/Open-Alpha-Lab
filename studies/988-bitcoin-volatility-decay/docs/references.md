# Sources & literature map — Study 988 (The Taming)

## Why a trend in volatility is hard to measure

- **Ding, Z., Granger, C. W. J. & Engle, R. F. (1993), "A Long Memory Property of Stock Market
  Returns and a New Model", *Journal of Empirical Finance* 1(1), 83-106.** The long-memory
  property of realised volatility — the reason a naive standard error on a volatility trend is
  badly wrong.
- **Baillie, R. T. (1996), "Long Memory Processes and Fractional Integration in Econometrics",
  *Journal of Econometrics* 73(1), 5-59.** Why fractionally-integrated series produce spurious
  trends in finite samples, which is precisely this study's null.
- **Granger, C. W. J. & Newbold, P. (1974), "Spurious Regressions in Econometrics", *Journal of
  Econometrics* 2(2), 111-120.** The founding statement of the failure mode.
- **Politis, D. N. & Romano, J. P. (1994), "The Stationary Bootstrap", *JASA* 89(428),
  1303-1313.** The resampling scheme behind `block_bootstrap_trend`.
- **Mann, H. B. (1945), *Econometrica* 13(3), 245-259**, and **Kendall, M. G. (1975), *Rank
  Correlation Methods*.** The distribution-free trend test, and the independence assumption in
  its variance formula that this study flags rather than ignores.
- **Sen, P. K. (1968), "Estimates of the Regression Coefficient Based on Kendall's Tau", *JASA*
  63(324), 1379-1389.** The robust slope estimator.

## Crypto volatility specifically

- **Baur, D. G. & Dimpfl, T. (2018), "Asymmetric Volatility in Cryptocurrencies", *Economics
  Letters* 173, 148-151.** Bitcoin's volatility responds to positive shocks more than negative
  ones — the opposite of equities, and a reason not to import equity intuitions here.
- **Katsiampa, P. (2017), "Volatility Estimation for Bitcoin: A Comparison of GARCH Models",
  *Economics Letters* 158, 3-6.** What fits Bitcoin's conditional variance.
- **Liu, Y. & Tsyvinski, A. (2021), "Risks and Returns of Cryptocurrency", *Review of Financial
  Studies* 34(6), 2689-2727.** The broader risk-factor picture.

## Volatility targeting

- **Moreira, A. & Muir, T. (2017), "Volatility-Managed Portfolios", *Journal of Finance* 72(4),
  1611-1644.** The case for sizing by inverse volatility, and the conditions under which it
  helps.
- **Harvey, C. R. et al. (2018), "The Impact of Volatility Targeting", *Journal of Portfolio
  Management* 45(1), 14-33.** The practitioner assessment, including where it does not help.

## Neighbours on this desk

**142-bitcoin-correlation**, **604-crypto-equity-beta**, **983-bitcoin-leads-equities**,
**256-volatility-clustering**, **371-vix-term-structure**, **774-levered-etf-decay**.
