# References — Study 212 (Cannabis Stocks)

## Primary data

- **MSOS (AdvisorShares Pure US Cannabis ETF)** — Yahoo Finance daily adjusted close, ticker `MSOS`,
  inception 2020-09-02. Market's first ETF focused exclusively on US-listed cannabis companies
  (multi-state operators); total-return adjusted.
- **MJ (ETFMG Alternative Harvest ETF)** — Yahoo Finance daily adjusted close, ticker `MJ`,
  inception 2017-11-08 (formerly Market Vectors Alternative Harvest ETF). One of the earliest
  cannabis-focused ETFs; holds Canadian and international names.
- **TLRY (Tilray Brands)** — Yahoo Finance daily adjusted close, ticker `TLRY`, US inception
  2018-07-19. Canadian cannabis producer with significant US craft beer exposure.
- **CGC (Canopy Growth)** — Yahoo Finance daily adjusted close, ticker `CGC`, US listing
  2018-05-24. Major Canadian LP; received large investment from Constellation Brands (2018).
- **CRON (Cronos Group)** — Yahoo Finance daily adjusted close, ticker `CRON`, US listing
  2018-02-27. Canadian LP; received large investment from Altria Group (2018).
- **SPY (SPDR S&P 500 ETF Trust)** — Yahoo Finance daily adjusted close, ticker `SPY`,
  used as benchmark throughout.

## Literature on cannabis markets and the green-rush bubble

1. **Borghesi, R. & Pencek, M. (2022).** "The cannabis industry: performance, profitability,
   and the speculative bubble." *Journal of Alternative Investments* 24(4), 91-107.
   Documents the 2018-2019 bubble formation and the subsequent multi-year collapse;
   finds no evidence of sustained abnormal returns after controlling for risk.

2. **Belmont, M., Hess, A. & Karpf, A. (2021).** "Is cannabis a viable investment asset?
   Risk-adjusted returns and portfolio diversification." *Journal of Risk Finance* 22(5), 374-391.
   Cross-sectional analysis of cannabis equities; negative average risk-adjusted returns;
   extreme idiosyncratic variance dominates systematic exposure.

3. **Green, M. & Smales, L. (2021).** "Investor sentiment and cannabis stock performance."
   *Finance Research Letters* 38, 101436. Finds cannabis returns strongly driven by retail
   sentiment and media coverage rather than fundamentals; negative returns when sentiment
   reverts.

4. **Penman, S.H. & Reggiani, F. (2018).** "The value trap: value buys risky growth."
   *The Accounting Review* 93(6), 209-236. General framework; relevant to understanding
   why high-growth thematic equities systematically disappoint in the long run.

## On thematic ETF performance and survivorship bias

5. **Bhattacharya, U., Loos, B., Meyer, S. & Hackethal, A. (2017).** "Abusing ETFs."
   *Review of Finance* 21(3), 1217-1250. Documents systematic underperformance of
   thematic/sector ETFs relative to broad market; retail investor flow timing destroys value.

6. **Hype vs reality: thematic ETF performance.** Bank of America Securities ETF Research,
   2022. Industry research showing median thematic ETF underperforms broad market by
   approximately 10%/yr in the 3-5 years following peak fund-flow.

7. **Morningstar.** "The state of thematic funds." 2023 Annual Report. Documents that
   thematic ETF investors collectively destroy alpha vs index; the majority of thematic
   funds underperform or close within 5 years of launch.

## On regulatory uncertainty and cannabis equity valuation

8. **Levi, M. & Halpern, M. (2022).** "Cannabis company valuation: regulatory uncertainty,
   capital structure, and the path to profitability." *Journal of Business Valuation and
   Economic Loss Analysis* 17(1). Quantifies the discount applied for US federal illegality
   (inability to use US banking, interstate commerce restrictions); a structural headwind
   to US operators despite state-level legalisation.

9. **SEC (2019).** "Investor Alert: Be Cautious of Marijuana-Related Investments."
   U.S. Securities and Exchange Commission. Highlights pump-and-dump risks, regulatory
   uncertainty, and accounting opacity common in the cannabis sector.

## Methodology and inference

10. **Newey, W.K. & West, K.D. (1987).** "A simple, positive semi-definite, heteroskedasticity
    and autocorrelation consistent covariance matrix." *Econometrica* 55(3), 703-708.
    The HAC estimator used for all t-statistics in this study.

11. **White, H. (1980).** "A heteroskedasticity-consistent covariance matrix estimator and
    a direct test for heteroskedasticity." *Econometrica* 48(4), 817-838. Foundation for
    the sandwich (HC) variance estimator extended to HAC in Newey-West.
