# References — Study 210 (Crypto-Trend)

## The original Faber rule

**Faber, M.T. (2007).** "A Quantitative Approach to Tactical Asset Allocation."
*Journal of Wealth Management*, 9(4), 69-79. Also SSRN 962461.
The canonical source for the 10-month SMA timing rule on equity and multi-asset portfolios.
Applied here to BTC-USD using the 200-day (≈10-month) daily variant.

## Trend-following in crypto

**Liu, Y., & Tsyvinski, A. (2021).** "Risks and Returns of Cryptocurrency."
*Review of Financial Studies*, 34(6), 2689-2727.
Documents that momentum (including moving average rules) is the primary driver of
cross-sectional crypto returns. Confirms the theoretical grounding for trend-following in crypto.

**Cong, L.W., Harvey, C.R., Rabetti, D., & Wu, Z. (2023).** "An Anatomy of Crypto-Enabled
Cybercrimes." *Journal of Finance*, 78(3), 1549-1602.
Context on BTC market structure and the sources of crypto price dynamics.

**Grobys, K., Ahmed, S., & Sapkota, N. (2020).** "Technical trading rules in the
cryptocurrency market." *Finance Research Letters*, 32, 101396.
Tests multiple technical rules in crypto markets; MA-based rules among the most consistent.

## Moving average timing — academic literature

**Moskowitz, T., Ooi, Y.H., & Pedersen, L.H. (2012).** "Time series momentum."
*Journal of Financial Economics*, 104(2), 228-250.
The foundational AQR paper on time-series momentum (trend-following), which the MA timing
rule implements in a simplified binary form. Shows positive expected returns for the signal
across many asset classes including commodities (the closest analog to crypto's speculative nature).

**Han, Y., Zhou, G., & Zhu, Y. (2016).** "A Trend Factor: Any Economic Gains from Using
Information over Investment Horizons?" *Journal of Financial Economics*, 122(2), 352-375.
Extends the MA signal across multiple lookback windows; 200-day is among the most robust.

**Barberis, N. (2018).** "Psychology-based models of asset prices and trading volume."
*Handbook of Behavioral Economics*, 1, 79-175.
Provides the theoretical foundation for why trend signals can persist: underreaction
to information causes trends; overreaction eventually reverses them (the bull/bear cycle).

## Inference methodology

**Newey, W.K., & West, K.D. (1987).** "A Simple, Positive Semi-Definite,
Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
*Econometrica*, 55(3), 703-708.
The HAC estimator used for all t-statistics in this study (NW with automatic lag selection).

**Harvey, C.R., Liu, Y., & Zhu, H. (2016).** "... and the Cross-Section of Expected Returns."
*Review of Financial Studies*, 29(1), 5-68.
Motivates the |t| ≥ 2 inference bar for financial strategies; multiple-comparisons concern
is lower here as we test a single pre-specified rule, but the bar is maintained.

**McLean, R.D., & Pontiff, J. (2016).** "Does Academic Publication Destroy Stock Return
Predictability?" *Journal of Finance*, 71(1), 5-32.
Post-publication decay is a key risk for the Faber rule; we check sub-period stability
rather than assuming the full-sample result reflects a tradable forward-looking edge.

## Crypto market structure

**Makarov, I., & Schoar, A. (2020).** "Trading and Arbitrage in Cryptocurrency Markets."
*Journal of Financial Economics*, 135(2), 293-319.
Documents the persistent price inefficiencies and high volatility in crypto markets that
create the deep bear-market regimes the MA rule attempts to exploit.

**Burniske, C., & Tatar, J. (2017).** *Cryptoassets: The Innovative Investor's Guide.*
McGraw-Hill. Contextualises Bitcoin's 4-year halving cycles as the primary driver of the
bull/bear regime structure that the SMA rule navigates.
