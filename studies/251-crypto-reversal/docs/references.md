# References — Study 251 (Crypto-Reversal)

## The paper under the microscope

**Babayev, F., & Aliyev, N. (2026).** "Crypto Has Fundamentals: A Seven-Factor Model
for Digital Asset Returns." SSRN 6818558 (QuantNest LLC / UCLA Anderson), May 2026.
The "QuantNest-7" (Q-7) model: market, size, short-term reversal, on-chain value,
residual volatility, on-chain quality, and perpetual-funding sentiment, on 90 tokens
(Jan 2020–May 2026). Reports REV (negated 21-day return) with a Newey-West *t* of 6.19
and a +151%/yr long-short quintile spread, and VOL inverted vs the equity low-vol anomaly.
The paper's own limitations section concedes that **survivorship bias inflates all factor
returns** and that the quintile spreads are **not achievable** after crypto bid-ask spreads
(which it notes can exceed 1% per trade), market impact and capacity. Its two novel factors
(value, quality) rely on paid Coin Metrics Pro data and undisclosed "proprietary" composites
and are not reproducible; this study tests the price-only slice (reversal, momentum) that is.

## The model it extends — and contradicts

**Liu, Y., Tsyvinski, A., & Wu, X. (2022).** "Common Risk Factors in Cryptocurrency."
*Journal of Finance*, 77(2), 1133-1177.
The three-factor (market, size, **momentum**) baseline. Documents cross-sectional
*momentum* (continuation), the opposite of the Q-7 paper's reversal finding — the central
tension this teardown adjudicates on a public panel.

**Liu, Y., & Tsyvinski, A. (2021).** "Risks and Returns of Cryptocurrency."
*Review of Financial Studies*, 34(6), 2689-2727.
Establishes time-series momentum in aggregate crypto returns.

## Short-term reversal in equities and crypto

**Jegadeesh, N. (1990).** "Evidence of Predictable Behavior of Security Returns."
*Journal of Finance*, 45(3), 881-898.
The original short-horizon (1-month) reversal in equities — the mechanism the paper imports
into crypto. Famously concentrated in illiquid, small names and largely arbitraged away by
microstructure and costs.

**Jegadeesh, N., & Titman, J. (1993).** "Returns to Buying Winners and Selling Losers."
*Journal of Finance*, 48(1), 65-91. The momentum benchmark; the paper cites it for the skip.

**Grobys, K., & Sapkota, N. (2019).** "Cryptocurrencies and momentum."
*Economics Letters*, 180, 6-10. Mixed evidence on crypto momentum; horizon- and
sample-dependent — consistent with this study's finding that the sign is fragile.

**Avramov, D., Cheng, S., & Metzker, L. (2022).** "Crypto market reversals and the
limits to arbitrage." Working paper. Documents reversal concentrated in hard-to-arbitrage,
high-cost tokens — directly relevant to the tradability verdict here.

## Why a daily-rebalanced long-short spread is not a strategy

**Novy-Marx, R., & Velikov, M. (2016).** "A Taxonomy of Anomalies and Their Trading Costs."
*Review of Financial Studies*, 29(1), 104-147.
Short-horizon, high-turnover anomalies (reversal is the canonical example) are precisely
those whose gross spreads are devoured by transaction costs — the core of the Mirage verdict.

**Makarov, I., & Schoar, A. (2020).** "Trading and Arbitrage in Cryptocurrency Markets."
*Journal of Financial Economics*, 135(2), 293-319. Documents wide, persistent spreads and
limited shortability in crypto — why a long-short book that shorts the small-cap winners is
far harder than the gross number suggests.

## Survivorship and inference methodology

**Newey, W.K., & West, K.D. (1987).** "A Simple, Positive Semi-Definite,
Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
*Econometrica*, 55(3), 703-708. The HAC estimator behind every *t* here.

**Fama, E.F., & MacBeth, J.D. (1973).** "Risk, Return, and Equilibrium: Empirical Tests."
*Journal of Political Economy*, 81(3), 607-636. The cross-sectional regression apparatus
this study mirrors to reproduce the paper's REV *t*.

**Harvey, C.R., Liu, Y., & Zhu, H. (2016).** "... and the Cross-Section of Expected Returns."
*Review of Financial Studies*, 29(1), 5-68. Motivates the |t| ≥ 2 inference bar and the
multiple-testing skepticism a seven-factor model warrants.

**Brown, S.J., Goetzmann, W., Ibbotson, R.G., & Ross, S.A. (1992).** "Survivorship Bias
in Performance Studies." *Review of Financial Studies*, 5(4), 553-580.
The canonical statement of how a survivor-only universe inflates measured returns — the
caveat the paper concedes and this study cannot escape on a yfinance panel.
