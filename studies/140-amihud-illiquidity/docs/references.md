# References & literature map — Study 140 (Amihud-Illiquidity)

## The canonical paper

- **Amihud, Y. (2002).** *Illiquidity and Stock Returns: Cross-Section and Time-Series Effects.*
  Journal of Financial Markets, 5(1), 31–56. The foundational paper: proposes ILLIQ =
  mean(|r_d|/DolVol_d) as a low-cost proxy for price impact/illiquidity. Documents (a) the
  cross-sectional premium — high-ILLIQ stocks earn higher expected returns — and (b) a
  time-series effect — aggregate market illiquidity predicts market returns. The cross-sectional
  premium was estimated on NYSE stocks 1964–1997 (CRSP, survivorship-free), and was strongest
  in small-cap deciles. This study tests the cross-sectional claim on a survivorship-biased
  large-cap S&P 500 panel, where the premium famously should be muted or absent.

## The liquidity-risk literature: why the premium should exist

- **Amihud, Y. & Mendelson, H. (1986).** *Asset Pricing and the Bid-Ask Spread.* Journal of
  Financial Economics, 17(2), 223–249. The theoretical foundation: investors demand a liquidity
  premium as compensation for bearing transaction costs (bid-ask spread), so illiquid assets
  must offer higher expected returns. The premium is proportional to the spread and investor
  holding period.

- **Pastor, L. & Stambaugh, R.F. (2003).** *Liquidity Risk and Expected Stock Returns.* Journal
  of Political Economy, 111(3), 642–685. A market-wide liquidity risk factor (sensitivity of
  individual returns to aggregate liquidity shocks) earns a positive risk premium of ~7.5%/yr
  in the cross-section. This is a *systematic* risk-factor framing of the liquidity premium,
  complementing Amihud's characteristic-based sort.

- **Acharya, V.V. & Pedersen, L.H. (2005).** *Asset Pricing with Liquidity Risk.* Journal of
  Financial Economics, 77(2), 375–410. A CAPM with liquidity costs shows that stocks with
  high illiquidity and high co-movement with market illiquidity earn higher expected returns —
  decomposing the premium into a level and three liquidity-beta channels.

## Why the premium is concentrated in micro-caps

- **Hasbrouck, J. (2009).** *Trading Costs and Returns for U.S. Equities: Estimating Effective
  Costs from Daily Data.* Journal of Finance, 64(3), 1445–1477. Shows that Amihud ILLIQ is a
  high-quality proxy for transaction costs in small stocks but loses predictive power in large,
  liquid names — because dollar volume is so large that |r|/DolVol is essentially zero for
  mega-caps. The premium is mechanically concentrated in micro-caps and is absent or negligible
  in large-cap stocks.

- **Goyenko, R.Y., Holden, C.W. & Trzcinka, C.A. (2009).** *Do Liquidity Measures Measure
  Liquidity?* Journal of Financial Economics, 92(2), 153–181. Compares ILLIQ and twelve other
  liquidity measures against TAQ-based effective spreads and price impact. ILLIQ is a better
  proxy for price impact in small stocks than in large stocks; for large-caps, more direct
  microstructure measures are needed.

## Survivorship bias in factor research

- **Kothari, S.P., Shanken, J. & Sloan, R.G. (1995).** *Another Look at the Cross-Section of
  Expected Stock Returns.* Journal of Finance, 50(1), 185–224. Documents how survivorship bias
  in Compustat/CRSP inflates book-to-market (and by extension any "quality" or "small" factor)
  effects by including only surviving firms. A recurring caution for any EDGAR or current-index
  panel study.

- **Brown, S.J., Goetzmann, W. & Ross, S.A. (1995).** *Survival.* Journal of Finance, 50(3),
  853–873. A classic treatment of how survivorship in financial databases creates upward-biased
  estimates of average returns — directly applicable to a panel built from current S&P 500
  members projected backwards.

## Post-publication decay and large-cap evidence

- **McLean, R.D. & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return
  Predictability?* Journal of Finance, 71(1), 5–32. Documents post-publication decay across
  97 anomalies — many of which (including liquidity-based factors) weaken significantly after
  the original paper draws arbitrage capital, and even more so in large liquid stocks where
  the cost of trading the anomaly is lowest.

- **Chordia, T., Subrahmanyam, A. & Tong, Q. (2014).** *Have Capital Market Anomalies
  Attenuated in the Light of Trading Costs?* Journal of Corporate Finance, 25, 1–12. Shows
  that accounting for realistic transaction costs eliminates the Amihud premium in liquid
  large-cap stocks; the premium survives only in micro-caps where arbitrage is costly.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../amihud_illiquidity/strategy.py) implements this inline.

- **Annual equal-weight quintile sort.** A standard Fama-French-style portfolio sort. Fama, E.F.
  & French, K.R. (1992), *The Cross-Section of Expected Stock Returns*, Journal of Finance,
  47(2), 427–465. The sorting and equal-weighting methodology mirrors the desk's other
  factor studies (Study 65 Scorecard, Study 121 Magic-Formula).

## Related desk studies

- **[Study 65 — Scorecard](../../65-scorecard/)**: Piotroski F-score quintile sort on EDGAR
  fundamentals — same annual-sort protocol, same survivorship-bias caveat.
- **[Study 121 — Magic-Formula](../../121-magic-formula/)**: Greenblatt's combined quality +
  cheapness rank on EDGAR — same large-cap panel, same survivorship-bias upper-bound conclusion.
- **[Study 52 — Smoke-Screen](../../52-smoke-screen/)**: another EDGAR-based factor study
  that names survivorship explicitly and treats results as upper bounds.
