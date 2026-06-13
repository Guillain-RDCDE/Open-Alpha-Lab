# References — Study 111 (VIX-Term-Structure)

## The claim's source

**Simon, D. (2021).** "VIX Futures Basis as a Signal for Equity Returns."
*Journal of Derivatives*, 28(3), 7–27.
— Documents a positive relationship between VIX futures contango/backwardation
and subsequent equity returns; the basis is treated as a fear-gauge signal.

**Johnson, T. L. (2017).** "Risk Premia and the VIX Term Structure."
*Journal of Financial and Quantitative Analysis*, 52(6), 2461–2490.
— Decomposes the VIX term structure into variance risk premia components; finds
the slope conveys information about expected vs realised volatility, but the
direction of the equity return prediction is ambiguous.

## The underlying effect (variance risk premium and slope dynamics)

**Carr, P., & Wu, L. (2006).** "A Tale of Two Indices."
*Journal of Derivatives*, 13(3), 13–29.
— Introduces VIX vs VXV (predecessor of VIX3M) term structure analysis; the slope
reflects the shape of the implied volatility surface across maturities.

**Bollerslev, T., Tauchen, G., & Zhou, H. (2009).** "Expected Stock Returns and
Variance Risk Premia."
*Review of Financial Studies*, 22(11), 4463–4492.
— Shows the variance risk premium (implied minus realised vol) predicts equity
returns at medium horizons; the VIX level (not the slope) drives most of the result.

**Dew-Becker, I., Giglio, S., Le, A., & Rodriguez, M. (2017).** "The Price of
Variance Risk."
*Journal of Financial Economics*, 123(2), 225–250.
— Identifies distinct short- and long-run variance risk premia; term-structure slope
captures their ratio but explains little incremental return variation.

## VIX3M (the 3-month VIX index)

**CBOE. (2008).** "CBOE 3-Month Volatility Index (VIX3M) White Paper."
CBOE Global Markets.
— Describes the construction of VIX3M (launched January 2008), which extends the
VIX methodology to a 93-day horizon; the VIX/VIX3M ratio is the most natural
measure of the near-term vs medium-term implied volatility slope.

## Timing strategies based on VIX signals

**Connors, L. A., & Alvarez, C. (2012).** *Short Term Trading Strategies that Work.*
TradingMarkets Publishing Group.
— A practitioner source frequently cited for VIX-level timing rules (not the slope);
serves as a reference for the genre the study tests and rejects.

**Volatility At Every Turn. (2014–2022).** Various blog posts at vixcentral.com.
— A practitioner series tracking the VIX term structure daily; popularised the
contango/backwardation framing used by retail traders as the source of the claim.

## Related desk studies

- **Study 86 — Tail-Radar:** Tests the CBOE SKEW index (not the slope) as a crash
  predictor; also finds no statistically robust forecasting power.
- **Study 92 — Easy-Money:** Uses the VIX futures curve contango to short VXX/UVXY
  volatility ETPs — a different and more implementable angle on the same term-structure
  information (shorting premium decay, not timing equity direction).

## Method lineage

**Newey, W. K., & West, K. D. (1987).** "A Simple, Positive Semi-definite,
Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
*Econometrica*, 55(3), 703–708.
— The HAC variance estimator (Bartlett kernel) used throughout for inference on
overlapping forward returns and daily spread series.

**McLean, R. D., & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
Predictability?"
*Journal of Finance*, 71(1), 5–32.
— Documents post-publication decay; a claimed predictor that fails on the full
2008–2026 window (well after the 2006–2017 originating papers) is consistent
with either a spurious original finding or rapid arbitrage-away.
