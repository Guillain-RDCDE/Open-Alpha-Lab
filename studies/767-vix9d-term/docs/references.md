# References — Study 767 (VIX9D-Term)

## The claim's source

**CBOE Global Markets. (2011).** "CBOE Short-Term Volatility Index (VXST / VIX9D)."
CBOE white paper and methodology.
— Introduces the 9-day volatility index (originally ticker VXST, now disseminated as
^VIX9D), applying the VIX methodology to SPX options with ~9 days to expiry. The
^VIX9D/^VIX ratio is the canonical measure of the *short-end* slope of the volatility
term structure — the object this study tests as an equity-timing signal.

**Simon, D. (2021).** "VIX Futures Basis as a Signal for Equity Returns."
*Journal of Derivatives*, 28(3), 7–27.
— Documents a positive relationship between VIX futures contango/backwardation and
subsequent equity returns; the basis is treated as a fear-gauge signal. The most direct
academic statement of the claim this study rejects at the *short* end of the curve.

**Johnson, T. L. (2017).** "Risk Premia and the VIX Term Structure."
*Journal of Financial and Quantitative Analysis*, 52(6), 2461–2490.
— Decomposes the VIX term structure into variance-risk-premium components; finds the
slope conveys information about expected vs realised volatility, but the direction of
the equity-return prediction is ambiguous — precisely the volatility-vs-direction
distinction this study's regime split makes explicit.

## The underlying effect (variance risk premium and slope dynamics)

**Carr, P., & Wu, L. (2006).** "A Tale of Two Indices."
*Journal of Derivatives*, 13(3), 13–29.
— Introduces VIX vs VXV term-structure analysis; the slope reflects the shape of the
implied-volatility surface across maturities. The 9-day tenor extends this framework to
the extreme front of the curve.

**Bollerslev, T., Tauchen, G., & Zhou, H. (2009).** "Expected Stock Returns and
Variance Risk Premia."
*Review of Financial Studies*, 22(11), 4463–4492.
— Shows the variance risk premium (implied minus realised vol) predicts equity returns
at medium horizons; the VIX *level*, not the slope, drives most of the result — a
warning that a slope signal may be spurious even where a level signal is real.

**Dew-Becker, I., Giglio, S., Le, A., & Rodriguez, M. (2017).** "The Price of Variance
Risk."
*Journal of Financial Economics*, 123(2), 225–250.
— Identifies distinct short- and long-run variance-risk premia; term-structure slope
captures their ratio but explains little incremental return variation — consistent with
this study's finding that the short-end slope forecasts vol, not direction.

## Short-dated volatility and its microstructure

**Andersen, T. G., Fusari, N., & Todorov, V. (2017).** "Short-Term Market Risks Implied
by Weekly Options."
*Journal of Finance*, 72(3), 1335–1386.
— Shows that very-short-dated options price a distinct, fast-moving jump/tail component
— the economic reason the 9-day tenor is the twitchiest point on the surface and
inverts far more often than the 3-month slope.

## Timing strategies based on VIX signals

**Connors, L. A., & Alvarez, C. (2012).** *Short Term Trading Strategies that Work.*
TradingMarkets Publishing Group.
— A practitioner source frequently cited for VIX-*level* timing rules (not the slope);
serves as a reference for the genre the study tests and rejects.

**vixcentral.com / practitioner vol-trading community. (2014–2024).** Daily tracking of
the VIX term structure including the 9-day/30-day front-end slope.
— Popularised the contango/backwardation framing — and specifically the "front end
inverted = get defensive" heuristic — used by retail vol-traders; the direct source of
the claim's popular form.

## Related desk studies

- **[Study 111 — VIX-Term-Structure](../../111-vix-term-structure/):** the same test on
  the ^VIX/^VIX3M (30-day vs 90-day) slope; also lands `None`/`Mirage`. This study is
  its short-end (9-day vs 30-day) cousin — and finds the front end inverts ~27% of the
  time vs ~10% for the 3-month slope, so it switches far more and costs more.

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
2011–2026 window is consistent with either a spurious original finding or rapid
arbitrage-away.
