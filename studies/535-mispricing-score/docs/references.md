# References & literature map — Study 535 (Mispricing-Score)

## The primary claim under test

- **Stambaugh, R. F., Yu, J. & Yuan, Y. (2015).** "Arbitrage Asymmetry and the
  Idiosyncratic Volatility Puzzle." *Journal of Finance*, 70(5), 1903–1948. The paper that
  popularised the **composite mispricing measure**: average the cross-sectional rankings of
  eleven well-known anomalies into a single score, sort on it, and the long-short spread is
  larger and more reliable than any single anomaly. Crucially they show the predictive power
  is **concentrated in the SHORT (overpriced) leg** — arbitrage asymmetry (short-sale costs,
  leverage limits) keeps overpriced stocks overpriced longer than underpriced stocks stay cheap.
- **Stambaugh, R. F., Yu, J. & Yuan, Y. (2012).** "The Short of It: Investor Sentiment and
  Anomalies." *Journal of Financial Economics*, 104(2), 288–302. The companion paper. The
  same eleven anomalies are strongest following high-sentiment periods, and again the effect
  lives mostly in the short leg — consistent with sentiment-driven overpricing that is costly
  to arbitrage away.

## The eleven anomalies in the SYY composite

- **Sloan, R. (1996).** "Do Stock Prices Fully Reflect Information in Accruals and Cash
  Flows About Future Earnings?" *Accounting Review*, 71(3), 289–315. Accruals anomaly.
- **Ritter, J. (1991)** & **Loughran, T. & Ritter, J. (1995).** IPO/SEO net-issuance
  underperformance — over-issuing firms underperform.
- **Novy-Marx, R. (2013).** "The Other Side of Value: The Gross Profitability Premium."
  *Journal of Financial Economics*, 108(1), 1–28. Profitability.
- **Cooper, M., Gulen, H. & Schill, M. (2008).** "Asset Growth and the Cross-Section of
  Stock Returns." *Journal of Finance*, 63(4), 1609–1651. Investment / asset growth.
- **Jegadeesh, N. & Titman, J. (1993).** "Returns to Buying Winners and Selling Losers."
  *Journal of Finance*, 48(1), 65–91. Momentum — one of the price-only members we replicate.
- **Campbell, J., Hilscher, J. & Szilagyi, J. (2008).** "In Search of Distress Risk."
  *Journal of Finance*, 63(6), 2899–2939. Financial distress / O-score component.
- **Ang, A., Hodrick, R., Xing, Y. & Zhang, X. (2006).** "The Cross-Section of Volatility
  and Expected Returns." *Journal of Finance*, 61(1), 259–299. Idiosyncratic-volatility
  member — the high-vol leg of our composite.
- **Bali, T., Cakici, N. & Whitelaw, R. (2011).** "Maxing Out: Stocks as Lotteries and the
  Cross-Section of Expected Returns." *Journal of Financial Economics*, 99(2), 427–446. The
  MAX (lottery) member of our composite.

## Why the edge should sit in the short leg — arbitrage asymmetry

- **Miller, E. (1977).** "Risk, Uncertainty, and Divergence of Opinion." *Journal of
  Finance*, 32(4), 1151–1168. Short-sale constraints let optimists set prices; overpricing
  persists. The theoretical backbone of SYY's short-leg result.
- **Shleifer, A. & Vishny, R. (1997).** "The Limits of Arbitrage." *Journal of Finance*,
  52(1), 35–55. Arbitrageurs are capital-constrained and risk-averse, so mispricing — especially
  costly-to-short overpricing — is not fully corrected.

## Replication, attenuation, and the survivorship trap we hit

- **Hou, K., Xue, C. & Zhang, L. (2020).** "Replicating Anomalies." *Review of Financial
  Studies*, 33(5), 2019–2133. Many of the eleven anomalies are weak or insignificant once
  micro-caps are excluded and value-weighting is used — directly relevant to a large-cap test.
- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5–32. ~32% post-publication decay; the SYY
  composite (published 2012/2015) is squarely in the post-publication window.
- **Shumway, T. (1997).** "The Delisting Bias in CRSP Data." *Journal of Finance*, 52(1),
  327–340. Delistings correlate with poor performance. A survivor basket **removes the worst
  overpriced names** — exactly the SYY short leg — so on survivors the short-leg edge can
  vanish or invert. This is the mechanism behind our adverse real-tape result.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703–708. The HAC
  long-run variance estimator in [`strategy.summary`](../mispricing_score/strategy.py).

## Related desk studies

- **[Study 238 — Betting-Against-Beta](../../238-betting-against-beta/)**: the low-beta member
  of the mispricing family — same rolling-sort, equal-weight, HAC infrastructure.
- **[Study 330 — Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)**: the volatility
  member of our composite, in its purest ETF form.
- **[Study 231 — Sloan-Accruals](../../231-sloan-accruals/)** and
  **[Study 244 — Asset-Growth](../../244-asset-growth/)**: two of the eleven SYY anomalies as
  stand-alone studies — the composite is meant to beat each of them.
- **[Study 363 — PEAD-Drift](../../363-pead-drift/)**: a non-composite single anomaly, for
  contrast with the meta-anomaly approach.
