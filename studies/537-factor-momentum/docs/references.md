# References & literature map — Study 537 (Factor-Momentum)

## The primary claim under test

- **Ehsani, S. & Linnainmaa, J. T. (2022).** "Factor Momentum and the Momentum Factor."
  *Journal of Finance*, 77(3), 1877–1919. The founding paper. Individual factors exhibit
  **time-series momentum**: a factor with high recent returns continues to outperform. The
  authors show factor momentum is pervasive (15 of 20 factors), spans **most** of individual
  stock momentum (stock momentum is largely a manifestation of underlying factors trending),
  and that timing factors on their own past returns earns a robust premium. This is the exact
  meta-premium we test.

- **Arnott, R., Clements, M., Kalesnik, V. & Linnainmaa, J. (2021).** "Factor Momentum."
  *Review of Financial Studies* / SSRN working paper. A complementary treatment (Arnott-Ehsani
  lineage) documenting that factor momentum is stronger and more robust than the underlying
  factors traded statically, and that it survives in both time-series and cross-sectional form
  across an international sample. Motivates the "timing beats static" comparison in this study.

## The momentum backbone the claim builds on

- **Jegadeesh, N. & Titman, S. (1993).** "Returns to Buying Winners and Selling Losers."
  *Journal of Finance*, 48(1), 65–91. The original cross-sectional stock-momentum result that
  factor momentum is argued to subsume.

- **Moskowitz, T., Ooi, Y. H. & Pedersen, L. H. (2012).** "Time Series Momentum." *Journal of
  Financial Economics*, 104(2), 228–250. The time-series ("trend") momentum that we apply *to
  the factors* — hold long when the trailing return is positive, short when negative.

- **Asness, C., Moskowitz, T. & Pedersen, L. H. (2013).** "Value and Momentum Everywhere."
  *Journal of Finance*, 68(3), 929–985. Momentum (and value) appear across asset classes and
  styles — the spirit of treating factors as tradable assets with their own momentum.

## The factors we build (price-only versions)

- **Frazzini, A. & Pedersen, L. H. (2014).** "Betting Against Beta." *Journal of Financial
  Economics*, 111(1), 1–31. The low-beta factor (`lowbeta`); see also our **[Study 238 —
  Betting-Against-Beta](../../238-betting-against-beta/)**.

- **Ang, A., Hodrick, R., Xing, Y. & Zhang, X. (2006).** "The Cross-Section of Volatility and
  Expected Returns." *Journal of Finance*, 61(1), 259–299. The low-volatility anomaly
  (`lowvol`); see **[Study 330 — Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)**.

- **Jegadeesh, N. (1990).** "Evidence of Predictable Behavior of Security Returns." *Journal of
  Finance*, 45(3), 881–898. Short-term (1-month) reversal (`strev`).

- **Banz, R. W. (1981).** "The Relationship Between Return and Market Value of Common Stocks."
  *Journal of Financial Economics*, 9(1), 3–18. The size effect — here a *price-only proxy*, a
  named limitation (no point-in-time market cap from yfinance).

## Why a small-survivor replication usually disappoints

- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5–32. ~32% post-publication attenuation; factor
  momentum (published 2021–22) is recent but its component factors are well-mined.

- **Hou, K., Xue, C. & Zhang, L. (2020).** "Replicating Anomalies." *Review of Financial
  Studies*, 33(5), 2019–2133. Many factor premia shrink dramatically on careful replication;
  the meta-strategy inherits the noise of its inputs.

- **Shumway, T. (1997).** "The Delisting Bias in CRSP Data." *Journal of Finance*, 52(1),
  327–340. Survivorship inflates factor long legs — our fixed survivor basket cannot include
  the firms that populate the factors' short legs.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703–708. The HAC
  long-run variance used in [`strategy.hac_t`](../factor_momentum/strategy.py).

## Related desk studies

- **[Study 20 — Freight-Train](../../20-freight-train/)**: time-series momentum across many
  markets — the same trend engine, applied to assets instead of factors.
- **[Study 24 — Stampede](../../24-stampede/)**: cross-sectional stock momentum — the effect
  Ehsani-Linnainmaa argue factor momentum *subsumes*.
- **[Study 238 — Betting-Against-Beta](../../238-betting-against-beta/)**: the `lowbeta` factor
  traded statically — same desk infrastructure (rolling sort, long-short, HAC inference).
- **[Study 330 — Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)**: the `lowvol`
  factor traded statically.
- **[Study 38 — Chorus](../../38-chorus/)**: blending weak decorrelated signals — the
  diversification logic behind averaging timed factors.
