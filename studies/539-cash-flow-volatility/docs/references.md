# References & literature map -- Study 539 (Cash-Flow-Volatility)

## The primary claim under test

- **Huang, A. G. (2009).** "The cross section of cashflow volatility and expected stock
  returns." *Journal of Empirical Finance*, 16(3), 409--429. The founding paper for this
  study: firms with higher *operating cash-flow volatility* earn **lower** average future
  returns, even after controlling for size, book-to-market, momentum, idiosyncratic return
  volatility, and accruals. Huang frames cash-flow uncertainty as a priced quality-like
  characteristic -- stable, predictable cash generators are rewarded; cash-flow lottery
  tickets are not. We test the long-low-CF-vol / short-high-CF-vol portfolio directly.

## Why the effect should exist -- the theoretical backbone

- **Minton, B. A. & Schrand, C. (1999).** "The impact of cash flow volatility on
  discretionary investment and the costs of debt and equity financing." *Journal of
  Financial Economics*, 54(3), 423--460. Higher cash-flow volatility raises the cost of
  external finance and forces firms to forgo positive-NPV investment -- a fundamental
  channel for lower valuations and (potentially) lower realised returns.
- **Dichev, I. D. & Tang, V. W. (2009).** "Earnings volatility and earnings
  predictability." *Journal of Accounting and Economics*, 47(1-2), 160--181. Earnings
  (and cash-flow) volatility strongly degrades the predictability of future earnings;
  uncertainty-averse investors discount such firms.
- **Gow, I. D. & Taylor, D. J. (2009).** "Earnings volatility and the cross-section of
  returns." Working paper. Documents related cross-sectional pricing of earnings/cash-flow
  smoothness, a sibling of the Huang result.

## Quality / low-risk relatives (the same family)

- **Asness, C. S., Frazzini, A., & Pedersen, L. H. (2019).** "Quality Minus Junk." *Review
  of Accounting Studies*, 24(1), 34--112. "Quality" -- profitable, growing, *safe* (low
  earnings/cash-flow volatility), well-managed firms -- commands a premium. Cash-flow-vol is
  one of the "safety" inputs. See desk [Study 242 -- Quality-Minus-Junk](../../242-quality-minus-junk/).
- **Novy-Marx, R. (2013).** "The other side of value: The gross profitability premium."
  *Journal of Financial Economics*, 108(1), 1--28. The profitability cousin; see desk
  [Study 122 -- Gross-Profitability](../../122-gross-profitability/).
- **Sloan, R. G. (1996).** "Do stock prices fully reflect information in accruals and cash
  flows about future earnings?" *The Accounting Review*, 71(3), 289--315. The accruals
  anomaly -- cash-flow quality vs accruals quality; see desk
  [Study 231 -- Sloan-Accruals](../../231-sloan-accruals/).

## Subsequent evidence, replication, and attenuation

- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5--32. Documents ~32% out-of-sample and ~58%
  post-publication decay across 97 anomalies. A 2009 cash-flow-volatility effect is a prime
  candidate for post-publication attenuation -- consistent with the null we observe.
- **Hou, K., Xue, C., & Zhang, L. (2020).** "Replicating Anomalies." *Review of Financial
  Studies*, 33(5), 2019--2133. A large fraction of published cross-sectional anomalies fail
  to replicate at conventional significance under careful, microcap-aware construction.
- **Harvey, C. R., Liu, Y., & Zhu, H. (2016).** "...and the Cross-Section of Expected
  Returns." *Review of Financial Studies*, 29(1), 5--68. Argues the multiple-testing
  threshold for a credible new factor is roughly *t* > 3, far above the |t| ~ 1 we find.

## Survivorship bias and data limitations

- **Shumway, T. (1997).** "The Delisting Bias in CRSP Data." *Journal of Finance*, 52(1),
  327--340. Delistings correlate with poor performance; removing failed firms biases factor
  returns. High-CF-vol firms are precisely the ones more likely to delist -- our survivor
  basket therefore *understates* the high-CF-vol short leg's true badness, biasing the
  long-short upward, and it is still negative.
- **Yahoo/yfinance fundamentals.** The free yfinance feed serves only the most recent
  ~5 quarters of cash-flow and balance-sheet statements, so the CF-volatility characteristic
  here is a thin, recent snapshot rather than a long rolling history. A faithful Huang (2009)
  replication needs Compustat-grade quarterly data back to the 1980s.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*,
  55(3), 703--708. The HAC long-run variance estimator behind the one-sample t-stat in
  [`strategy.hac_tstat`](../cash_flow_volatility/strategy.py).

## Related desk studies

- **[Study 122 -- Gross-Profitability](../../122-gross-profitability/)**: the profitability
  quality factor on the same infrastructure.
- **[Study 124 -- Cash-Flow-Yield](../../124-cash-flow-yield/)**: cash-flow *level* (yield)
  rather than cash-flow *dispersion*.
- **[Study 231 -- Sloan-Accruals](../../231-sloan-accruals/)**: cash-flow vs accrual quality.
- **[Study 242 -- Quality-Minus-Junk](../../242-quality-minus-junk/)**: the umbrella quality
  factor of which cash-flow-vol "safety" is one component.
- **[Study 330 -- Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)**: the
  return-volatility cousin of cash-flow-volatility.
