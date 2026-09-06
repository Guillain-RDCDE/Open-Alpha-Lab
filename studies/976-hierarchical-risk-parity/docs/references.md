# Sources & literature map — Study 976 (The Family Tree)

## The method

- **López de Prado, M. (2016), "Building Diversified Portfolios that Outperform Out of Sample",
  *Journal of Portfolio Management* 42(4), 59-69.** Hierarchical risk parity: the algorithm,
  the argument against matrix inversion, and the original Monte Carlo.
- **López de Prado, M. (2018), *Advances in Financial Machine Learning*, ch. 16.** The
  reference implementation and the clearest statement of why the tree is supposed to help.
- **Mantegna, R. N. (1999), "Hierarchical Structure in Financial Markets", *European Physical
  Journal B* 11(1), 193-197.** The correlation distance `sqrt(0.5(1-rho))` and the minimum
  spanning tree of a market — where the hierarchy idea comes from.
- **Raffinot, T. (2017), "Hierarchical Clustering-Based Asset Allocation", *Journal of Portfolio
  Management* 44(2), 89-99.** Alternative linkages and clustering criteria; finds the choice
  matters less than the fact of clustering.

## The competitors

- **Maillard, S., Roncalli, T. & Teïletche, J. (2010), "The Properties of Equally Weighted Risk
  Contribution Portfolios", *Journal of Portfolio Management* 36(4), 60-70.** Equal risk
  contribution, the industry standard tested here.
- **DeMiguel, V., Garlappi, L. & Uppal, R. (2009), "Optimal Versus Naive Diversification",
  *Review of Financial Studies* 22(5), 1915-1953.** 1/N as the benchmark nothing reliably beats.
- **Jagannathan, R. & Ma, T. (2003), "Risk Reduction in Large Portfolios: Why Imposing the Wrong
  Constraints Helps", *Journal of Finance* 58(4), 1651-1683.** Long-only is shrinkage — the
  reason HRP's no-shorts property is doing more work than the clustering.
- **Ledoit, O. & Wolf, M. (2003, 2004).** The other answer to the same problem, tested on this
  desk as study **975**.

## Sceptical readings

- **Jain, P. & Jain, S. (2019), "Can Machine Learning-Based Portfolios Outperform Traditional
  Risk-Based Portfolios? The Need to Account for Covariance Misspecification", *Risks* 7(3),
  74.** Finds HRP's advantage largely disappears once the comparison is fair.
- **Lohre, H., Rother, C. & Schäfer, K. A. (2020), "Hierarchical Risk Parity: Accounting for
  Tail Dependencies in Multi-Asset Multi-Factor Allocations", in *Machine Learning for Asset
  Management*.** A careful multi-asset evaluation; the honest version of the claim.

## Neighbours on this desk

**975-covariance-shrinkage**, **977-max-diversification**, **978-resampled-frontier**,
**171-naive-1-over-n**, **890-sector-risk-parity**, **896-risk-parity-trend**,
**974-diversification-saturation**.
