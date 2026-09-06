# Sources & literature map — Study 999 (The Break)

## Sequential detection

- **Page, E. S. (1954), "Continuous Inspection Schemes", *Biometrika* 41(1/2), 100-115.** The
  CUSUM, invented for factory quality control and still the reference method.
- **Lorden, G. (1971), "Procedures for Reacting to a Change in Distribution", *Annals of
  Mathematical Statistics* 42(6), 1897-1908.** Proves CUSUM is asymptotically optimal in
  worst-case expected delay for a given false-alarm rate — the theoretical basis for section 2's
  claim that the delay cannot be improved on.
- **Moustakides, G. V. (1986), "Optimal Stopping Times for Detecting Changes in Distributions",
  *Annals of Statistics* 14(4), 1379-1387.** The exact optimality result.
- **Siegmund, D. (1985), *Sequential Analysis: Tests and Confidence Intervals*, Springer.**
  Wald's identity and the average-run-length calculations behind `theoretical_delay`.

## Retrospective segmentation

- **Scott, A. J. & Knott, M. (1974), "A Cluster Analysis Method for Grouping Means in the
  Analysis of Variance", *Biometrics* 30(3), 507-512.** Binary segmentation.
- **Killick, R., Fearnhead, P. & Eckley, I. A. (2012), "Optimal Detection of Changepoints with a
  Linear Computational Cost", *JASA* 107(500), 1590-1598.** PELT — exact and fast, and the
  natural upgrade to this study's greedy recursion.
- **Bai, J. & Perron, P. (1998), "Estimating and Testing Linear Models with Multiple Structural
  Changes", *Econometrica* 66(1), 47-78.** The econometrician's standard for multiple breaks.

## Regime change in markets

- **Hamilton, J. D. (1989), "A New Approach to the Economic Analysis of Nonstationary Time
  Series and the Business Cycle", *Econometrica* 57(2), 357-384.** Markov switching — the
  alternative framing, in which regimes are latent states rather than breaks.
- **Adams, R. P. & MacKay, D. J. C. (2007), "Bayesian Online Changepoint Detection",
  arXiv:0710.3742.** The Bayesian sequential method, which handles an unknown change size
  better than CUSUM.
- **Andreou, E. & Ghysels, E. (2002), "Detecting Multiple Breaks in Financial Market Volatility
  Dynamics", *Journal of Applied Econometrics* 17(5), 579-600.** Change-point detection applied
  to volatility specifically, which is the version that works.
- **Inclan, C. & Tiao, G. C. (1994), "Use of Cumulative Sums of Squares for Retrospective
  Detection of Changes of Variance", *JASA* 89(427), 913-923.** The direct ancestor of
  `variance_cusum`.

## Neighbours on this desk

**625-macro-regime-switching**, **992-vol-clustering-halflife**, **990-var-breach-count**,
**985-last-hike-timing**.
