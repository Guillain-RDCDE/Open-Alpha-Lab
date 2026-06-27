# References & literature map -- Study 506 (Industry-Momentum)

## The primary claim under test

- **Moskowitz, T. J. & Grinblatt, M. (1999).** "Do Industries Explain Momentum?" *Journal
  of Finance*, 54(4), 1249--1290. The founding paper. The authors decompose individual-stock
  momentum and find that an *industry* momentum strategy -- buying past-winner industries and
  shorting past-loser industries -- is highly profitable and *subsumes* much of individual-stock
  momentum. After controlling for industry momentum, the residual individual-stock momentum is
  weak and often insignificant. Their headline: momentum is largely an industry effect.

## The momentum effect it builds on

- **Jegadeesh, N. & Titman, S. (1993).** "Returns to Buying Winners and Selling Losers:
  Implications for Stock Market Efficiency." *Journal of Finance*, 48(1), 65--91. The original
  cross-sectional momentum result -- buy past 3--12 month winners, sell losers, hold 3--12
  months -- and the source of the 12-1 (skip-the-most-recent-month) formation convention we use.
- **Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013).** "Value and Momentum
  Everywhere." *Journal of Finance*, 68(3), 929--985. Momentum is pervasive across asset
  classes and is best understood alongside value as a global factor pair.

## Rebuttal, qualification, and the industry-vs-stock debate

- **Grundy, B. D. & Martin, J. S. (2001).** "Understanding the Nature of the Risks and the
  Source of the Rewards to Momentum Investing." *Review of Financial Studies*, 14(1), 29--78.
  Argue that momentum is *not* primarily an industry effect once one controls for time-varying
  factor exposures -- a direct counter to Moskowitz-Grinblatt. The debate is unresolved and
  sample-dependent, which is exactly why a clean ETF replication is worth running.
- **Lewellen, J. (2002).** "Momentum and Autocorrelation in Stock Returns." *Review of
  Financial Studies*, 15(2), 533--564. Shows that portfolio (including industry) momentum can
  arise from lead-lag cross-serial correlations, not just own-autocorrelation.

## Sector-rotation and ETF implementation

- **Faber, M. T. (2010).** "Relative Strength Strategies for Investing." Cambria working
  paper / SSRN 1585517. A practitioner's relative-strength sector-rotation rule on ETFs --
  the retail-tradable cousin of industry momentum.
- **Conrad, J. & Kaul, G. (1998).** "An Anatomy of Trading Strategies." *Review of Financial
  Studies*, 11(3), 489--519. Decomposes momentum/contrarian profits into cross-sectional vs
  time-series components -- the accounting that tells you whether an *industry* sort can work.

## Costs, decay, and survivorship

- **Novy-Marx, R. & Velikov, M. (2016).** "A Taxonomy of Anomalies and Their Trading Costs."
  *Review of Financial Studies*, 29(1), 104--147. Momentum is among the *highest*-turnover
  anomalies; trading costs materially erode net returns. Sector-ETF momentum trades far fewer,
  far more liquid instruments -- one reason the ETF expression can survive costs better.
- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5--32. ~32% post-publication attenuation;
  Moskowitz-Grinblatt (1999) is old enough to be well-arbitraged.
- **Shumway, T. (1997).** "The Delisting Bias in CRSP Data." *Journal of Finance*, 52(1),
  327--340. Delistings correlate with poor performance -- removing them biases a single-name
  loser leg upward. Our single-name basket is survivor-only, so its loser short is an
  upper-bound; the SPDR sector book is essentially survivorship-free.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*,
  55(3), 703--708. The HAC long-run-variance estimator in
  [`strategy.hac_tstat`](../industry_momentum/strategy.py).

## Related desk studies

- **[Study 225 -- Sector-Rotation](../../225-sector-rotation/)**: a different sector lens --
  rotation rules rather than a strict 12-1 cross-sectional momentum sort.
- **[Study 237 -- Residual-Momentum](../../237-residual-momentum/)**: single-name momentum on
  *residual* (idio) returns -- the orthogonal cut to "is it the industry?".
- **[Study 146 -- Country-Momentum](../../146-country-momentum/)** and
  **[Study 147 -- FX-Momentum](../../147-fx-momentum/)**: the same 12-1 machine on countries
  and currencies -- momentum as a cross-asset phenomenon.
- **[Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/)**: shares the desk
  infrastructure (monthly cross-sectional sort, equal-weight legs, HAC inference, costs+borrow).
