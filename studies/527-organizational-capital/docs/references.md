# References & literature map -- Study 527 (Organizational-Capital)

## The primary claim under test

- **Eisfeldt, A. L. & Papanikolaou, D. (2013).** "Organization Capital and the Cross-Section
  of Expected Returns." *Journal of Finance*, 68(4), 1365--1406. The founding paper. They
  build a firm-level **organizational capital** stock by perpetual inventory of SG&A
  expenditures (depreciation `delta = 15%/yr`, a steady-state seed, CPI-deflated), scale it
  by book assets, and sort. Firms with **high** organizational capital earn returns ~4.6%/yr
  higher than low-OC firms. Their risk story: organizational capital is embodied partly in
  *key talent*, which is mobile -- shareholders bear the risk that this input can be hired
  away or can extract rents, so high-OC firms command a risk premium.

## Why the effect might exist -- the theoretical backbone

- **Prescott, E. C. & Visscher, M. (1980).** "Organization Capital." *Journal of Political
  Economy*, 88(3), 446--461. The original conceptual treatment: information and matching
  accumulated within a firm (about workers, teams, processes) is a productive but intangible
  asset. The intellectual root of treating SG&A as investment rather than pure expense.
- **Atkeson, A. & Kehoe, P. J. (2005).** "Modeling and Measuring Organization Capital."
  *Journal of Political Economy*, 113(5), 1026--1053. A general-equilibrium model that
  measures the aggregate organizational-capital stock and motivates the perpetual-inventory
  accounting Eisfeldt-Papanikolaou later apply at the firm level.
- **Lev, B. & Radhakrishnan, S. (2005).** "The Valuation of Organization Capital." In
  *Measuring Capital in the New Economy*, NBER. Develops an accounting estimate of
  organizational capital from SG&A and shows it predicts firm productivity and value.

## Subsequent evidence, replication, and skepticism

- **Peters, R. H. & Taylor, L. A. (2017).** "Intangible Capital and the Investment-q
  Relation." *Journal of Financial Economics*, 123(2), 251--272. Splits intangibles into
  knowledge capital (from R&D) and organizational capital (from SG&A) via perpetual
  inventory; widely used as the standard capitalization recipe. The exact construction we
  borrow (SG&A -> OC stock with `delta = 0.15` after a steady-state seed).
- **Eisfeldt, A. L., Falato, A., & Xiaolan, M. Z. (2023).** "Human Capitalists." *NBER
  Macroeconomics Annual*. Documents the growing share of firm value tied to skilled-labour
  (equity-compensated) talent -- the mobile input behind the OC risk premium -- and how it
  reshapes the cross-section.
- **Hou, K., Xue, C., & Zhang, L. (2020).** "Replicating Anomalies." *Review of Financial
  Studies*, 33(5), 2019--2133. Many published cross-sectional anomalies attenuate or vanish
  under uniform replication, value-weighting, and microcap screens; intangible/quality sorts
  are among the more fragile. A caution against taking a single published t-stat at face value.
- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5--32. Documents ~32% post-publication
  attenuation of anomaly returns; the OC premium (published 2013) is squarely in the window
  where arbitrage and crowding would erode it.

## Survivorship bias and universe construction

- **Shumway, T. (1997).** "The Delisting Bias in CRSP Data." *Journal of Finance*, 52(1),
  327--340. Delistings correlate with poor performance; excluding them biases factor returns
  upward. Our current-large-cap basket omits firms that went bankrupt or were acquired --
  exactly the high-risk tail an OC risk story would put on the long leg.
- **Linnainmaa, J. T. & Roberts, M. R. (2018).** "The History of the Cross-Section of Stock
  Returns." *Review of Financial Studies*, 31(7), 2606--2649. Many anomalies are weaker
  out-of-sample and in periods unseen during their discovery; data-snooping and survivorship
  matter for any single-universe replication.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*,
  55(3), 703--708. The HAC long-run variance estimator behind the one-sample *t* in
  [`strategy.summary`](../organizational_capital/strategy.py).
- **U.S. SEC EDGAR XBRL Frames / Company Facts API** (data.sec.gov). The fundamentals source:
  ~18-19 years of annual 10-K SG&A and total assets per large-cap, used for the perpetual
  inventory.

## Related desk studies

- **[Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/)**: the same desk
  cross-sectional sort + HAC-inference infrastructure on a risk-based factor.
- **[Study 330 -- Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)**: another
  risk-premium claim where the headline ranking holds but the tradable spread does not certify.
