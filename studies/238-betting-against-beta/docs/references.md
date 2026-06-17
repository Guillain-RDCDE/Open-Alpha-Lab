# References & literature map -- Study 238 (Betting-Against-Beta)

## The primary claim under test

- **Frazzini, A. & Pedersen, L. H. (2014).** "Betting Against Beta." *Journal of Financial
  Economics*, 111(1), 1--31. The founding paper for the BAB factor. The authors argue that
  investors who face leverage constraints (margin requirements, risk limits) prefer high-beta
  assets, driving up their prices and depressing their expected returns. Low-beta stocks are
  thus underpriced, and a leveraged long-low-beta / short-high-beta portfolio earns a
  "betting against beta" premium. Documented on 20 countries and multiple asset classes.
  Key result: BAB factor Sharpe ~0.65--0.78 on a broad US equity universe (1984--2012).

## Why the effect should exist -- the theoretical backbone

- **Black, F. (1972).** "Capital Market Equilibrium with Restricted Borrowing." *Journal of
  Business*, 45(3), 444--454. The seminal paper showing that if investors cannot borrow
  freely (or face a borrowing premium), the Security Market Line is flatter than CAPM
  predicts. Low-beta assets are underpriced relative to CAPM; high-beta assets are
  overpriced. Frazzini-Pedersen formalise and extend this intuition.
- **Black, F., Jensen, M. C., & Scholes, M. (1972).** "The Capital Asset Pricing Model:
  Some Empirical Tests." In *Studies in the Theory of Capital Markets*, ed. Michael C.
  Jensen. The early empirical finding that the market beta/return relationship is flatter
  than CAPM predicts -- consistent with the BAB mechanism.
- **Pedersen, L. H. (2015).** *Efficiently Inefficient: How Smart Money Invests and Market
  Prices Are Determined*. Princeton University Press. Chapter on the BAB factor explains
  the leverage-aversion mechanism in accessible terms.

## Subsequent evidence, replication, and attenuation

- **Hou, K., Xue, C., & Zhang, L. (2020).** "Replicating Anomalies." *Review of Financial
  Studies*, 33(5), 2019--2133. The BAB factor survives their replication battery with
  significant t-stats, though magnitudes are modest post-1980 on a broad universe.
- **Novy-Marx, R. & Velikov, M. (2016).** "A Taxonomy of Anomalies and Their Trading Costs."
  *Review of Financial Studies*, 29(1), 104--147. BAB has high turnover due to monthly
  rebalancing and leverage; trading costs materially reduce net returns on a retail
  implementation. Net alpha shrinks significantly vs gross returns.
- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5--32. Documents ~32% attenuation after
  publication; the BAB factor, published 2014, is susceptible to post-publication decay.
- **Buffa, A. M., Vayanos, D., & Woolley, P. (2022).** "Asset Management Contracts and
  Equilibrium Prices." *Journal of Political Economy*, 130(12), 3146--3201. Agency frictions
  in asset management can also generate the BAB premium -- investors/managers chase high-beta
  when benchmarked vs a cap-weighted index.

## The leverage constraint mechanism

- **Asness, C. S., Frazzini, A., & Pedersen, L. H. (2012).** "Leverage Aversion and Risk
  Parity." *Financial Analysts Journal*, 68(1), 47--59. Extends the BAB logic to asset
  allocation: leverage-averse investors underdiversify into high-beta/risky assets, giving
  risk parity strategies their foundation.
- **Baker, M., Bradley, B., & Wurgler, J. (2011).** "Benchmarking as a Source of Risk."
  *Financial Analysts Journal*, 67(1), 40--58. Active managers benchmarked to a cap-weighted
  index have an incentive to hold high-beta stocks (to keep tracking error low); this
  persistent demand inflates high-beta prices and reinforces the BAB premium.

## Survivorship bias and universe construction

- **Shumway, T. (1997).** "The Delisting Bias in CRSP Data." *Journal of Finance*, 52(1),
  327--340. Delistings are correlated with poor performance -- removing them biases factor
  returns upward. High-beta stocks are more likely to be delisted (bankruptcy, mergers), so
  survivorship bias particularly inflates the high-beta short leg's apparent return.
- **Linnainmaa, J. T. & Roberts, M. R. (2018).** "The History of the Cross-Section of Stock
  Returns." *Review of Financial Studies*, 31(7), 2606--2649. Shows many anomalies, including
  beta-based ones, are weaker in the pre-sample period; data-snooping and survivorship effects
  matter.

## Trading costs and implementability

- **Frazzini, A., Israel, R., & Moskowitz, T. J. (2015).** "Trading Costs of Asset Pricing
  Anomalies." Working paper, AQR. BAB has meaningful trading costs from leverage and monthly
  rebalancing; net of realistic institutional costs the premium shrinks but survives on a
  broad universe. Harder to implement for retail investors.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*,
  55(3), 703--708. The HAC long-run variance estimator in
  [`strategy.summary`](../betting_against_beta/strategy.py).

## Related desk studies

- **[Study 103 -- Turtle-Trader](../../103-turtle-trend/)**: momentum / trend-following --
  another strategy that relies on cross-sectional sorting and leverage.
- **[Study 120 -- Excess-CAPE-Yield](../../120-excess-cape-yield/)**: valuation-based timing
  -- a different lens on systematic market mispricing.
- **[Study 122 -- Gross-Profitability](../../122-gross-profitability/)**: Novy-Marx (2013)
  quality factor -- same desk infrastructure (rolling sort, equal-weight, HAC inference).
- **[Study 144 -- Permanent-Portfolio](../../144-permanent-portfolio/)**: a low-beta
  multi-asset alternative to equity-beta concentration.
