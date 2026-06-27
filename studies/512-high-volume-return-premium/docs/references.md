# References & literature map -- Study 512 (High-Volume-Return-Premium)

## The primary claim under test

- **Gervais, S., Kaniel, R. & Mingelgrin, D. H. (2001).** "The High-Volume Return Premium."
  *Journal of Finance*, 56(3), 877--919. The founding paper. Sorting NYSE stocks each
  formation period by *abnormal* trading volume (the period's volume relative to the stock's
  own recent norm), they find that high-volume stocks subsequently *appreciate* and
  low-volume stocks *depreciate* over horizons of one to several weeks. They attribute the
  premium to a visibility/attention shock: an unusual-volume day raises the stock's salience,
  expanding its investor base and (via Merton's investor-recognition logic) pushing its price
  up. This is the exact sort, the exact horizon, and the exact long-short we test here.

## Why the effect might exist -- the theoretical backbone

- **Merton, R. C. (1987).** "A Simple Model of Capital Market Equilibrium with Incomplete
  Information." *Journal of Finance*, 42(3), 483--510. The investor-recognition hypothesis:
  a stock known to more investors commands a higher price (lower expected return going
  forward, higher realised return as recognition spreads). GKM lean on this to explain why a
  visibility shock (high volume) predicts appreciation.
- **Barber, B. M. & Odean, T. (2008).** "All That Glitters: The Effect of Attention and News
  on the Buying Behavior of Individual and Institutional Investors." *Review of Financial
  Studies*, 21(2), 785--818. Attention-grabbing events (including abnormal volume) drive net
  buying by individuals -- a behavioural channel for a short-horizon volume->return link.
- **Lee, C. M. C. & Swaminathan, B. (2000).** "Price Momentum and Trading Volume." *Journal
  of Finance*, 55(5), 2017--2069. Volume conditions momentum: high-volume winners and
  low-volume losers behave differently going forward. Complementary evidence that volume
  carries cross-sectional return information.

## Subsequent evidence, replication, and attenuation

- **Kaniel, R., Ozoguz, A. & Starks, L. (2012).** "The High-Volume Return Premium:
  Cross-Country Evidence." *Journal of Financial Economics*, 103(2), 255--279. The premium
  appears in many international markets but with substantial cross-country variation; it is
  stronger where investor recognition frictions are larger -- i.e. *not* obviously in the
  most-watched US mega-caps.
- **Chordia, T., Subrahmanyam, A. & Anshuman, V. R. (2001).** "Trading Activity and Expected
  Stock Returns." *Journal of Financial Economics*, 59(1), 3--32. The *level* and *variability*
  of trading activity relate to expected returns (a negative volatility-of-volume premium),
  a different but adjacent volume effect that muddies a naive high-volume sort.
- **Chordia, T., Roll, R. & Subrahmanyam, A. (2011).** "Recent Trends in Trading Activity and
  Market Quality." *Journal of Financial Economics*, 101(2), 243--263. Volume and liquidity
  dynamics have shifted markedly since 2000 (decimalisation, HFT, ETFs); a premium identified
  on 1960s--1990s NYSE tapes need not survive on a modern large-cap tape.
- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5--32. Documents ~32% post-publication
  attenuation of anomalies on average; GKM (published 2001) is squarely in the window where
  decay is expected.

## Survivorship bias and universe construction

- **Shumway, T. (1997).** "The Delisting Bias in CRSP Data." *Journal of Finance*, 52(1),
  327--340. Delistings correlate with poor performance; removing failed names biases factor
  returns upward. A volume-spike sort is *especially* exposed -- a blow-up or takeover names
  trade on huge abnormal volume, and those are exactly the names absent from a survivor basket.
- **Linnainmaa, J. T. & Roberts, M. R. (2018).** "The History of the Cross-Section of Stock
  Returns." *Review of Financial Studies*, 31(7), 2606--2649. Many cross-sectional effects are
  weaker out-of-sample and on biased samples; data-snooping and survivorship effects matter.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703--708. The HAC
  long-run variance estimator behind the spread's HAC *t* in
  [`strategy.summary`](../high_volume_return_premium/strategy.py).

## Related desk studies

- **[Study 238 -- Betting-Against-Beta](../238-betting-against-beta/)**: same cross-sectional
  sort / equal-weight long-short / HAC-inference infrastructure, on a beta signal.
- **[Study 330 -- Low-Volatility-Anomaly](../330-low-volatility-anomaly/)**: a calm-vs-wild
  cross-section, the closest risk-sort neighbour.
- **[Study 109 -- OBV-Divergence](../109-obv-divergence/)**: an on-balance-volume timing
  signal -- a different way of asking whether volume leads price.
- **[Study 418 -- Money-Flow-Index](../418-money-flow-index/)**: the volume-weighted RSI; the
  nearest volume-indicator neighbour on the bench.
