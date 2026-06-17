# References & literature map -- Study 269 (Baltic-Dry)

## The canonical claim

- **Baltic Exchange (1985-).** *The Baltic Dry Index (BDI).* A daily composite of
  freight rates for dry bulk cargo (Capesize, Panamax, Supramax) carrying iron
  ore, coal, and grain. Launched at 1000 in January 1985. Because freight is
  booked to move physical commodities and capacity is fixed in the short run, the
  index is widely described in market commentary as a "pure", un-gameable read on
  real global demand -- and therefore a leading indicator for industrial
  activity and, by extension, the stock market.

- **Bakshi, G., Panayotov, G. & Skoulakis, G. (2011).** *The Baltic Dry Index as
  a Predictor of Global Stock Returns, Commodity Returns, and Global Economic
  Activity.* AFA 2012 Chicago Meetings Paper. The most-cited academic test of the
  claim. Finds that BDI growth has *some* predictive power for global equity and
  commodity returns and for global economic activity at the 1-12 month horizon,
  strongest in-sample and around recessions. Subsequent work and our own
  replication show the effect is concentrated in crisis episodes and weak out of
  sample.

## Mechanism and skeptics

- **Kilian, L. (2009).** *Not All Oil Price Shocks Are Alike: Disentangling
  Demand and Supply Shocks in the Crude Oil Market.* American Economic Review,
  99(3), 1053-1069. The intellectual backbone for "freight/commodity prices as a
  demand proxy": global real-activity shocks move shipping and commodity markets.
  But the same logic implies the BDI is *contemporaneous* with demand, not a
  clean *lead* -- which is what our regression finds (the signal mostly fires
  alongside, not ahead of, the 2008 crash).

- **Apergis, N. & Payne, J. E. (2013).** *New Evidence on the Information and
  Predictive Content of the Baltic Dry Index.* International Journal of Financial
  Studies. Documents that the BDI's predictive content is unstable across
  sub-periods and heavily influenced by the shipping-supply cycle (the 2003-2008
  newbuild order boom and subsequent glut), not just demand -- contaminating any
  "pure demand" interpretation.

- **Supply-side contamination.** The 2008-2012 collapse in the BDI was driven as
  much by a flood of newly delivered Capesize vessels (ordered during the
  2004-2008 boom) as by falling demand. A signal that mixes a demand read with a
  shipbuilding-cycle read is not the clean macro thermometer the folklore claims.

## Post-publication / out-of-sample decay

- **McLean, R. D. & Pontiff, J. (2016).** *Does Academic Research Destroy Stock
  Return Predictability?* Journal of Finance, 71(1), 5-32. Anomaly returns decay
  ~58% after publication. The BDI-predicts-stocks story has been in the popular
  press since the mid-2000s; our 2010-2026 sub-period slope (*t* = 2.28) is much
  weaker than the 2000-2009 supercycle/GFC decade (*t* = 5.08).

## Related desk studies

- **[Study 158 -- Super-Bowl](../../158-super-bowl/)**: a folklore predictor with
  a hardcoded event table and real market returns -- the same "famous indicator,
  honest teardown" pattern.
- **[Study 223 -- Same-Month-Seasonality](../../223-same-month-seasonality/)**:
  the structural template (synthetic positive-control generator + cached/proxy
  real series + HAC inference bar) this study mirrors.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (Econometrica). Used on both the regression slope and the strategy
  return series, because monthly macro-predictor regressions have strongly
  autocorrelated residuals.
- **Predictive-regression caveats.** Stambaugh (1999), *Predictive Regressions*
  (JFE): persistent predictors bias the slope and inflate t-stats in small
  samples -- a reason to discount a single-regime *t* = 2.8.
