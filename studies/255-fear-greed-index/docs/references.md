# References & literature map -- Study 255 (Fear-Greed)

## The claim

- **CNN Business -- Fear & Greed Index.** The index is a 0--100 composite of
  seven equally-weighted sub-indicators: market momentum (S&P 500 vs its
  125-day average), stock-price strength (52-week highs vs lows on the NYSE),
  stock-price breadth (McClellan volume), put/call options ratio, junk-bond
  demand (yield spread), market volatility (VIX vs its 50-day average), and
  safe-haven demand (stocks vs Treasuries over 20 days).  CNN launched it
  around 2011.  Bands: 0-24 Extreme Fear, 25-44 Fear, 45-55 Neutral,
  56-74 Greed, 75-100 Extreme Greed.  The folk trading rule is contrarian:
  *buy in Extreme Fear, lighten/sell in Extreme Greed.*

- **Buffett, W.** popularised the maxim *"Be fearful when others are greedy and
  greedy when others are fearful"* (Berkshire Hathaway shareholder letters /
  2008 NYT op-ed).  The Fear & Greed Index is the retail-facing operationalisation
  of that contrarian intuition.

## Does sentiment forecast returns? The academic record

- **Baker, M. & Wurgler, J. (2006).** *Investor Sentiment and the Cross-Section
  of Stock Returns.* Journal of Finance, 61(4), 1645--1680.  Builds a composite
  sentiment index (the closest academic analogue to Fear & Greed) and finds it
  predicts the *cross-section* (small, young, volatile, distressed stocks) far
  more than the aggregate market.  Aggregate-level timing power is weak.

- **Tetlock, P. C. (2007).** *Giving Content to Investor Sentiment: The Role of
  Media in the Stock Market.* Journal of Finance, 62(3), 1139--1168.  High media
  pessimism predicts downward pressure on prices followed by reversion -- a
  short-horizon, largely transitory effect, not a durable timing edge.

- **Brown, G. W. & Cliff, M. T. (2004).** *Investor Sentiment and the Near-Term
  Stock Market.* Journal of Empirical Finance, 11(1), 1--27.  Sentiment is
  strongly contemporaneously correlated with returns but has little *predictive*
  power for near-term market moves -- the survey/sentiment level moves *with*
  the market, not ahead of it.

- **Da, Z., Engelberg, J. & Gao, P. (2015).** *The Sum of All FEARS.* Review of
  Financial Studies, 28(1), 1--32.  A Google-search "FEARS" sentiment index
  predicts short-run return *reversals* and volatility -- evidence that extreme
  sentiment matters most at very short horizons, consistent with a weak,
  transitory, hard-to-trade weekly effect.

## Why a contrarian sentiment timer usually fails out of sample

- **Welch, I. & Goyal, A. (2008).** *A Comprehensive Look at the Empirical
  Performance of Equity Premium Prediction.* Review of Financial Studies, 21(4),
  1455--1508.  The canonical demolition of market-timing predictors: most
  in-sample predictive variables fail to beat the prevailing-mean benchmark
  out of sample.  A sentiment gauge is in exactly this family.

- **Bailey, D. H., Borwein, J., Lopez de Prado, M. & Zhu, Q. J. (2014).**
  *Pseudo-Mathematics and Financial Charlatanism.* Notices of the AMS, 61(5).
  Backtest overfitting and the danger of selecting a threshold (here, the
  25/75 Extreme-Fear/Greed cutoffs) after seeing the data.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (Econometrica).
- **Block-bootstrap Sharpe CI.** Politis & Romano (1994), *The Stationary
  Bootstrap* (JASA).

## Related desk studies

- **[Study 158 -- Super-Bowl](../../158-super-bowl/)** and the wider folklore
  cohort: binary "predictors" that mostly ride the unconditional up-rate.
- **[Study 223 -- Same-Month Seasonality](../../223-same-month-seasonality/)**:
  the structural reference this study mirrors (synthetic panel + cached real tape).
- VIX-term and MOVE volatility studies in the cross-asset family share the
  "sentiment / fear gauge as timing signal" question.
