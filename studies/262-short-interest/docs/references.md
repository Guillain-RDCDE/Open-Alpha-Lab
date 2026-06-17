# References & literature map -- Study 262 (Short-Interest)

## The canonical claim(s) -- two camps that disagree

- **Boehmer, E., Jones, C. M. & Zhang, X. (2008).** *Which Shorts Are Informed?*
  Journal of Finance, 63(2), 491--527.
  The "smart money" camp. Using proprietary NYSE order data, heavily shorted
  stocks subsequently *under-perform* lightly shorted ones by ~1.6%/month
  (risk-adjusted) -- short sellers are informed, so high short interest is a
  *bearish* signal. The implied trade is to short the heavily shorted names.

- **Desai, H., Ramesh, K., Thiagarajan, S. R. & Balachandran, B. V. (2002).**
  *An Investigation of the Informational Role of Short Interest in the Nasdaq
  Market.* Journal of Finance, 57(5), 2263--2287.
  Confirms that high short-interest Nasdaq firms earn significant negative
  abnormal returns, with the effect strongest in the most heavily shorted decile.

- **Asquith, P., Pathak, P. A. & Ritter, J. R. (2005).** *Short Interest,
  Institutional Ownership, and Stock Returns.* Journal of Financial Economics,
  78(2), 243--276.
  Finds the under-performance of high short-interest stocks is concentrated in
  stocks with *low* institutional ownership (a short-sale constraints story),
  and that the effect is economically modest after costs.

## The opposing "squeeze / reversal" camp

- **Lamont, O. A. (2012).** *Go Down Fighting: Short Sellers vs. Firms.* Review
  of Asset Pricing Studies, 2(1), 1--30.
  Documents how heavily shorted "battleground" stocks can squeeze violently; the
  crowded-short condition is exactly what produces explosive upside reversals.

- **GameStop / meme-stock episode (2021).** The canonical real-world squeeze:
  a stock with short interest reportedly exceeding 100% of float rose ~20x in
  weeks as shorts covered. A vivid reminder that high short interest is also a
  *bullish* time bomb -- the opposite of the Boehmer-Jones-Zhang prediction.

## Mechanism, constraints, and decay

- **Diether, K. B., Lee, K.-H. & Werner, I. M. (2009).** *Short-Sale Strategies
  and Return Predictability.* Review of Financial Studies, 22(2), 575--607.
  Short-selling *flow* (not just the static level) predicts returns at short
  horizons; the static bi-monthly short-interest snapshot is a coarse, stale
  proxy for what informed shorts are actually doing.

- **D'Avolio, G. (2002).** *The Market for Borrowing Stock.* Journal of Financial
  Economics, 66(2-3), 271--306.
  The heavily shorted names are exactly the hard-to-borrow, high-fee names. Any
  edge from shorting them is taxed by borrow costs of 5-50%+/yr -- the central
  tradability problem for this study.

- **McLean, R. D. & Pontiff, J. (2016).** *Does Academic Research Destroy Stock
  Return Predictability?* Journal of Finance, 71(1), 5--32.
  Anomaly returns decay ~58% post-publication; the short-interest effect,
  documented in the early 2000s, is among the most arbitraged.

## Why this study is honest about its limits

This study is a *single cross-sectional snapshot* of ~60 names (the hardcoded
short-interest table), not a multi-decade panel. It therefore cannot deliver the
robust time-series HAC *t* >= 2 the desk requires to call a signal REAL: one
roll of the dice over 60 stocks is high-variance, and the two camps above
(informed shorts vs squeeze) partially cancel. The full Boehmer-Jones-Zhang
result needs the long monthly NYSE panel with delisting-inclusive returns, which
is out of scope here.

## Related desk studies

- **[Study 223 -- Same-Month Seasonality](../../223-same-month-seasonality/)**:
  another cross-sectional sort; there the signal is a robust time series, here it
  is a single snapshot.
- **[Study 158 -- Super-Bowl](../../158-super-bowl/)**: the hardcoded-table
  pattern this study mirrors for its offline core.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *Econometrica*.
- **Permutation / label-shuffle test.** Fisher (1935), *The Design of
  Experiments* -- the exact null used here for the cross-sectional spread.
- **Bootstrap CI.** Efron (1979), *Bootstrap Methods* (Annals of Statistics).
- **Survivorship notation.** Shumway (1997), *The Delisting Bias in CRSP Data*
  (Journal of Finance).
