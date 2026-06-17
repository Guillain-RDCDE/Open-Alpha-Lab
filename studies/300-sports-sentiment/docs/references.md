# References & literature map -- Study 300 (Sports-Sentiment)

## The claim under test

**Edmans, A., Garcia, D. & Norli, O. (2007).** "Sports Sentiment and Stock
Returns." *The Journal of Finance*, 62(4), 1967-1998.
The canonical result. Using a panel of 39 countries and international soccer
results (World Cup, Euros, Copa America, plus cricket, rugby, basketball),
the authors find that a national team's **elimination loss** is followed by a
significantly **negative** next-day return in that country's stock market --
roughly **-49 bps** on average for World Cup losses, and larger for important
games and smaller countries. The effect is strongly **asymmetric**: wins do not
produce an equal-and-opposite gain. Their interpretation is mood: a salient,
nationally-shared disappointment depresses investor risk appetite for a day.
Crucially, the shock is **dated and exogenous** -- the fixture is scheduled
years in advance and the result is orthogonal to fundamentals, which rules out
most reverse-causality and omitted-variable explanations.

## Why a tradable re-test can still come up empty

- **Proxy mismatch.** Edmans et al. use **broad national stock indices in local
  currency**. An outside trader reaches the market through a **single-country
  ETF** (EWU, EWG, EWQ, EWZ...), which holds a different (large-cap, liquid)
  basket, is priced in **USD**, and trades in **US hours**. A fragile one-day
  mood effect in the local cash market need not survive that translation.
- **Sample size.** The original pools thousands of country-match observations
  across many sports. A curated set of ~60 marquee soccer eliminations on liquid
  ETF markets is far smaller; the per-event noise (~120 bps daily vol) swamps a
  40-50 bps signal at n=60 (the minimum detectable mean at |t|=2 is ~30 bps).
- **Selection of events.** We hardcode *knockout elimination losses* only (the
  asymmetric, negative-mood case Edmans emphasises), but marquee games skew
  toward strong economies and late tournament stages -- a different mix than the
  full panel.

## Method lineage

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
  *Econometrica*, 55(3), 703-708. We test H0: mean next-day return = 0 with a
  HAC standard error (Bartlett kernel, 1 lag) -- guarding against same-day
  co-movement when two countries are eliminated on the same tournament day.
- **Event study / abnormal return.** The clean way to measure the original
  effect is a market-model abnormal return on the local index. Our tradable
  proxy uses the raw next-session ETF return (price-only), which is the quantity
  a short-on-loss strategy would actually earn.
- **Bootstrap.** i.i.d. resampling of the event returns to get the sampling
  distribution of the mean and a one-sided p-value (loss effect wants the mean
  well below zero).
- **Positive control.** A synthetic panel with a planted -49 bps effect confirms
  the test recovers the signal when it exists, so a null on the real tape is a
  statement about the proxy/sample, not the method.

## Related literature on mood and markets

- **Hirshleifer, D. & Shumway, T. (2003).** "Good Day Sunshine: Stock Returns
  and the Weather." *Journal of Finance*, 58(3), 1009-1032. Morning sunshine at
  the exchange city correlates with higher returns -- the same mood-pricing
  channel.
- **Kamstra, M., Kramer, L. & Levi, M. (2003).** "Winter Blues: A SAD Stock
  Market Cycle." *American Economic Review*, 93(1), 324-343. Seasonal Affective
  Disorder and equity returns -- mood again, on a calendar.
- **Saunders, E. M. (1993).** "Stock Prices and Wall Street Weather."
  *American Economic Review*, 83(5), 1337-1345. The early weather-and-mood paper.

## Data sources

- **iShares MSCI single-country ETFs** (EWU UK, EWG Germany, EWQ France,
  EWI Italy, EWP Spain, EWN Netherlands, EWW Mexico, EWZ Brazil/regional proxy)
  and **^GSPC** for the US, daily closes via yfinance, cached per-ticker under
  `_cache/sports_<TICKER>_daily.parquet`. Price-only (no dividends), USD.
- **Elimination dates** are historical tournament facts (FIFA World Cup, UEFA
  European Championship, CONMEBOL Copa America), hardcoded in `data.py`.

## Related desk studies

- **[Study 158 -- Super-Bowl](../../158-super-bowl/)**: a sports result as a
  market omen -- the spurious-correlation cousin of this mood study.
- Weather / SAD / weekend mood-and-calendar studies elsewhere on the desk.
