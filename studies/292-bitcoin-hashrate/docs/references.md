# References & literature map -- Study 292 (Bitcoin-Hashrate)

## The folk claim ("price follows hashrate")

- **Capriole Investments -- "Hash Ribbons" (Charles Edwards, 2019).**
  *Bitcoin Hash Ribbons & Miner Capitulation.* The most-cited practitioner
  formulation: a crossover of the 30-day above the 60-day moving average of
  hash rate is read as the end of "miner capitulation" and a bullish buy
  trigger. Popularised the "price follows hashrate" narrative in crypto
  Twitter and newsletters. No peer review; the indicator is curve-fit to a few
  bull-market bottoms.

- **"Bitcoin's price follows hashrate" -- crypto folklore.** A recurring claim
  on r/Bitcoin and in mining-company investor decks. The causal story usually
  runs *backwards*: hash rate is a lagging function of price (miners switch on
  when BTC is expensive enough to be profitable), so any contemporaneous
  correlation is mostly price -> hashrate, not hashrate -> price.

## What the academic literature actually finds

- **Hayes, A. (2017).** *Cryptocurrency value formation: An empirical study
  leading to a cost of production model for valuing bitcoin.* Telematics and
  Informatics, 34(7), 1308--1321. Argues hash rate enters a *cost-of-production*
  floor for BTC, but the direction of causality is price-driven mining
  investment, not a leading signal.

- **Kristoufek, L. (2020).** *Bitcoin and its mining on the equilibrium path.*
  Energy Economics, 85, 104588. Finds a long-run cointegration between price,
  hash rate and difficulty, but the short-run dynamics are dominated by
  price; hash rate adjusts *to* price with a lag (the opposite of a leading
  indicator).

- **Fantazzini, D. & Kolodin, N. (2020).** *Does the hashrate affect the bitcoin
  price?* Journal of Risk and Financial Management, 13(11), 263. Direct test of
  the folk claim: after controlling for endogeneity, hash rate has **no robust
  predictive power** for future BTC prices; the apparent link is reverse
  causality and common trend.

## Methodological cautions

- **Granger, C. W. J. & Newbold, P. (1974).** *Spurious regressions in
  econometrics.* Two independent trending (I(1)) series will show high R^2 and
  significant slopes in levels even with no relationship. Hash rate and price
  both trend up over the sample -- the textbook spurious-regression trap, which
  is why this study works in *growth rates*, not levels.

- **Newey, W. K. & West, K. D. (1987).** HAC standard errors, used for every
  t-stat here.

- **Single-survivor bias.** BTC is the one cryptocurrency that survived and
  1000x'd; the hash-rate / price co-trend is conditioned on that survival. Any
  long-biased timing rule benefits mechanically. Named on the Signal axis.

## Related desk studies

- **[Study 158 -- Super-Bowl](../../158-super-bowl/)**: the folklore-pattern
  template -- a hardcoded event/series table pinned against real returns.
- **[Study 210 -- Crypto-Trend](../../210-crypto-trend/)** and
  **[Study 209 -- ETH-BTC-Ratio](../../209-eth-btc-ratio/)**: other crypto
  cross-asset / trend signals on the desk.
- **[Study 223 -- Same-Month-Seasonality](../../223-same-month-seasonality/)**:
  the data-driven reference pattern (synthetic panel + cached real series) this
  study mirrors structurally.
