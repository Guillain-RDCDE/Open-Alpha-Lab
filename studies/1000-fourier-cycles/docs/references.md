# Sources & literature map — Study 1000 (The Cycle Hunt)

## The test that settles it

- **Fisher, R. A. (1929), "Tests of Significance in Harmonic Analysis", *Proceedings of the
  Royal Society A* 125(796), 54-59.** The exact distribution of the largest periodogram
  ordinate. Published before almost all of the market-cycle literature it refutes, and
  implemented here in `fisher_g_test`.
- **Bartlett, M. S. (1955), *An Introduction to Stochastic Processes*, CUP.** The properties of
  the periodogram, including the crucial fact that it is *not* a consistent estimator of the
  spectrum — its variance does not shrink as the sample grows, only its resolution improves.
- **Priestley, M. B. (1981), *Spectral Analysis and Time Series*, Academic Press.** The standard
  reference for everything in section 1, including windowing and the leakage/resolution
  trade-off.

## Spurious cycles

- **Slutzky, E. (1937), "The Summation of Random Causes as the Source of Cyclic Processes",
  *Econometrica* 5(2), 105-146.** The founding result: a moving average of random shocks
  *looks* cyclical. Everything in this study is a footnote to it.
- **Yule, G. U. (1926), "Why Do We Sometimes Get Nonsense-Correlations Between Time-Series?",
  *JRSS* 89(1), 1-63.** The companion warning, and the origin of "nonsense correlation".
- **Granger, C. W. J. (1966), "The Typical Spectral Shape of an Economic Variable",
  *Econometrica* 34(1), 150-161.** Economic series have a characteristic spectrum dominated by
  low frequencies with no peaks — which is the correct null for section 3.
- **Nelson, C. R. & Kang, H. (1981), "Spurious Periodicity in Inappropriately Detrended Time
  Series", *Econometrica* 49(3), 741-751.** Detrending done wrongly *creates* cycles. Section 1
  detrends linearly and says so for this reason.

## Cycles in markets specifically

- **Hurst, J. M. (1970), *The Profit Magic of Stock Transaction Timing*.** The cycle-trading
  tradition, included as the thing being tested.
- **Bachelier, L. (1900), *Théorie de la Spéculation*.** The original random-walk model, whose
  spectrum is exactly what section 2 simulates.
- **Lo, A. W. & MacKinlay, A. C. (1988), "Stock Market Prices Do Not Follow Random Walks",
  *Review of Financial Studies* 1(1), 41-66.** The variance-ratio evidence against a pure random
  walk — which is real, and is *not* evidence of periodicity, a distinction this study is built
  around.

## Neighbours on this desk

**996-palindrome-dates**, **067-monday-effect**, **283-sell-in-may**,
**999-cusum-change-points**, **992-vol-clustering-halflife**.
