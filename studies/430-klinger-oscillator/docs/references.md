# References — Study 430 (Klinger Volume Oscillator)

## The claim's source

- **Stephen J. Klinger**, *"The Klinger Oscillator,"* in **Technical Analysis of Stocks &
  Commodities** (1997) — the original publication of the Volume Force / Klinger Volume Oscillator
  (KVO). Klinger's pitch is explicitly that the oscillator reads **long-term volume** to
  anticipate price turns (accumulation vs distribution), i.e. that **volume leads price**.
- **Investopedia — "Klinger Oscillator"** and **StockCharts / TradingView indicator docs** —
  the modern restatements of the folk rule: long when KVO is above zero (or above its signal
  line), flat/short when below. These are the steelmanned "be-long on accumulation" framings we
  tested.

## Volume, price, and "leading" indicators

- **Karpoff, J. (1987), "The Relation Between Price Changes and Trading Volume: A Survey,"**
  *Journal of Financial and Quantitative Analysis* — the classic survey; volume relates to the
  *magnitude* of price moves, not their *direction* or *timing* (no clean "lead").
- **Lo, A. & Wang, J. (2000), "Trading Volume,"** *Review of Financial Studies* — modern
  treatment of what volume does and does not forecast.
- **Granville, J. (1963), On-Balance Volume** — the ancestor "volume leads price" indicator;
  the same category of claim, repeatedly found to be a lagging transform of past data.
- General point used in the teardown: KVO is a **difference of EMAs** of a volume series — a
  low-pass filter. Filters *delay* their input; "a lagging average leads price" is a category
  error. (See any signal-processing treatment of EMA phase lag.)

## Benchmarks the rule is raced against

- **Faber, M. (2007), "A Quantitative Approach to Tactical Asset Allocation,"** *Journal of
  Wealth Management* — the 10-month / 200-day moving-average trend filter, the dumb simpler rule
  Klinger must beat (and doesn't).
- **Appel, G. (1979), Moving Average Convergence/Divergence (MACD)** — the other obvious
  price-only competitor.

## Shared method (the desk's protocol)

- **Newey, W. & West, K. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix,"** *Econometrica* — the HAC *t* used on the
  daily strategy-minus-benchmark difference.
- **White, H. (2000), "A Reality Check for Data Snooping,"** *Econometrica* & **Politis & Romano
  (1994), stationary/circular block bootstrap** — the spirit of the block-permutation placebo
  (shuffle position blocks, preserve persistence, ask if a random schedule does as well).
- **Sharpe, W. (1994), "The Sharpe Ratio,"** *Journal of Portfolio Management* — excess-vs-excess
  risk-adjusted comparison.
- House method: [`../../../METHODOLOGY.md`](../../../METHODOLOGY.md) (the seven beats, the
  inference bar, the verdict rubric).

## Related desk studies

- [`../178-cci`](../178-cci) — Commodity Channel Index: another oscillator that fails the
  follow-the-extreme test and lands NONE × MIRAGE.
- [`../104-bollinger-reversion`](../104-bollinger-reversion) — Bollinger-band reversion: the
  random-entry / fair-coin control idiom this study borrows.
- [`../109-obv-divergence`](../109-obv-divergence) — On-Balance-Volume divergence: the sibling
  "volume leads price" claim, the same lagging-volume-transform category.
- [`../343-data-mining-roulette`](../343-data-mining-roulette) — why grid-searching KVO spans
  until something beats buy-and-hold is curve-fitting, not evidence.
- [`../363-pead-drift`](../363-pead-drift) — the gold-standard real-tape study whose shape and
  synthetic-control discipline this one follows.
