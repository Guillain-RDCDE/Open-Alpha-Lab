# References & literature map — Study 77 (Golden-Mean)

## The claim under test

- **Fibonacci retracements as support/resistance.** A pervasive belief in technical
  analysis: when price pulls back from a recent swing high (or low), it will find
  support (or resistance) at levels corresponding to 23.6%, 38.2%, 50%, 61.8%, or
  78.6% of the swing range — ratios derived from the Fibonacci sequence. The "golden
  ratio" 61.8% is considered the most powerful. Similarly, round numbers ($100, $500,
  SPX 5000, BTC $100,000) are said to act as psychological support/resistance because
  large options positions and stop orders cluster there. The folk hypothesis is that
  *these specific levels* attract more bounces than neighbouring price points, implying
  a statistically detectable difference between Fibonacci/round-number levels and
  placebo control levels within the same swing.

## Why the steelman is almost coherent — what the claims lean on

- **Order clustering at round numbers.** Harris (1991), *Stock Price Clustering and
  Discreteness* (Review of Financial Studies), documents that stock prices and
  transactions cluster at round numbers — a real microstructure effect. If stop-loss
  and limit orders accumulate at round numbers, touching those levels triggers a wave
  of order flow that *could* in principle create a detectable bounce. Our round-number
  vs midpoint test directly measures this claim. The effect appears in the raw data but
  is +1.16 bps in 5-day returns — statistical noise.
- **Self-fulfilling prophecy via attention and order clustering.** Osler (2003),
  *Currency Orders and Exchange Rate Dynamics: An Explanation for the Predictability of
  Technical Analysis* (Journal of Finance), documents that stop-loss and take-profit
  orders cluster at round exchange-rate levels, producing short-lived price clustering.
  For equities, the analogous claim is weaker: millions of discretionary traders
  watching Fibonacci levels could, in principle, make them partially self-fulfilling.
  Our placebo control is designed to cleanly test whether the self-fulfilling effect is
  specific to Fibonacci ratios or generic to any level that traders watch.
- **Psychological round numbers in equity markets.** Donaldson & Kim (1993),
  *Price Barriers in the Dow Jones Industrial Average* (Journal of Financial and
  Quantitative Analysis), report that the DJIA slows near multiples of 100. Mitchell
  (2001), *The Impact of External Parties on Brand-Name Capital: The 1982 Tylenol
  Poisonings and Subsequent Cases* — a reminder that extraordinary claims require
  robust controls.
- **Fibonacci specifically: negative or mixed evidence.** Bhattacharya & Bhattacharya
  (2012), *Does the Fibonacci Sequence Have Any Support From the Evidence? A Study on
  the Return Dynamics of S&P 500* (IUP Journal of Applied Finance), find no reliable
  Fibonacci-level predictability after accounting for the base rate of price levels
  in any range. Kempen (2016), *Fibonacci Retracements: A Study of the Predictive
  Power of Fibonacci Numbers in the S&P 500* (working paper), confirms weak or absent
  Fibonacci effects when tested against a proper control. Both are consistent with our
  finding that placebo levels outperform Fibonacci by 3.98 bps.

## The two traps this study is really about

- **Confusing equity drift with level magic.** A naive event study that counts how
  often price goes up after touching a Fibonacci level will find a positive number —
  because stocks go up most of the time. The only valid comparison is Fibonacci vs an
  equally well-designed control level. Our placebo arm captures this: both Fibonacci
  (53.3% bounce rate) and placebo (53.5% bounce rate) are nearly identical, and both
  merely reflect the positive expected return of the market.
- **Survivorship of levels.** Fibonacci practitioners often *post-hoc* identify which
  levels "worked" — the same cherry-picking fallacy that inflates technical analysis
  generally (Sullivan, Timmermann & White 1999, *Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap*, Journal of Finance). With five possible Fibonacci
  ratios and the ability to choose the swing and direction, the level of implicit
  multiple testing is enormous. Our protocol prevents this by pre-specifying the three
  "classic" levels (38.2%, 50%, 61.8%) and measuring all of them consistently.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*
  (Econometrica) — [`strategy.summarize`](../golden_mean/strategy.py).
- **Event study with control arm.** Fama, Fisher, Jensen & Roll (1969), *The
  Adjustment of Stock Prices to New Information* (International Economic Review) —
  the design of comparing signal-arm to placebo-arm on identical entry events.
- **Reproducibility stamp.** Content fingerprints on each headline tape, as-of date,
  and frozen R-dict in the notebooks mirror the desk's standard reproducibility
  protocol.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), daily OHLCV across six liquid
  instruments (SPY, QQQ, AAPL, MSFT, TSLA, NVDA). Daily data stretches back ~25
  years for most instruments, giving substantial power for the event-study approach.
  Every headline is pinned with a content fingerprint and as-of date.
  See [`docs/results.md`](results.md).

## Related desk studies

- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the SMA(5/10) crossover scalp —
  the same "does this signal beat a fair coin?" protocol applied to a moving-average
  signal rather than a price-level signal.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 golden cross — another
  technical analysis claim tested against a random baseline.
- **[Study 17 — Glass-Ceiling](../../17-glass-ceiling/)**: resistance breakouts — the
  *opposite* claim to Fibonacci support (price breaking through resistance rather than
  bouncing at it), also finding no edge.
- **[Study 02 — Falling-Knife](../../02-falling-knife/)**: the "buying the dip" claim,
  which relies on the same kind of level-based reasoning ("price has fallen X%, it must
  bounce") and is also NONE/Mirage.
