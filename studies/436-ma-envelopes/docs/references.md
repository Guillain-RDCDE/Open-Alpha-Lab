# References & literature map — Study 436 (Moving-Average Envelopes)

## The claim under test

- **The folk recipe.** A *moving-average envelope* plots a simple moving average with two
  parallel lines a **fixed percent** above and below it (e.g. SMA(20) ± 5%). The technique is
  attributed to the 1960s chartist tradition — Joseph Granville and J. M. Hurst both popularised
  "moving-average bands" — and predates Bollinger Bands by two decades. The rule taught on
  modern charting sites (Investopedia, StockCharts ChartSchool, TradingView): when price pierces
  the **lower** envelope it is "oversold" → buy and ride it back to the mid; when it pierces the
  **upper** envelope it is "overbought" → sell/fade. Envelope advocates argue the *fixed-percent*
  band is **better than Bollinger Bands** because it does not "breathe" with volatility, giving a
  cleaner, more stable signal. We steelman this as: *a percent envelope, turned into a long/flat
  timing rule on a liquid index, earns a higher net excess-of-cash Sharpe than buy-and-hold —
  and a higher one than the volatility-scaled Bollinger Band.*

## Why the steelman is *almost* coherent — the real effect it leans on

- **Short-horizon mean reversion exists — but mostly in the cross-section.** Lehmann (1990),
  *"Fat Tails, Edge Preservation, and Optimal Suboptimal Portfolio Choice"*, and Jegadeesh
  (1990), *"Evidence of Predictable Behavior of Security Returns"* (Journal of Finance),
  document one-week / one-month **reversal at the individual-stock level**. A band around a
  moving average is a deviation-from-mean detector, so the logic is not absurd — but index ETFs
  (SPY, QQQ, DIA) net out the idiosyncratic reversal that drives those results.
- **Bollinger Bands as the natural benchmark.** Bollinger (2001), *Bollinger on Bollinger
  Bands*, makes the band half-width proportional to a 20-day standard deviation — the direct
  volatility-scaled competitor to the fixed-percent envelope, and the obvious "is it better?"
  control. Study [104 — Bollinger-Reversion](../../104-bollinger-reversion/) finds the Bollinger
  lower-band buy adds only ~52 bps over a random-day buy (*t* ≈ 0.6); the envelope here is the
  even-simpler cousin and fares no better.
- **Indices trend; they do not mean-revert around a 20-day average on a tradable horizon.**
  The structural upward drift of broad equity indices (earnings growth, the equity risk premium)
  means a "buy the dip below the average" rule mostly *under-participates* in the very drift it is
  trying to harvest — it sits in cash 93% of the time and collects a fraction of the beta.

## The failure mode exposed

- **Diluted beta dressed as a timing edge.** The envelope is in the market ~6.6% of the time and
  earns a positive Sharpe simply because the market drifts up — but a block-permutation placebo
  that **keeps the exposure and scrambles the timing** is not beaten (*p* = 0.23). The band
  carries no information; the long-fraction does. This is the same "is it the signal or just being
  long?" trap dissected in Study 104 and Study 178.
- **Fixed-percent is the *weaker* band.** Because the envelope half-width ignores volatility, it
  fires too rarely in calm regimes and too late in turbulent ones — exactly when a
  volatility-scaled band adapts. On SPY the Bollinger version earns +0.444 Sharpe vs the
  envelope's +0.278. The folk claim that percent envelopes beat Bollinger is **backwards** here.
- **Data-snooping on the band parameters.** Tightening to a 2–3% envelope nudges the *t* above 2,
  but only by raising exposure (more dip-buys) — i.e. by adding beta, which the permutation test
  immediately flags. Sullivan, Timmermann & White (1999), *"Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap"* (Journal of Finance), and Brock, Lakonishok & LeBaron (1992),
  *"Simple Technical Trading Rules and the Stochastic Properties of Stock Returns"* (Journal of
  Finance), document how technical-rule "edges" evaporate once selection and exposure are
  controlled for. Park & Irwin (2007), *"What Do We Know About the Profitability of Technical
  Analysis?"* (Journal of Economic Surveys), survey the broad null result.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  [`strategy.hac_tstat`](../ma_envelopes/strategy.py).
- **Block bootstrap / permutation placebo.** Politis & Romano (1994), *"The Stationary
  Bootstrap"* (JASA) — the circular block scramble in
  [`strategy.block_permutation_pvalue`](../ma_envelopes/strategy.py).
- **Reproducibility stamp.** As-of freeze + content fingerprint each headline run carries
  ([`data.fingerprint`](../ma_envelopes/data.py)).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`, `auto_adjust=True` → total-return adjusted),
  five liquid index tapes (SPY, QQQ, DIA, IWM, EFA) back to inception. The offline reproducible
  core and the test-suite run on the deterministic [`data.synthetic_panel`](../ma_envelopes/data.py)
  generator, never the network. Each headline is pinned with an as-of date and a per-tape content
  fingerprint (see [`docs/results.md`](results.md)).

## Related desk studies

- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: the volatility-scaled
  band this study uses as its head-to-head benchmark — same mean-reversion family, same null.
- **[Study 178 — CCI](../../178-cci/)**: Lambert's oscillator, another normalised
  deviation-from-mean rule on daily equities; also carries no exploitable edge.
- **[Study 106 — Supertrend](../../106-supertrend/)** and the SMA crossover teardowns: the
  trend-following counterpart, same honest "does the indicator beat just being long?" treatment.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 golden cross — the bare moving
  average as a timing filter, the SMA(200) benchmark used here in miniature.
