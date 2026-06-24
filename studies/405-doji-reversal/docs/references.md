# References & literature map — Study 405 (Doji Reversal)

## The claim under test

- **The folk recipe.** The doji is the canonical single-candle reversal pattern in
  Japanese candlestick analysis, popularised in the West by **Steve Nison**, *Japanese
  Candlestick Charting Techniques* (1991, 2nd ed. 2001). A doji forms when the open and
  close are virtually equal (a tiny real body), signalling *indecision*: "neither bulls nor
  bears are in control." Nison and the retail-education canon teach that a doji after an
  advance warns of a top and a doji after a decline warns of a bottom — trade *against* the
  prior move. We steelman this as: *a doji, conditioned on the 2-day move into it, predicts
  a forward reversal that beats the unconditional against-the-move base rate, net of costs.*

## Why the steelman is *almost* coherent — the real effect it leans on

- **Short-horizon mean-reversion is genuinely there.** Individual US equities exhibit a
  small daily/weekly return reversal. **Jegadeesh (1990)**, *"Evidence of Predictable
  Behavior of Security Returns"* (Journal of Finance), documents one-month reversal at the
  stock level; **Lehmann (1990)**, *"Fads, Martingales, and Market Efficiency"* (QJE),
  documents weekly contrarian profits; **Lo & MacKinlay (1990)**, *"When Are Contrarian
  Profits Due to Stock Market Overreaction?"* (RFS), decompose how much is reversal vs
  cross-serial. This base-rate bounce is exactly what an against-the-prior-move bet
  harvests on *any* bar — and what a doji, sitting on a day that closed where it opened, is
  mistaken for.
- **The candle is a noisy proxy for a low-range day.** A doji is, mechanically, a session
  whose net move was small relative to its range. It is *not* a measurement of exhaustion or
  order-flow balance — so any forward predictability it shows must be checked against what
  an ordinary bar gives, not against zero.

## The failure mode exposed

- **Misattribution: baseline beats the candle.** Pinned against the unconditional base rate,
  the doji *underperforms* at every horizon (delta negative), and a label-shuffle placebo
  beats it ~84% of the time at 5 days. The "reversal" is the general short-horizon
  mean-reversion documented above, wearing a candlestick costume.
- **The trend-filter trap.** The folk refinement "only fade a doji against the trend"
  appears to work (below-SMA dojis bounce harder) but it is the same confound: *any*
  below-average bar bounces. Conditioning on the trend relabels the base-rate effect rather
  than isolating a doji effect.
- **Benchmark choice manufactures the edge.** Anyone backtesting "doji reversal" against a
  *zero* benchmark rather than the base rate will mistake the baseline reversion for a
  signal — the classic confound that **Park & Irwin (2007)**, *"What Do We Know About the
  Profitability of Technical Analysis?"* (Journal of Economic Surveys), and **Sullivan,
  Timmermann & White (1999)**, *"Data-Snooping, Technical Trading Rule Performance, and the
  Bootstrap"* (Journal of Finance), warn about. **Marshall, Young & Rose (2006)**,
  *"Candlestick Technical Trading Strategies: Can They Create Value for Investors?"*
  (Journal of Banking & Finance), tested candlestick patterns on the DJIA components and
  found **no value** beyond chance — directly on point.

## Method lineage (the desk's shared engine)

- **HAC / Newey–West *t*-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  implemented in [`doji_reversal.strategy.hac_tstat`](../doji_reversal/strategy.py).
- **Label-shuffle / permutation placebo.** Distribution-free null built by resampling
  same-size random bar sets — [`strategy.run_experiment`](../doji_reversal/strategy.py).
- **Reproducibility stamp.** Per-tape content fingerprint + explicit as-of, mirroring the
  desk's `quantlab/repro.py` convention (see [`docs/results.md`](results.md)).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), 2001-06-25 → 2025-12-31 across SPY + 28
  long-listed US large-caps (cache-first; offline once cached). The reproducible core and
  the positive control run on the deterministic
  [`data.synthetic_panel`](../doji_reversal/data.py) generator, never the network. Each
  headline is pinned with an as-of date and a content fingerprint (see
  [`docs/results.md`](results.md)).

## Related desk studies

- **[Study 178 — CCI](../../178-cci/)**: a normalised overbought/oversold oscillator, the
  same "does a technical reversal rule beat a fair benchmark?" question on daily equities.
- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: band-touch
  mean-reversion — the continuous cousin of the doji's reversal premise.
- **[Study 363 — PEAD-Drift](../../363-pead-drift/)**: the desk's event-study idiom
  (per-event forward windows vs a base rate), and a *real* effect for contrast.
- **[Study 343 — Data-Mining-Roulette](../../343-data-mining-roulette/)** and the
  research-method demos: how a benchmark or a free parameter manufactures a signal — the
  same misattribution lesson this study illustrates on real candles.
