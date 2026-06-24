# References & literature map — Study 409 (Tweezer Tops & Bottoms)

## The claim under test

- **The folk recipe.** A *tweezer bottom* is two consecutive candles whose **lows match**
  (the second tests the same floor and holds) after a decline — read as a bullish reversal,
  *buy*. A *tweezer top* is two candles whose **highs match** after a rally — a bearish
  reversal, *sell*. The pattern is standard in every candlestick curriculum. We steelman it
  as: *after two aligned wicks, the forward 1–10-day return reverses by more than the same
  stock's unconditional base rate, and the timing beats a random-date placebo.*

## Where the claim comes from

- **Steve Nison, *Japanese Candlestick Charting Techniques* (1991, 2nd ed. 2001).** The
  book that introduced Japanese candlesticks to Western traders; it catalogues tweezer tops
  and bottoms among the two-candle reversal patterns and stresses that they "carry more
  weight" when they appear after an extended trend (the trend-filter folklore our myth-check
  tests).
- **Thomas Bulkowski, *Encyclopedia of Candlestick Charts* (2008)** and his
  ThePatternSite statistics. Bulkowski reports tweezer patterns among the *least* reliable
  candlestick formations — a published reliability ranking consistent with this teardown's
  NONE.
- **Gregory Morris, *Candlestick Charting Explained* (3rd ed., 2006).** A widely cited
  practitioner reference that documents the pattern definitions and the "needs a prior
  trend" qualifier.

## Why a steelman is *almost* coherent — and why it fails

- **Support / resistance intuition.** A price level tested twice and held *looks* like a
  defended line. But "two daily lows within 10 bps of each other after a down-leg" is a
  common coincidence on liquid names (≈ 22 tweezer events per name per year here), and a
  coincidence is not a forecast.
- **The base-rate confound — the central lesson.** US equities drift up, so *any* long
  trade backtests as significant. The raw tweezer-bottom HAC *t* of +4.16 is entirely this
  drift: its +26.78 bps *undershoots* the +32.35 bps unconditional base rate. The honest
  frame is excess-over-base-rate (or a market-neutral pairing), which reads ≈ 0. This is the
  canonical way candlestick backtests mislead — see Park & Irwin (2007).
- **Technical-pattern profitability is fragile.** Park & Irwin (2007), *"What Do We Know
  About the Profitability of Technical Analysis?"* (Journal of Economic Surveys), survey
  ~95 studies and find chart-pattern edges are mostly artefacts of data-snooping, period
  selection, and unadjusted benchmarks. Marshall, Young & Rose (2006), *"Candlestick
  Technical Trading Strategies: Can They Create Value for Investors?"* (Journal of Banking &
  Finance), test candlestick rules on the DJIA and find **no value** after a bootstrap
  correction — directly on point.
- **Weak-form efficiency.** Fama (1970), *"Efficient Capital Markets"* (Journal of Finance):
  a two-bar visual shape on liquid large-caps is exactly the kind of public, costless signal
  efficiency predicts will not predict.

## The failure mode exposed

- **Borrowed beta masquerading as a signal.** The pattern's only positive number is the
  market's drift it sits on top of. Strip the drift and the excess *t* never clears ±1.5 at
  any horizon, and a date-shuffle placebo (random entry days, same count and long/short mix)
  matches the real signal at *p* = 0.32.
- **The "needs a trend" rescue doesn't work.** Removing the prior-trend filter leaves the
  excess equally null — the qualifier adds no information (myth-check).

## Method lineage (the desk's shared engine)

- **HAC / Newey–West *t*-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  implemented in [`strategy._hac_t`](../tweezer_tops_bottoms/strategy.py).
- **Label/date-shuffle placebo.** A permutation null in the spirit of the bootstrap
  reality-check literature — Brock, Lakonishok & LeBaron (1992) and Sullivan, Timmermann &
  White (1999), *"Data-Snooping, Technical Trading Rule Performance, and the Bootstrap"*
  (Journal of Finance).
- **Reproducibility stamp.** As-of freeze + content fingerprint
  ([`data.fingerprint` / `data.panel_fingerprint`](../tweezer_tops_bottoms/data.py)),
  mirroring `docs/results.md`.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`, total-return adjusted), 10-year window
  (2016-06-24 → 2026-06-23) across 31 liquid US large-caps + SPY. The offline reproducible
  core and the synthetic control run on the deterministic
  [`data.synthetic_panel`](../tweezer_tops_bottoms/data.py) generator, never the network.
  Each headline is pinned with an as-of date and per-tape content fingerprints (see
  [`docs/results.md`](results.md)).

## Related desk studies

- **[Study 178 — CCI](../../178-cci/)**: a normalised oscillator's overbought/oversold rule
  — same "does a chart signal beat a fair benchmark?" question, same honest infrastructure.
- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: band-touch
  mean-reversion, the indicator cousin of a two-candle reversal.
- **[Study 363 — PEAD-Drift](../../363-pead-drift/)**: the gold-standard event study whose
  forward-return / placebo / cost shape this study mirrors (and where the effect is REAL —
  a useful contrast for what a passing pattern looks like).
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the golden cross — another
  much-taught chart shape that doesn't survive an honest benchmark.
