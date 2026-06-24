# References & literature map — Study 402 (Engulfing Pattern)

## The claim under test

- **The folklore.** The engulfing candle is the single most-taught reversal in candlestick
  analysis: a small down ("red") candle followed by a larger up ("green") candle whose real
  body **engulfs** the prior body is a **bullish engulfing** — "the buyers have taken over,
  the bottom is in, go long." The mirror is a **bearish engulfing** — "the top is in, go
  short / sell." The promise is a *next-day (and next few-day) reversal* in the pattern's
  direction. It is in every charting app and every beginner course.
- **Where it comes from.** Steve Nison, *Japanese Candlestick Charting Techniques* (1991, NYIF)
  introduced candlestick patterns — including *tsutsumi* (the engulfing pattern) — to Western
  traders; Gregory Morris, *Candlestick Charting Explained* (1992/2006) catalogues the engulfing
  as a primary reversal signal. Both present it as a high-reliability turn marker, especially
  after a trend and on heavy volume.

## What the academic record actually says

- **Candlestick patterns generally fail out-of-sample.** Marshall, Young & Rose,
  *Candlestick technical trading strategies: Can they create value for investors?* (2006,
  Journal of Banking & Finance) test the full candlestick zoo on the Dow components and find
  **no value** once you bootstrap the null properly — the patterns do not beat a random-timing
  benchmark. Horton (2009, *Stars, crows, and doji*) reaches the same negative conclusion across
  hundreds of US names.
- **Engulfing specifically.** Studies that isolate the engulfing pattern (e.g. Lu, Shiu &
  Liu, 2012, on the Taiwan market; Zhu, Atri & Yegen, 2016) report weak, market-dependent, and
  often *direction-inconsistent* results — exactly the picture our US large-cap tape shows: the
  bullish leg rides the market's drift and the bearish leg fights it.
- **Why a high "win rate" is not an edge.** A pattern whose forward return is signed long after
  bullish and short after bearish is roughly market-neutral, so it must be judged against **zero**
  (and against the unconditional always-up drift), not against the eye-test of "it bounced." The
  desk's research-method demos (data-mining-roulette, multiple-testing) frame why.

## Method lineage (the desk's shared engine)

- **Precise OHLC detector.** [`strategy.is_engulfing`](../engulfing_pattern/strategy.py) — the
  textbook real-body engulfing rule (opposite colours, current body strictly larger and fully
  containing the prior body).
- **Event study with one execution lag.** [`strategy.forward_returns`](../engulfing_pattern/strategy.py)
  — confirm at the close of day *t*, **enter the next open** (day *t+1*), exit close +H; signed by
  pattern direction. The standard look-ahead-free event-study convention.
- **HAC inference + label-shuffle placebo.** [`strategy.hac_t`](../engulfing_pattern/strategy.py)
  (Newey & West, 1987, *A simple, positive semi-definite, heteroskedasticity and autocorrelation
  consistent covariance matrix*, Econometrica) and
  [`strategy.placebo_pvalue`](../engulfing_pattern/strategy.py) — draw the same number of random
  bars, sign each by a coin, and ask how often a random pick beats the observed mean (Fisher's
  randomization logic; Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **Deterministic synthetic control.** [`data.synthetic_panel`](../engulfing_pattern/data.py)
  plants a known day-after reversal proportional to a knob; with the edge at zero the detector
  must NOT manufacture significance, and with a planted edge it must light up — the offline core
  runs with no network.

## Data sources used here

- **yfinance** daily OHLCV (`auto_adjust=True`, total-return adjusted closes/opens) for a fixed
  **30-name** basket: 29 long-listed liquid US large-caps + **SPY**, 2005-01-03 → 2026-06-18,
  cached under `_cache/engulf_<TICKER>_1d.parquet`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../178-cci`](../178-cci) and [`../104-bollinger-reversion`](../104-bollinger-reversion) — the
  technical-indicator teardowns this study's data/strategy idiom copies (zone-entry detector,
  forward-return event study, HAC *t* vs a random-timing null).
- [`../363-pead-drift`](../363-pead-drift) — the gold-standard event study and the counter-example:
  a reversal/drift folk effect that *does* clear the bar (PEAD) versus engulfing, which doesn't.
- The **research-method demos** (data-mining-roulette, multiple-testing, look-ahead) — why a
  conditional "win rate" is not evidence and why the null must be a random-timing benchmark.
