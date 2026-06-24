# References & literature map — Study 404 (Shooting Star)

## The claim under test

- **The folk recipe.** The shooting star is a one-bar bearish reversal: a small real body at
  the bottom of the session range, a long upper shadow (at least ~2× the body), little or no
  lower shadow, appearing **after an uptrend**. The narrative — buyers drove price up
  intraday and then surrendered the entire gain by the close, signalling exhaustion at the
  top — is taught in every candlestick primer. We steelman it as: *the conditional forward
  (short) return after a shooting star, net of costs, exceeds shorting a random day in the
  same name.*
- **Steve Nison** popularised Japanese candlesticks for Western markets in *Japanese
  Candlestick Charting Techniques* (1991) and *Beyond Candlesticks* (1994); the shooting
  star (and its bullish mirror the inverted hammer) is a staple of that taxonomy. The
  technique traces to **Munehisa Homma**'s 18th-century rice-trading methods.

## Why the steelman is *almost* coherent — and where it breaks

- **Intraday rejection is a real microstructure event.** A long upper wick genuinely
  encodes a within-session reversal: price was bid up and sold off. The leap of faith is
  that this *one-day* footprint forecasts the *next several days*. Caginalp & Laurent (1998),
  *"The Predictive Power of Price Patterns"* (Applied Mathematical Finance), found some
  candlestick configurations carried short-horizon information on S&P 500 stocks in an
  earlier era — but the effect is small and era-dependent.
- **The systematic evidence is largely negative.** Marshall, Young & Rose (2006),
  *"Candlestick Technical Trading Strategies: Can They Create Value for Investors?"*
  (Journal of Banking & Finance), test the full candlestick zoo on Dow stocks and find
  **no value** once you account for the data-snooping inherent in the menu of patterns.
  Horton (2009), *"Stars, crows, and doji,"* reaches similar conclusions. Our result —
  a *significant wrong-sign* short return on pooled geometry — is consistent with this body
  of work and with short-horizon momentum dominating single-bar reversal on large-caps.
- **Multiple testing inside one study.** With 26 names and four horizons, a lone *t* > 2
  (PG here) is expected by chance. Sullivan, Timmermann & White (1999), *"Data-Snooping,
  Technical Trading Rule Performance, and the Bootstrap"* (Journal of Finance), is the
  canonical warning; our per-name table makes the snooping visible rather than cherry-picking
  the winner.

## The failure mode exposed

- **An upper wick is often a momentum spike, not a top.** On a trending large-cap, a long
  upper shadow frequently marks an intraday reaction to news that the stock *grows into*
  over the following days — continuation, not exhaustion. Fama (1970) weak-form efficiency
  predicts exactly that a visible one-bar shape carries no exploitable forecast for liquid
  names; Lo, Mamaysky & Wang (2000), *"Foundations of Technical Analysis"* (Journal of
  Finance), find chart patterns carry *some* statistical information but rarely survive as a
  tradable edge net of costs.
- **The short side is the expensive side.** Even a coin-flip gross edge dies once you pay
  the spread twice and borrow per day. D'Avolio (2002), *"The Market for Borrowing Stock"*
  (Journal of Financial Economics), documents how borrow costs erode short strategies.

## Method lineage (the desk's shared engine)

- **HAC / Newey–West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  used for the overlapping-window inference in [`strategy.hac_t`](../shooting_star/strategy.py).
- **Label-shuffle / permutation placebo.** The per-name shuffle in
  [`strategy.placebo_pvalue`](../shooting_star/strategy.py) is a randomisation test in the
  spirit of Brock, Lakonishok & LeBaron (1992), *"Simple Technical Trading Rules and the
  Stochastic Properties of Stock Returns"* (Journal of Finance).
- **Reproducibility stamp.** As-of freeze + content fingerprint, mirroring the desk
  convention in `quantlab/repro.py`.

## Data sources used here

- **Yahoo! Finance daily OHLC** (via `yfinance`, un-adjusted), full history across 26 US
  large-caps + SPY. The offline reproducible core and the positive control run on the
  deterministic [`data.synthetic_panel`](../shooting_star/data.py) generator, never the
  network. Each headline is pinned with an as-of date and a basket content fingerprint
  (see [`docs/results.md`](results.md)).

## Related desk studies

- **[Study 403 — Hammer & Hanging Man](../../403-hammer-hanging-man/)**: the bullish mirror —
  same one-bar geometry flipped, same machinery, also no edge. The direct sibling.
- **[Study 402 — Engulfing Pattern](../../402-engulfing-pattern/)**: the two-bar candlestick
  reversal, same honest treatment.
- **[Study 178 — CCI](../../178-cci/)**: a normalised oscillator's overbought/oversold rule —
  another "does a textbook signal beat a coin?" teardown that lands NONE × MIRAGE.
- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)**: band mean-reversion,
  the same reversal-vs-momentum question on liquid daily equity tapes.
