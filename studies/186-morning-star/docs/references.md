# References & literature map — Study 186 (Morning-Star)

## The claim under test

- **The folk recipe.** The morning-star and evening-star are three-candle reversal
  patterns from Japanese candlestick charting, popularised in the West by Steve
  Nison (1991), *Japanese Candlestick Charting Techniques* (New York Institute of
  Finance). The morning-star signals the end of a bearish run: a large bearish
  candle (day 1), a small "star" body that gaps below the first candle's body
  (day 2, indecision), then a large bullish candle that closes well into the first
  candle's body (day 3). The evening-star is the mirror image, signalling the end
  of a bullish run. Both patterns claim to mark multi-day turning points visible in
  price action alone, without reference to fundamentals or volume.

  We steelman this as: *the pattern's 1-day and 5-day forward returns (in the
  claimed direction) exceed the unconditional forward returns of random days
  drawn from the same tape with the same direction*, measured pooled across
  15 US equities from 2010 to 2026.

## Empirical evidence — what the literature actually says

- **Nison, S. (1991)**, *Japanese Candlestick Charting Techniques*, New York
  Institute of Finance. The canonical Western source for the pattern definitions.
  No statistical test of excess returns is offered; the argument is qualitative.
- **Marshall, Young & Rose (2006)**, *Candlestick Technical Trading Strategies:
  Can They Create Value for Investors?*, Journal of Banking & Finance 30(8):
  2303–2323. Tests 35 candlestick patterns on DJIA components 1992–2002, including
  morning/evening star variants; after White's Reality Check, no pattern generates
  statistically significant abnormal returns. Closest precursor to our methodology.
- **Caginalp & Laurent (1998)**, *The Predictive Power of Price Patterns*,
  Applied Mathematical Finance 5(3-4): 181–205. Reports predictive value for a
  small S&P 500 sample but lacks a multiple-comparisons adjustment and a
  random-day baseline.
- **Horton (2009)**, *Stars, Crows, and Doji: The Use of Candlesticks in Stock
  Selection*, Quarterly Review of Economics and Finance 49(2): 283–294. Mixed
  evidence; the morning-star in particular shows negligible excess in the 1987–2006
  period on DJIA stocks after controlling for prior-period return.
- **Lu, Shiu & Liu (2012)**, *Profitable Candlestick Trading Strategies — The
  Evidence from a New Perspective*, Review of Financial Economics 21(2): 63–68.
  Finds marginal evidence on Taiwan market data; results are inconsistent on US
  large-cap equities.
- **Fock, Klein & Zwergel (2005)**, *Performance of Candlestick Analysis on
  Intraday Futures Data*, Journal of Derivatives 13(1): 28–40. Tests on futures
  intraday data; no persistent edge survives transaction costs.

## Why the finding (negative excess) is coherent

The morning-star triggers on a three-day sequence that ends with a strong bullish
day. The random-day control drawn from the *same* volatile down-window already
captures the mean-reversion bounce expected in that regime. The pattern's defining
third candle has consumed part of the reversion that a random day from that window
would measure, so the *next-day* forward return conditional on the pattern is lower
than the unconditional next-day return from a random down-move window. This is a
sequencing/timing issue, not a genuine bearish signal.

Key related evidence:
- **Jegadeesh (1990)**, *Evidence of Predictable Behavior of Security Returns*,
  Journal of Finance 45(3): 881–898. Short-term mean reversion at weekly horizons
  provides the structural backdrop that the morning-star is supposed to trade.
- **De Bondt & Thaler (1985)**, *Does the Stock Market Overreact?*, Journal of
  Finance 40(3): 793–805. The mean-reversion phenomenon that a three-candle
  reversal is designed to exploit; the issue is that the pattern fires too late
  in the reversion cycle.

## The multiple-comparisons framework

- **White (2000)**, *A Reality Check for Data Snooping*, Econometrica 68(5):
  1097–1126. Marshall et al. (2006) apply this; we use Bonferroni (4 tests:
  2 patterns × 2 horizons) as a simpler, conservative adjustment.
- **Harvey, Liu & Zhu (2016)**, *... and the Cross-Section of Expected Returns*,
  Review of Financial Studies 29(1): 5–68. The effective t-stat threshold for
  a new factor is 3.0 given literature-wide snooping; our Bonferroni threshold
  of 2.50 is already a study-internal bar. The morning-star clears it but in the
  wrong direction.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix*, Econometrica 55(3): 703–708 — implemented in
  [`strategy.summarize_pattern`](../morning_star/strategy.py).
- **Random-day baseline control.** The forward return on a random draw of days
  with the same directional assignment isolates what the pattern adds over the
  unconditional drift. Cf. the random-direction control in
  [Study 72 — Loaded-Dice](../../72-loaded-dice/) and the matched base-rate control
  in [Study 76 — Rice-Paper](../../76-rice-paper/) (the sibling single-candle study).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), 15 US equities (SPY + 14
  S&P 500 constituents). Daily OHLCV from 2010-01-04 to 2026-06-15, ~4,137
  trading days per ticker. Each run is pinned with a per-ticker content
  fingerprint (see [`docs/results.md`](results.md)). The offline reproducible
  core and tests run on the deterministic [`data.synthetic_daily`](../morning_star/data.py)
  generator.

## Related desk studies

- **[Study 76 — Rice-Paper](../../76-rice-paper/)**: the single-candle family
  (bullish/bearish engulfing, hammer, shooting star, doji) on the same 15-ticker
  daily panel with the same random-day baseline. Also Signal=NONE/Tradability=MIRAGE.
- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the SMA(5/10) intraday
  crossover — same "technical signal vs a coin" methodology on 5-minute bars.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the daily 50/200 golden
  cross — another technical-analysis rule, same family, same honest test.
- **[Study 48 — Groundhog](../../48-groundhog/)**: calendar-pattern vs
  base-rate control — same multiple-comparisons discipline applied to seasonal
  effects.
