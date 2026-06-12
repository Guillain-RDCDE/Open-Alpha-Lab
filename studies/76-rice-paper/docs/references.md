# References & literature map — Study 76 (Rice-Paper)

## The claim under test

- **The folk recipe.** Japanese candlestick charting is credited to Munehisa
  Homma, an 18th-century rice trader in Sakata, Japan, who reportedly used
  visual bar-pattern analysis to trade the Dōjima Rice Exchange. The modern
  western popularisation is Steve Nison (1991), *Japanese Candlestick Charting
  Techniques* (New York Institute of Finance) — the canonical textbook. The
  claim: patterns such as the bullish/bearish engulfing, the hammer, the
  shooting star, and the doji signal short-term price reversals, giving a
  discretionary or systematic trader a statistical edge on the *next few days*
  of returns. We steelman it as: *the pattern's forward return (1–5 days)
  exceeds the unconditional forward return on random days with the same
  directional assignment*, measured across a large cross-section of US equities.

## Empirical evidence — what the literature actually says

- **Brock, Lakonishok & LeBaron (1992)**, *Simple Technical Trading Rules and
  the Stochastic Properties of Stock Returns*, Journal of Finance 47(5):
  1731–1764. Classic paper showing MA and range-breakout rules with positive
  returns on the DJIA, 1897–1986 — but on daily close data, no candlestick
  patterns, and pre-transaction-cost.
- **Caginalp & Laurent (1998)**, *The Predictive Power of Price Patterns*,
  Applied Mathematical Finance 5(3–4): 181–205. One of the earliest systematic
  candlestick studies; reports predictive value in a small S&P 500 sample but
  lacks a multiple-comparisons adjustment and uses fixed-dollar exits.
- **Marshall, Young & Rose (2006)**, *Candlestick Technical Trading Strategies:
  Can They Create Value for Investors?*, Journal of Banking & Finance 30(8):
  2303–2323. Applies 35 candlestick patterns to DJIA components 1992–2002;
  after data-snooping bootstrap correction (White's Reality Check), **no
  pattern** generates statistically significant abnormal returns. The most
  rigorous pre-2010 study and the closest precursor to our methodology.
- **Lu, Shiu & Liu (2012)**, *Profitable Candlestick Trading Strategies — The
  Evidence from a New Perspective*, Review of Financial Economics 21(2): 63–68.
  Find marginal evidence on Taiwan stock market data; results do not replicate
  on US large-caps in later work.
- **Horton (2009)**, *Stars, Crows, and Doji: The Use of Candlesticks in Stock
  Selection*, Quarterly Review of Economics and Finance 49(2): 283–294. Mixed
  evidence; warns that pattern-specific studies without correction for the
  number of patterns tested will inflate false-positive rates.

## Why the steelman is *almost* coherent — the real effects it leans on

- **Short-term mean reversion.** Jegadeesh (1990), *Evidence of Predictable
  Behavior of Security Returns*, Journal of Finance 45(3): 881–898; and De
  Bondt & Thaler (1985), *Does the Stock Market Overreact?*, Journal of
  Finance 40(3): 793–805. If daily returns mean-revert, then a "bearish
  engulfing" after a down-day and an "up" close could capture a real
  continuation/reversal structure. Our synthetic positive control confirms the
  pattern detectors *do* harvest mean-reversion when it is planted — the real
  tape just does not carry it robustly enough.
- **Microstructure and bid-ask bounce.** Roll (1984), *A Simple Implicit
  Measure of the Effective Bid-Ask Spread*, Journal of Finance 39(4):
  1127–1139. Short-horizon negative autocorrelation in daily returns partly
  reflects the bid-ask bounce; patterns that fire on large-body days (engulfing)
  may correlate with this — but at daily frequency the effect is too small
  to clear transaction costs.

## The multiple-comparisons problem

- **White (2000)**, *A Reality Check for Data Snooping*, Econometrica 68(5):
  1097–1126. The standard framework for testing whether the best of many
  rules is real. Marshall et al. (2006) apply this explicitly to candlesticks;
  we use Bonferroni as a simpler, more conservative adjustment (12 tests:
  6 patterns × 2 horizons).
- **Harvey, Liu & Zhu (2016)**, *… and the Cross-Section of Expected Returns*,
  Review of Financial Studies 29(1): 5–68. Argues the effective t-stat threshold
  for a new factor should be 3.0, given the number of hypotheses tested in the
  literature; we use 2.64 (Bonferroni/12) as the study-internal bar and note
  that even the highest observed t-stat (2.16) falls below both.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix*, Econometrica 55(3): 703–708 — [`strategy.summarize_pattern`](../rice_paper/strategy.py).
- **Random-day baseline control.** This study's honest comparison: the forward
  return on a random draw of days with the same directional assignment isolates
  what the pattern adds over the unconditional drift. Cf. the random-direction
  control in [Study 72 — Loaded-Dice](../../72-loaded-dice/) and the matched
  base-rate control in [Study 48 — Groundhog](../../48-groundhog/).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), 15 US equities (SPY + 14
  S&P 500 constituents). Daily OHLCV from 2010-01-04 to 2026-06-12, ~4,136
  trading days per ticker. Each run is pinned with a per-ticker content
  fingerprint (see [`docs/results.md`](results.md)). The offline reproducible
  core and tests run on the deterministic [`data.synthetic_daily`](../rice_paper/data.py)
  generator.

## Related desk studies

- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the SMA(5/10) intraday
  crossover — same "technical signal vs a coin" methodology but on 5-minute
  bars instead of daily candlestick shapes.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the daily 50/200 golden
  cross — another technical-analysis rule, same family, same honest test.
- **[Study 48 — Groundhog](../../48-groundhog/)**: calendar-pattern vs
  base-rate control — same multiple-comparisons discipline applied to seasonal
  effects.
- **[Study 32 — Rip-Tide](../../32-rip-tide/)**: explicit mean-reversion
  strategy — if anything, mean-reversion at the daily horizon is where a real
  but small effect lives; candlestick patterns are a noisier, less systematic
  way to target the same mechanic.
