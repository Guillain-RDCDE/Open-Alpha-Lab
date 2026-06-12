# References & literature map — Study 78 (Crossed-Wires)

## The claim under test

- **The folk recipe.** A perennial favourite in trading books, YouTube tutorials, and algorithmic
  trading courses: *"Watch the MACD(12,26,9). When the MACD line crosses above its signal line,
  go long; when it crosses below, go short or flat. The indicator summarises the medium-term trend —
  it's a lagging entry but a high-probability direction call."* There is no single canonical paper;
  the indicator was created by Gerald Appel in the late 1970s and has been taught ever since.
  We steelman it as: *the MACD(12,26,9) signal-line crossover direction carries enough medium-term
  momentum information to beat a random-direction entry, net of costs, on daily bars.*

## Why the steelman is coherent — the real effect MACD leans on

- **Trend-following and time-series momentum.** Moskowitz, Ooi & Pedersen (2012), *Time-Series
  Momentum* (Journal of Financial Economics), document robust return continuation at the monthly
  horizon for a broad asset universe. The MACD is effectively a lagged measure of medium-term
  return history; signal-line crosses are its way of filtering noise from that signal. Our
  synthetic control confirms the MACD does harvest persistence when it exists — the question is
  whether the real daily tape has enough at the 12/26-bar horizon.
- **EMA-based indicators as trend proxies.** The MACD(12,26,9) is algebraically equivalent to
  a lagged difference of two EMAs multiplied by a smoothing factor — a close cousin of the
  moving-average crossovers tested in Study 21 (Fools-Gold, 50/200 golden cross) and Study 72
  (Loaded-Dice, SMA(5/10) scalp). All three are regime detectors, not forecasters.
- **Prior empirical evidence: mixed and shrinking.** Brock, Lakonishok & LeBaron (1992), *Simple
  Technical Trading Rules* (Journal of Finance), reported MA rules with predictive power in the
  Dow Jones history, but this covered long pre-1990 periods with limited competition. Park &
  Irwin (2007), *What Do We Know About the Profitability of Technical Analysis?* (Journal of
  Economic Surveys), document that most of the reported advantage evaporates out of sample,
  after data-snooping correction, and after realistic costs. Our HAC t = +0.94 is consistent
  with their "marginal at best, mostly noise" conclusion.

## The two traps this study is really about

- **Low trade count and inference power.** With ~97 trades per instrument over five years, the
  standard error on the per-trade mean is large. Even a genuine edge of 10–15 bps/trade would
  need ~200–300 trades to clear a HAC t of 2 — MACD's low turnover (~19 triggers/yr) means the
  daily tape is a low-power telescope. Compare to Study 72's 3,801 pooled trades at the 5-minute
  horizon: more statistical power, same null result. Our t = +0.94 is consistent with both "no
  edge" and "small edge, underpowered test" — the honest verdict is WEAK.
- **The fixed-tick trap.** As in Study 72, a small take-profit with a far stop manufactures a
  95%+ win-rate at negative expectancy (mean −26 bps, skew −7). Appel's original framing
  ("wait for the histogram to turn, ride the trend") is essentially a fixed-tick recipe — see
  Taleb (2004), *Fooled by Randomness*, and Feller's gambler's-ruin literature for the
  mathematical framing. We expose it mechanically here.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../crossed_wires/strategy.py).
- **Average true range.** Wilder (1978), *New Concepts in Technical Trading Systems* — the
  risk unit R for the symmetric barriers, [`strategy.atr`](../crossed_wires/strategy.py).
- **MACD indicator.** Appel (1979), *The Moving Average Convergence/Divergence Method* — the
  indicator definition; EMA(fast=12) − EMA(slow=26), smoothed by EMA(9).
- **Reproducibility stamp.** Content fingerprint + as-of date in [`docs/results.md`](results.md).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), `period="5y"` (approximately 2021-06 to
  2026-06) across six liquid tapes (SPY, QQQ, IWM, AAPL, TSLA, NVDA). Daily history gives
  ~1,260 bars per ticker — enough warmup for MACD's 35-bar minimum, but still only ~97
  triggers per ticker over five years. The as-of date and per-tape content fingerprint are
  recorded in [`docs/results.md`](results.md).

## Related desk studies

- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: SMA(5/10) on 5-minute bars —
  the directly comparable EMA-crossover cousin. HAC t = −1.12 (firmly null), much
  higher-powered test (3,801 trades), same verdict. MACD on daily is *even lower-powered*.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the daily 50/200 golden cross — another
  MA crossover family member, also NONE/MIRAGE.
- **[Study 08 — True-Strength](../../08-true-strength/)**: the TSI indicator on daily bars —
  the momentum indicator desk companion (RSI family), same regime.
- **[Study 15 — Rubber-Band](../../15-rubber-band/)**: RSI mean-reversion entry — the
  counter-signal to trend-following; also in the indicator zoo.
- **[Study 67 — Fed-Drift](../../67-fed-drift/)**: a post-event drift study with a known
  catalyst — what a *real* signal (HAC t >> 2) looks like relative to MACD's t = +0.94.
