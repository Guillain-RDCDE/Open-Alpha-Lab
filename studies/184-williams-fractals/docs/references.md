# References & literature map — Study 184 (Williams-Fractals)

## The claim under test

- **Bill Williams' Fractal Indicator.** Introduced in Williams, B. (1995), *Trading Chaos:
  Applying Expert Techniques to Maximize Your Profits* (Wiley), and refined in Williams, B.
  (1998), *New Trading Dimensions: How to Profit from Chaos in Stocks, Commodities and
  Currencies* (Wiley).  The pattern: a bar whose high (bearish fractal) or low (bullish
  fractal) is strictly beyond the two bars on each side marks a local 5-bar swing
  extreme.  Williams used fractals as a core component of his "Alligator" trading system
  alongside the Alligator indicator (three displaced SMAs).  The folk recipe most commonly
  cited in retail literature: *"a bearish fractal marks the top of a swing; once a bullish
  fractal is broken to the upside, enter long."*  We steelman both the reversal and
  breakout readings and test each against a random-direction control on the same bars.

## Why the pattern is *almost* coherent — the real effect it leans on

- **Local extremes and the reflection principle.** A 5-bar swing high is by construction a
  local maximum of the high series.  The reflection principle of Brownian motion
  (Billingsley, P. (1995), *Probability and Measure*, 3rd ed., Wiley) implies that even in
  a random walk, local maxima are followed by mean reversion on average over the very next
  few bars — but this effect is concentrated in the first bar and decays rapidly.  By
  requiring two confirmation bars (t+1, t+2 must both be *below* the fractal bar), the
  Williams pattern consumes exactly the bars over which the reversion would appear, leaving
  the subsequent trade in a regime closer to a martingale.  Our per-hold-period sweep
  confirms this: the 3-day hold (the shortest that still captures the confirmation lag)
  gives the most negative t-stat (−1.59), consistent with trading *after* the mean
  reversion has already occurred.

- **Breakout / support-resistance.** The breakout framing assumes fractal highs/lows act
  as price memory — once a prior swing high is exceeded, price continues.  This is the
  logic behind classic breakout and channel strategies (Donchian channels, Turtle rules).
  Irwin & Park (2007), *What Do We Know About the Profitability of Technical Analysis?*
  (Journal of Economic Surveys), survey the breakout literature and note that
  out-of-sample performance is thin and cost-sensitive.  Our result (breakout t = −0.01)
  is consistent: the fractal level carries no special information beyond its definition as a
  local extreme.

- **Microstructure noise and look-ahead in fractal studies.** Many papers reporting fractal
  efficacy use lookahead (treating the pattern as known at bar t instead of t+2) or fail to
  account for the two-bar confirmation lag.  Aronson, D. (2006), *Evidence-Based Technical
  Analysis* (Wiley), documents how unreported look-ahead inflates apparent win-rates for
  pattern-based rules.  We are explicit: the fractal is confirmed at the close of bar t+2,
  and the trade opens at the open of bar t+3.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey, W. & West, K. (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*
  (Econometrica, 55(3), 703–708) — per-trade return series may be autocorrelated (overlapping
  holds), so the robust t-stat is the correct inference measure;
  [`strategy.summarize`](../williams_fractals/strategy.py) implements it.
- **Random-direction control.** The "same entries, random direction" baseline is the
  desk's standard for direction-sensitive pattern tests — it measures whether the signal
  adds directional information over a fair coin.  Identical to the controls in
  [Study 72 (Loaded-Dice)](../../72-loaded-dice/) and
  [Study 127 (Williams-R)](../../127-williams-r/).
- **Forward-return backtest.** Fixed-horizon close-to-close returns with a single
  round-trip cost deducted.  No look-ahead: signal known at t, entry at t+1's open.
- **Synthetic positive control.** An AR(1) momentum tape confirms the breakout engine
  recovers an edge *when momentum exists*; the real-tape null is therefore a statement
  about the market, not a flaw in the machinery.

## Related desk studies

- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the SMA(5/10) 5-minute crossover
  scalp — the same "pattern-vs-coin" family, intraday fidelity.
- **[Study 127 — Williams-R](../../127-williams-r/)**: Larry Williams' %R oscillator —
  the other Williams indicator, mean-reversion reading, same daily frequency.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 golden cross — the
  moving-average breakout family Williams' breakout framing belongs to.
- **[Study 76 — Rice-Paper](../../76-rice-paper/)**: candlestick patterns on daily bars —
  the same "chart pattern vs coin" framework applied to Japanese candles.
- **[Study 17 — Glass-Ceiling](../../17-glass-ceiling/)**: breakout from resistance —
  the conceptual predecessor to fractal breakout trading.

## Data sources

- **Yahoo Finance daily bars** (via `yfinance`), adjusted closes, 10-year window
  (2016-06-15 → 2026-06-15).  Six tickers: SPY, QQQ, IWM, AAPL, MSFT, GLD.  All results
  pinned with per-tape content fingerprints; see [`docs/results.md`](results.md).  The
  offline test-suite and reproducible core run on the deterministic
  [`data.synthetic_daily`](../williams_fractals/data.py) generator.
