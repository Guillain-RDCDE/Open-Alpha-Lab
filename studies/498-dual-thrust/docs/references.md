# References & literature map — Study 498 (Dual Thrust)

## The claim under test

- **The folklore.** From a trailing-`N`-day span form a *Range* = `max(HH−LC, HC−LL)` (highest
  high, lowest low, highest close, lowest close), then draw two trigger bands around today's
  open: `buy_line = open + k1·Range` and `sell_line = open − k2·Range`. Go long on a break above
  the buy line, short on a break below the sell line. The pitch — repeated on every algo-trading
  blog and bundled with most backtesting frameworks — is that a clean break of the opening-range
  band "catches the day's trend" and is a high-probability momentum entry.
- **The source.** **Michael Chalek** (a commodity trader and TradeStation-era system developer)
  created **Dual Thrust** in the 1980s; it circulated through the Futures Truth / Robbins World
  Cup trading-system culture and became a staple example in the open-source quant world (it is a
  built-in demo strategy in many Chinese and Western backtesting platforms — e.g. vn.py,
  zipline-style tutorials, and countless Quant blogs). It is a member of the **opening-range
  breakout (ORB)** family alongside Toby Crabel's work and the classic Donchian/turtle breakout.
- **Lineage.** Toby Crabel, *Day Trading with Short Term Price Patterns and Opening Range
  Breakout* (1990), is the canonical academic-adjacent treatment of the opening-range idea;
  Chalek's Dual Thrust is the asymmetric, range-scaled cousin. The Donchian channel breakout
  (Richard Donchian) and the Turtle system (Dennis/Eckhardt) are the trend-following relatives;
  see desk study [`../../088-turtle`](../../088-turtle) and [`../../439-donchian`](../../439-donchian).

## Why this is a "theory" / mechanical-proxy study

Dual Thrust has free parameters (`N`, `k1`, `k2`) that practitioners optimize per market.
Following the desk's design for this kind, we encode the **tightest mechanical rule a proponent
would accept** — Chalek's classic `N = 5`, symmetric `k1 = k2 = 0.5` — and test the **long**
side (the headline, and the only direction that survives on an upward-drifting tape):

- **No look-ahead.** The Range uses the **prior** `N` bars (HH/LL/HC/LC shifted by one), so the
  bands are known at today's open; the breakout is read on the close of `t`; the trade is
  entered at the close of `t+1` (one documented lag).
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* long entry inherits
  the drift. We add a **scrambled-Range placebo** that permutes which day each Range belongs to,
  destroying the opening-range geometry while keeping the Range marginal and the `k` coefficients
  — the direct test of "does the breakout geometry matter?"

Parameter optimization (per-market `N`, `k1`, `k2`) only adds *hindsight* (free parameters),
which can inflate in-sample fit; the fixed-parameter version here is the charitable **upper
bound** on the method's daily-bar performance.

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a long-only entry rule against **zero** measures that drift, not the rule. Worse, a
  breakout rule *enters after price has already risen*, so it can systematically time the drift
  badly — which is exactly what the negative breakout-minus-random delta shows here. See Fama &
  French on the equity premium; the desk's standing rule is *signal-vs-baseline*, never
  *signal-vs-zero*.
- **Data snooping on trading rules.** Sullivan, Timmermann & White (1999, *Data-Snooping,
  Technical Trading Rule Performance, and the Bootstrap*, Journal of Finance) and White (2000,
  *A Reality Check for Data Snooping*, Econometrica) show how parameter-fitted trading rules
  manufacture significance unless raced against a fair benchmark. Lo, Mamaysky & Wang (2000,
  *Foundations of Technical Analysis*, JF) formalize testing chart/technical rules against a
  properly matched null.
- **Breakout vs reversion.** Jegadeesh & Titman (1993) document momentum at 3–12 months but
  Jegadeesh (1990) and Lehmann (1990) document short-horizon *reversal* at the weekly/daily
  scale — so a daily-bar breakout entry is fighting the empirical short-horizon mean-reversion,
  consistent with the negative delta found here.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the breakout-vs-random difference.

## Method lineage (the desk's shared engine)

- **Dual-Thrust Range + trigger bands.** [`strategy.dual_thrust_lines`](../dual_thrust/strategy.py),
  [`strategy.breakout_entries`](../dual_thrust/strategy.py) — the mechanical geometry with the
  trailing-Range (no look-ahead) baked in.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../dual_thrust/strategy.py),
  [`strategy.hac_t`](../dual_thrust/strategy.py), [`strategy.run_experiment`](../dual_thrust/strategy.py).
- **Geometry placebo.** [`strategy.scrambled_range_placebo`](../dual_thrust/strategy.py) —
  permute the Range across dates, keep the marginal and the `k` coefficients.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../dual_thrust/data.py) plants a
  real breakout-continuation (knob `edge`); with `edge = 0` the rule must NOT beat random — the
  offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) OHLC for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../088-turtle`](../../088-turtle) and [`../../439-donchian`](../../439-donchian) — the
  channel/breakout family; Dual Thrust is the open-scaled, asymmetric ORB cousin.
- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the gold-standard template for
  this study: same random-entry baseline + geometry placebo + synthetic-control idiom.
- The broader technical-indicator zoo (CCI, Bollinger, Supertrend…) — most land None × Mirage
  for the same reason: a rule fitted to past price re-describes (or, like Dual Thrust, *mis-times*)
  the trend. The research-method demos (data-mining-roulette, look-ahead, curve-fitting) frame
  why a signal-vs-zero *t* is not enough.
