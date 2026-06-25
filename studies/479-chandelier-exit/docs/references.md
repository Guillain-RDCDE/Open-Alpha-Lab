# References & literature map — Study 479 (Chandelier Exit)

## The claim under test

- **The folklore.** Hang a long-flat trailing stop a fixed multiple of volatility below the
  highest high since entry: `stop = HH(n) - m·ATR(n)`, canonically `n = 22`, `m = 3`. Because the
  stop scales with the Average True Range, it "breathes" with the market — staying loose in
  trends (letting winners run) and tightening in calm (cutting losers). The pitch is that the
  ATR-managed long **beats passive buy-and-hold** on a risk-adjusted basis.
- **The source.** **Chuck LeBeau** named and popularized the chandelier exit (it "hangs down from
  the ceiling" of the trade's high), taught through his System Traders Club and his book with
  **David Lucas**, *The Technical Traders Guide to Computer Analysis of the Futures Markets*
  (1992). The ATR building block is **J. Welles Wilder Jr.**, *New Concepts in Technical Trading
  Systems* (1978), which defines True Range and the Wilder-smoothed ATR. The modern restatements
  live in StockCharts' ChartSchool, Investopedia and TradingView's built-in "Chandelier Exit".
- **Variants.** The same trail appears as the **ATR trailing stop**, the **SuperTrend** band, and
  Kase/Keltner-style volatility stops — all affine in `HH ± k·ATR` and inheriting the same
  drift/beta confound tested here. (Desk sibling [`../../109-supertrend`] is the closest cousin.)

## Why this is a "mechanical rule" study

The chandelier is fully objective once `n` and `m` are fixed, so there is no discretion to
steelman — we simply encode the canonical 22/3 and test it honestly:

- **No look-ahead.** ATR and the running high use only bars up to and including *t*; the stop and
  the breakout are read on the close of *t*; any flip is executed at the **next** close (one
  documented lag).
- **The honest baselines.** (a) **Buy-and-hold** — the passive long the chandelier claims to beat
  (the thesis axis); (b) a **drift-matched random-entry** control (same instrument, epoch, hold),
  because *any* long-only rule on an upward-drifting index inherits the drift; (c) a
  **scrambled-ATR placebo** that permutes which ATR width sits on which bar, destroying the
  volatility geometry while keeping its marginal — the direct test of "is the ATR trail itself
  doing anything?".

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of a long-only entry rule against **zero** measures that drift, not the rule. The desk's
  standing rule is *signal-vs-baseline*, never *signal-vs-zero*. See Fama & French on the equity
  premium.
- **Data snooping on trading rules.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, *Journal of Finance*) formalize testing chart/technical rules against a matched null;
  Sullivan, Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance, and the
  Bootstrap*, *JF*) and White (2000, *A Reality Check for Data Snooping*, *Econometrica*) show how
  volatility-fitted stops manufacture apparent significance unless raced against a fair benchmark.
- **Trailing stops add no expected return.** A stop merely truncates the holding-period return
  distribution; for an i.i.d. (or martingale) price it cannot raise expected return — it only
  lowers exposure and thus both variance and mean. Kaminski & Lo (2014, *When Do Stop-Loss Rules
  Stop Losses?*, *Journal of Financial Markets*) show stops help only when returns are
  *momentum-driven*, the exact condition our synthetic control plants.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the entry-vs-random difference.

## Method lineage (the desk's shared engine)

- **Wilder ATR + chandelier trail.** [`strategy.atr`](../chandelier_exit/strategy.py),
  [`strategy.chandelier_position`](../chandelier_exit/strategy.py) — the mechanical long-flat
  state with the one-bar trading lag baked in.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../chandelier_exit/strategy.py),
  [`strategy.hac_t`](../chandelier_exit/strategy.py), [`strategy.run_experiment`](../chandelier_exit/strategy.py).
- **Equity-curve thesis axis.** [`strategy.strategy_equity`](../chandelier_exit/strategy.py) —
  chandelier-managed long vs buy-and-hold (CAGR / Sharpe / maxDD / time-in-market).
- **Geometry placebo.** [`strategy.scrambled_atr_placebo`](../chandelier_exit/strategy.py) —
  permute the ATR widths, keep positions and marginal.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../chandelier_exit/data.py)
  plants real momentum (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../109-supertrend`](../../109-supertrend) — the SuperTrend band is the same `HH ± k·ATR`
  trail; closest cousin.
- [`../../001-turtle`](../../001-turtle) and the broader trend/breakout family — Donchian
  breakouts share the chandelier's re-entry trigger; most beat random only at the shortest
  horizon, exactly as here.
- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — same random-entry + geometry-placebo
  idiom; the template this study is built from.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the chandelier is a clean live example of a breakout flicker
  plus beta masquerading as a "smart stop".
