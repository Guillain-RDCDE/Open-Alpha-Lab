# References & literature map — Study 487 (Elder's Triple Screen)

## The claim under test

- **The folklore.** Combine three filters across two timeframes to take a long only when "the
  tide, the wave and the ripple" agree: (1) a higher-timeframe **trend** must be up (the tide),
  (2) a lower-timeframe **oscillator** must be oversold against that trend (the wave — a pullback
  inside the up-move), and (3) a **breakout** must trigger the entry (the ripple — a trailing
  buy-stop above the prior bar's high). The pitch is that aligning timeframes filters out noise
  and yields a high-odds trade.
- **The source.** **Dr. Alexander Elder** introduced the *Triple Screen* trading system in a
  1986 *Futures* magazine article and popularised it in **_Trading for a Living_** (Wiley, 1993)
  and **_Come Into My Trading Room_** (Wiley, 2002). His own toolkit pairs a **weekly
  MACD-histogram** (Screen 1, trend), a daily oscillator such as the **Force Index** or
  **stochastic** (Screen 2, pullback), and a **trailing buy-stop** above the prior bar (Screen
  3, entry). Elder also originated the Force Index and the "Elder-ray" indicators used here.
- **Variants.** Practitioners swap the daily oscillator (Force Index ↔ stochastic ↔ Elder-ray),
  the timeframe pair (weekly/daily ↔ daily/hourly), and the trigger (breakout ↔ trendline
  break). All are **parameterisations of the same multi-timeframe-confluence idea** and inherit
  the same drift confound tested here.

## Why this is a "theory" / mechanical-proxy study

Elder's system is *semi-discretionary*: a trader eyeballs the weekly trend and times the daily
entry. Following the desk's design for this kind, we encode the **tightest mechanical rule a
proponent would accept** and state the irreducible choices explicitly:

- **Objective Screen 1.** Weekly MACD-histogram slope (`hist.diff() > 0`) computed on resampled
  weekly closes, forward-filled to days and **shifted one day** — a documented lag, no
  look-ahead.
- **Objective Screen 2.** A daily Force-Index proxy (EMA of the close-to-close change) below
  zero within the last 5 bars — the oversold pullback against the up-tide.
- **Objective Screen 3.** Close above the prior bar's high — the breakout trigger; entry at the
  next close (one lag).
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* dip-buy inherits the
  drift. We add a **screen-scramble placebo** that circularly shifts the weekly trend relative to
  price, destroying the timeframe alignment while keeping each screen's marginal — the direct
  test of "does the multi-timeframe filter matter?"

Discretionary timing adds *hindsight* (a free parameter), which can only inflate in-sample fit;
the mechanical version is therefore the charitable **upper bound** on the method.

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a long-only entry rule against **zero** measures that drift, not the rule. See Fama &
  French on the equity premium; the desk's standing rule is *excess-vs-excess* and
  *signal-vs-baseline*, never *signal-vs-zero*. Here the one-sample *t* hits +5.2 at 20 days yet
  the triple-vs-random Welch *t* is +0.44 — the gap is the beta.
- **Data snooping on trading rules.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, JF) formalize testing chart/indicator rules against a properly matched null;
  Sullivan, Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance, and
  the Bootstrap*, JF) and White (2000, *A Reality Check for Data Snooping*, Econometrica) show
  how rules tuned to past price manufacture significance unless raced against a fair benchmark.
  A three-filter system multiplies the implicit search space, making the random-entry baseline
  and the alignment placebo essential.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the triple-vs-random difference.

## Method lineage (the desk's shared engine)

- **Weekly trend + daily screens.** [`strategy.weekly_trend_up`](../elder_triple_screen/strategy.py),
  [`strategy.macd_hist`](../elder_triple_screen/strategy.py),
  [`strategy.force_index`](../elder_triple_screen/strategy.py),
  [`strategy.triple_screen_entries`](../elder_triple_screen/strategy.py) — the mechanical screens
  with the one-day weekly lag baked in.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../elder_triple_screen/strategy.py),
  [`strategy.hac_t`](../elder_triple_screen/strategy.py), [`strategy.run_experiment`](../elder_triple_screen/strategy.py).
- **Alignment placebo.** [`strategy.scrambled_screen_placebo`](../elder_triple_screen/strategy.py) —
  circularly shift the weekly trend, keep each screen's marginal.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../elder_triple_screen/data.py)
  plants a real post-alignment bounce (knob `edge`); with `edge = 0` the detector must NOT
  manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the same "geometry/structure
  forecasts" folklore tested with the random-entry baseline + a geometry-scramble placebo; the
  engine idiom here is its sibling.
- [`../../104-bollinger-reversion`](../../104-bollinger-reversion) — "the band reverts price"
  folklore tested with the random-entry baseline.
- The broader technical-indicator zoo (MACD, stochastic, ADX, Supertrend…) — most land
  None × Mirage for the same reason: an indicator fitted to past price re-describes the trend.
  Triple Screen stacks three such filters and still nets out as beta.
- The **research-method demos** (multiple-testing, data-mining-roulette, signal-stacking) frame
  why combining several weak filters does not manufacture an edge unless each is real and
  decorrelated — exactly what the alignment placebo here refutes.
