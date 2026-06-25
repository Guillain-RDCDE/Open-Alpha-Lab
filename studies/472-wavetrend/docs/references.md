# References & literature map — Study 472 (WaveTrend, LazyBear)

## The claim under test

- **The folklore.** The *WaveTrend Oscillator* fires a **buy** when its main line **WT1**
  crosses **up** through its short signal line **WT2** while WT1 is **oversold** (deeply
  negative, below ≈ −60). The oscillator is "turning up from an extreme", so a bounce is
  "due". This is one of the most copied TradingView indicators, taught in countless YouTube
  and chart-site tutorials as a momentum-reversal entry.
- **The source.** The TradingView script **"WaveTrend Oscillator [WT]"** was published by the
  pseudonymous author **LazyBear** (2014) and is the canonical reference everyone forks. Its
  math is an explicit re-skin of an older idea: a *commodity-channel*-style normalisation
  ``ci = (tp − EMA(tp)) / (0.015 · meanAbsDev)`` — the ``0.015`` constant and the
  mean-absolute-deviation scaling come straight from **Donald Lambert's Commodity Channel
  Index** (CCI, *Commodities* magazine, 1980). WaveTrend doubles down by smoothing the channel
  index with a second EMA (the "wave") and adding an SMA signal line for the cross.
- **Variants.** Many forks change the lengths (n1/n2), the bands (±53/±60/±100), add a
  "VWAP"-style difference (WT1 − WT2) histogram, or combine WaveTrend with money-flow. All are
  affine/parametric variants of the same EMA-of-normalised-deviation geometry tested here.

## Why this is a "theory" / mechanical-proxy study

WaveTrend, as taught, is *semi-discretionary*: traders eyeball the cross and "confirm" it with
context. Following the desk's design for this kind, we encode the **tightest mechanical rule a
proponent would accept** and state the choices explicitly:

- **Causal lines.** ``esa = EMA(tp, n1)``, ``d = EMA(|tp − esa|, n1)``,
  ``ci = (tp − esa)/(0.015 d)``, ``WT1 = EMA(ci, n2)``, ``WT2 = SMA(WT1, signal)`` — all
  causal, the cross at bar *t* uses only closes ≤ *t*. No look-ahead.
- **Objective entry.** WT1 crosses above WT2 while WT1 was below the oversold band on the prior
  bar; no eyeballing, no "context".
- **The honest baseline.** On an upward-drifting index the only meaningful comparison is the
  **random-entry** control (same instrument, epoch and hold), because *any* dip-buy inherits the
  drift. We add a **scrambled-signal placebo** that permutes WT1's increments before
  re-cumulating — keeping the marginal (the oversold band still bites equally often) but
  destroying the wave structure — the direct test of "does the WaveTrend geometry matter?"

## Why a high one-sample t is not evidence (but here the *baseline* test passes)

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a long-only entry rule against **zero** measures that drift, not the rule. See Fama &
  French on the equity premium; the desk's standing rule is *signal-vs-baseline*, never
  *signal-vs-zero*. For WaveTrend the one-sample *t*'s are large (+3.4 to +4.7) — but unusually,
  the **cross-vs-random** Welch test *also* clears *t* ≥ 2, so the edge is not purely beta.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, JF) formalize testing chart patterns against a properly matched null; Sullivan,
  Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance, and the
  Bootstrap*, JF) and White (2000, *A Reality Check for Data Snooping*, Econometrica) show how
  trend-fitted rules manufacture significance unless raced against a fair benchmark. With only
  **115 trades** the WaveTrend result must be read against exactly this caution — the
  seed-sensitivity of the random baseline (Welch *t* spanning +1.2 to +3.4 across seeds) and the
  fact that the unconditional-drift test only clears 2 at 10–20 days is *why this lands Fragile,
  not Investable*.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the cross-vs-random difference.

## Method lineage (the desk's shared engine)

- **WaveTrend lines.** [`strategy.wavetrend`](../wavetrend/strategy.py) — causal EMAs/SMA of the
  HLC3 typical price, LazyBear's exact normalisation.
- **Oversold cross-up entry.** [`strategy.cross_up_entries`](../wavetrend/strategy.py) — the
  mechanical rule with the one-bar lag.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../wavetrend/strategy.py),
  [`strategy.hac_t`](../wavetrend/strategy.py), [`strategy.run_experiment`](../wavetrend/strategy.py).
- **Geometry placebo.** [`strategy.scrambled_signal_placebo`](../wavetrend/strategy.py) —
  permute WT1 increments, keep the marginal, re-run the same rule.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../wavetrend/data.py) plants a
  real WaveTrend bounce (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — the template; "price respects the
  channel" tested with the same random-entry + geometry-placebo idiom (there: None × Mirage).
- [`../178-cci`](../178-cci) — WaveTrend's direct ancestor (the CCI normalisation); a useful
  contrast for what the extra EMA smoothing buys.
- [`../108-stochastic`](../108-stochastic) and the broader oscillator zoo — most oversold-cross
  rules land None × Mirage. WaveTrend *looked* like an exception on a single random-baseline seed,
  but averaging the Welch *t* over 30 seeds leaves a real edge only in a thin 10–20-day band
  (Weak × Mixed) — a useful reminder that one lucky seed is not an edge (cf. Study 452).
- [`../452-spinning-top`](../452-spinning-top) — the cautionary twin: a naive seed=7 baseline gave
  a 20d Welch *t* > 3 that collapsed to ~+1.7 once averaged over seeds, correctly graded None.
  WaveTrend is the same trap caught one notch higher — it survives at 10–20d but not across the board.
