# References & literature map — Study 470 (Stochastic Momentum Index)

## The claim under test

- **The folklore.** The **Stochastic Momentum Index** (SMI) "times turns": because it measures the
  close's distance from the *midpoint* of the recent high/low range (not just its position above the
  low, as Lane's classic stochastic does), it is said to lead price — when the SMI stops falling and
  **rises up out of oversold** (conventionally below −40), a bottom is forming and you buy; the
  symmetric overbought turn is a sell. This is repeated on Investopedia, StockCharts ChartSchool,
  TradingView and every indicator catalogue.
- **The source.** **William Blau** introduced the SMI in *"Stochastic Momentum"*, *Technical
  Analysis of Stocks & Commodities*, January 1993, and developed it in his book *Momentum,
  Direction, and Divergence* (Wiley, 1995). The SMI is a refinement of **George C. Lane's**
  stochastic oscillator (popularised in the late 1950s–1980s): Lane's %K = 100·(C−LL)/(HH−LL); Blau
  replaces the low with the range *midpoint* M = (HH+LL)/2 and **double-smooths** both the numerator
  (C−M) and the denominator (range/2) with two EMAs, yielding a ±100 oscillator that is far less
  jagged than raw %K. Classic parameters: range N = 13, smoothing s1 = 25, s2 = 2.
- **Lineage.** The SMI sits in the **double-smoothed oscillator** family Blau pioneered alongside
  his TSI (True Strength Index) and the double-smoothed stochastic. All share the same engine: an
  EMA-of-an-EMA applied to a raw momentum/range quantity. This is why the parameter-scramble placebo
  in this study probes the *family*, not just one tuning.

## Why a signal-vs-zero *t* is not evidence (and what we used instead)

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t* of
  a long-only entry rule against **zero** measures that drift, not the rule. See Fama & French on the
  equity premium. The desk's standing rule is *signal-vs-baseline*, never *signal-vs-zero* — here the
  baseline is a **large** random-entry draw (1000 dates/ticker) so it is a *stable* estimate of the
  tape's true drift, essential because the SMI turns are sparse (only 226 in 21 years).
- **Data snooping on chart tools.** **Lo, Mamaysky & Wang (2000)**, *Foundations of Technical
  Analysis*, *Journal of Finance* 55(4), formalize testing technical patterns against a properly
  matched null distribution. **Sullivan, Timmermann & White (1999)**, *Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap*, *Journal of Finance* 54(5), and **White (2000)**,
  *A Reality Check for Data Snooping*, *Econometrica* 68(5), show how rules selected/tuned on the
  same data manufacture significance unless raced against a fair benchmark. The SMI's classic (13,
  25, 2) is itself a tuned triple — which is exactly why the **parameter-scramble placebo** matters:
  it asks whether the *specific* tuning is load-bearing or whether any oversold-dip oscillator works.
- **Mean reversion is a real, documented short-horizon effect.** That the SMI-turn *does* beat the
  drift baseline at 5–20 days is consistent with the well-documented short-horizon reversal /
  oversold-bounce literature (e.g. Jegadeesh 1990, *Evidence of Predictable Behavior of Security
  Returns*, JF; Lehmann 1990). The placebo's finding — *any* smoothed oversold oscillator captures
  it — says the credit belongs to that broad effect, not to Blau's particular indicator.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the turn-vs-random difference.

## Method lineage (the desk's shared engine)

- **Causal SMI + oversold-turn entries.** [`strategy.smi`](../stochastic_momentum_index/strategy.py),
  [`strategy.smi_turn_entries`](../stochastic_momentum_index/strategy.py) — the double-smoothed ±100
  oscillator and the "rising out of oversold" trigger, all causal (no future bars).
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../stochastic_momentum_index/strategy.py),
  [`strategy.hac_t`](../stochastic_momentum_index/strategy.py),
  [`strategy.run_experiment`](../stochastic_momentum_index/strategy.py).
- **Parameter placebo.** [`strategy.scrambled_param_placebo`](../stochastic_momentum_index/strategy.py)
  — recompute the SMI with perturbed (N, s1, s2), keep the same tape and family.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../stochastic_momentum_index/data.py)
  plants a real oversold bounce keyed off the same causal SMI (knob `edge`); with `edge = 0` the
  detector must NOT manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../180-stochastic`](../../180-stochastic) and [`../../182-williams-r`](../../182-williams-r) —
  the classic stochastic / %R oscillators the SMI refines; same oversold-buy idiom.
- [`../../104-bollinger-reversion`](../../104-bollinger-reversion) — the "band/channel reverts price"
  folklore tested with the random-entry baseline; another short-horizon mean-reversion probe.
- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the template study; lands None ×
  Mirage where the SMI lands Real × Fragile, a clean contrast between relabelled drift and a thin
  but genuine short-horizon reversion effect.
- The **research-method demos** (data-mining-roulette, multiple-testing, curve-fitting) frame why a
  signal-vs-zero *t* is not enough and why the parameter placebo is the load-bearing test here.
