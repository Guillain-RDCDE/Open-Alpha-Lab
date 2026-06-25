# References & literature map — Study 493 (New-Highs-New-Lows breadth)

## The claim under test

- **The folklore.** Count the stocks at fresh 52-week **highs**, subtract those at fresh 52-week
  **lows**, and track the *net* — the **new-highs / new-lows (NH-NL) line**. "Breadth leads
  price": the line tops and bottoms *before* the index, so a surge of net new highs (a *breadth
  thrust*) confirms a rally, and a collapse (or a *divergence* — price up, breadth fading) warns
  of a top. A market-internals staple on every breadth dashboard.
- **The source.** The idea traces to **Charles H. Dow** and the Dow-Theory notion of
  *confirmation* by participation. It was formalised for stock selection and market timing by
  **William J. O'Neil** in the *Investor's Business Daily* methodology (*How to Make Money in
  Stocks*), and the NH-NL count is the backbone of the **Hindenburg Omen** (Jim Miekka) and is a
  close cousin of **Martin Zweig's breadth thrust** (the 10-day advance ratio, *Winning on Wall
  Street*). Modern write-ups (StockCharts ChartSchool, Investopedia, John Murphy's *Technical
  Analysis of the Financial Markets*) restate the rule.
- **Variants.** Advance/decline line, McClellan Oscillator/Summation, Zweig breadth thrust,
  up-volume/down-volume, % of stocks above their 200-day MA — all are **affine cousins** of the
  same "how broad is participation?" statistic and inherit the same drift confound tested here.

## Why this is a "breadth proxy" study

A *true* NH-NL reading counts thousands of exchange-listed issues, which we cannot fetch and
cache offline. Following the desk's design for breadth studies, we **proxy** the universe with a
small basket of liquid ETFs (SPY QQQ IWM DIA GLD) and state the limitation explicitly:

- **Objective extremes.** New high iff close equals the trailing 252-day maximum (look-back
  includes *t*; usable same day, no future data). The NH-NL fraction is smoothed over 10 days —
  the IBD-style line.
- **Objective thrust.** A long fires on the first up-cross of +0.20; no hand-picking, no
  divergence eyeballing.
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch, hold), because *any* long entry inherits the
  drift — and breadth is *mechanically* high after a rally. We add a **shuffled-membership
  placebo** that destroys the cross-sectional breadth structure while keeping each member's
  marginal new-high rate — the direct test of "does the aggregation matter?"

A 5-ETF basket is a **coarse upper bound** on what a real A/D feed could show; the result here is
lopsided enough that a richer universe would have to overturn a very clear finding.

## Why the high one-sample t is not evidence

- **Drift / beta + mechanical coupling.** US equity indices have a positive unconditional daily
  mean, and the NH-NL line is *definitionally* elevated after price has risen (members near new
  highs ⇔ price rose). A one-sample *t* of a long-only entry rule against **zero** therefore
  measures the drift and the lag, not a lead. The desk's standing rule is *signal-vs-baseline*,
  never *signal-vs-zero*; see Fama & French on the equity premium.
- **Data snooping on market-timing rules.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, JF) formalize testing technical signals against a properly matched null; Sullivan,
  Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance, and the
  Bootstrap*, JF) and White (2000, *A Reality Check for Data Snooping*, Econometrica) show how
  trend-coupled rules manufacture significance unless raced against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the thrust-vs-random difference.

## Method lineage (the desk's shared engine)

- **52-week extremes + NH-NL line.** [`strategy.net_new_high_line`](../new_highs_new_lows/strategy.py),
  [`strategy.breadth_thrust_entries`](../new_highs_new_lows/strategy.py) — the mechanical breadth
  construction, trailing-only.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../new_highs_new_lows/strategy.py),
  [`strategy.hac_t`](../new_highs_new_lows/strategy.py), [`strategy.run_experiment`](../new_highs_new_lows/strategy.py).
- **Structure placebo.** [`strategy.shuffled_membership_placebo`](../new_highs_new_lows/strategy.py)
  — permute each member's new-high series in time, keep marginals.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../new_highs_new_lows/data.py)
  plants a real breadth-lead (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. The preferred sector-ETF breadth basket (XLK XLF XLE … + SPY) is attempted with
  retry/back-off and cached on a real network hit, but the frozen study and the gate run against
  the offline 5-ETF proxy. All headline numbers are pinned in [`docs/results.md`](results.md) and
  reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../103-117`](../..) — the technical-indicator zoo (Turtle, Bollinger, Supertrend, ADX, OBV…);
  most land None × Mirage for the same reason a signal fitted to past price re-describes the trend.
- [`../../301-triple-rsi`](../../301-triple-rsi) — a viral "90% win-rate" rule whose win-rate is the
  shape of the exit, not an edge; the same beta-in-a-costume diagnosis.
- [`../../186-advance-decline`](../../186-advance-decline) and the broader breadth folklore — the
  sibling "participation leads price" claim tested with the random-entry idiom.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; NH-NL breadth is a clean live example of beta and mechanical
  lag masquerading as a leading indicator.
