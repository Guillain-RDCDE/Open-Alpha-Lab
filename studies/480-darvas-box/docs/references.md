# References & literature map — Study 480 (Darvas Box)

## The claim under test

- **The folklore.** Draw a *box* around a consolidation: a stock makes a **new high**; it fails
  to take that high out for a few sessions (the recent high becomes the **box top**); the pullback
  low becomes the **box bottom**. A **close above the box top** is the breakout — *buy*, stop just
  below the box bottom (an ATR/box stop), and ride the next box up. "The breakout forecasts
  continuation." This is a retail/momentum staple, restated on every trend-trading site and built
  into screeners as the "Darvas box" indicator.
- **The source.** **Nicolas Darvas**, a professional ballroom dancer, described the method in
  *How I Made $2,000,000 in the Stock Market* (1960) — he traded by telegram while touring, using
  only the highest and lowest prices to define boxes and buy fresh-high breakouts. The book is the
  primary source; modern write-ups (Investopedia's "Darvas Box Theory", Darryl Guppy's and Jon
  Boorman's restatements, StockCharts' ChartSchool) re-encode the rule.
- **Variants.** The "box" is a special case of a **Donchian/turtle channel breakout** (buy a new
  N-day high) and of classic momentum/breakout systems; min-box length, ATR-stop width and the
  channel lookback are the free parameters. All are **affine variants of the same trailing-high
  breakout** and inherit the same drift confound tested here.

## Why this is a "theory" / mechanical-proxy study

Darvas' boxes are *semi-subjective*: a discretionary trader decides when a consolidation "counts"
as a box. Following the desk's design for this kind, we encode the **tightest mechanical rule a
proponent would accept** and state the irreducible subjectivity explicitly:

- **Objective box top.** The trailing 20-day high of the close (a Donchian upper band), usable
  only with a one-bar shift — a documented trailing window, no look-ahead.
- **Objective consolidation.** We require the close to have sat **below** the box top for ≥ 5
  consecutive bars before the breakout, so a box has actually formed (not a runaway).
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* long-only breakout
  inherits the drift. We add a **shuffled-box placebo** that scatters the entry dates at random
  (same count, same marginal) — the direct test of "does the box timing matter?"

Hand-picked boxes add *hindsight* (a free parameter), which can only inflate in-sample fit; the
mechanical version is therefore the charitable **upper bound** on the method.

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of a long-only entry rule against **zero** measures that drift, not the rule. See Fama & French
  on the equity premium; the desk's standing rule is *excess-vs-excess* and *signal-vs-baseline*,
  never *signal-vs-zero*. Here the breakout's one-sample *t* hits +6.4 yet it *loses* to random.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical Analysis*,
  Journal of Finance) formalize testing chart patterns against a properly matched null; Sullivan,
  Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*,
  JF) and White (2000, *A Reality Check for Data Snooping*, Econometrica) show how trend-fitted
  breakout rules manufacture significance unless raced against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the breakout-vs-random difference.

## Method lineage (the desk's shared engine)

- **Trailing box levels + breakout entries.** [`strategy.box_levels`](../darvas_box/strategy.py),
  [`strategy.breakout_entries`](../darvas_box/strategy.py) — the mechanical Donchian-style box with
  the consolidation requirement and the one-bar trailing shift baked in.
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../darvas_box/strategy.py),
  [`strategy.hac_t`](../darvas_box/strategy.py), [`strategy.run_experiment`](../darvas_box/strategy.py).
- **Geometry placebo.** [`strategy.shuffled_box_placebo`](../darvas_box/strategy.py) — scatter the
  entry dates, keep count + marginal, destroy the box timing.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../darvas_box/data.py) plants a
  real post-breakout continuation (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the sibling "price respects the
  drawn channel" folklore tested with the same random-entry baseline idiom.
- [`../../437-donchian-breakout`](../../437-donchian-breakout) and
  [`../../103-turtle-trader`](../../103-turtle-trader) — the channel-breakout family the Darvas box
  belongs to; same drift confound.
- [`../../178-cci`](../../178-cci) and the broader technical-indicator zoo — most land
  None × Mirage for the same reason: an indicator fitted to past price re-describes the trend.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the Darvas box is a clean live example of beta masquerading as
  a breakout strategy.
