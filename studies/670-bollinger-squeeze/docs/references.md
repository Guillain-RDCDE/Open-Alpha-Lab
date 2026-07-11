# References & literature map — Study 670 (Bollinger-Squeeze)

## The claim under test

- **The folklore.** The **"TTM Squeeze"** — popularised by trader John Carter in *Mastering the
  Trade* (McGraw-Hill, 2006) and built into TradeStation/thinkorswim as a standard indicator —
  says that when the Bollinger Bands (Bollinger, 1980s; *Bollinger on Bollinger Bands*, 2001)
  contract to sit **entirely inside** the Keltner Channel (Chester Keltner, 1960s; the ATR-based
  version popularised by Linda Raschke), volatility has compressed to an unusual low and a big
  directional move is imminent. The recipe: wait for the "squeeze" to fire OFF (bands
  re-expand), then trade the breakout in the direction the price commits to.
- **The academic anchor.** Bollinger Bands and volatility-breakout systems are folk technical
  analysis, not a peer-reviewed anomaly; the closest academic grounding is the broad literature
  on **volatility clustering** (Engle 1982, ARCH; Bollerslev 1986, GARCH) — volatility genuinely
  is autocorrelated and mean-reverting, which is the honest kernel of truth the squeeze
  narrative borrows. Whether a *specific band-geometry event* adds directional information on
  top of ordinary vol clustering is the empirical question this study answers; the technical-
  analysis literature itself (e.g. Lo, Mamaysky & Wang 2000 on pattern recognition) finds mixed,
  largely unconvincing directional evidence for classical chart patterns once a fair control is
  applied.

## What we measure, and the honesty rails

- **Two claims, deliberately separated.** "Volatility expands after a squeeze" (mechanical,
  close to definitional — a squeeze IS low trailing vol, so *something* reverts) is tested
  separately from "the breakout direction is profitable" (the actual tradable claim). Conflating
  them is how folklore survives contact with data — a chart that shows vol expanding after a
  squeeze *looks* like it vindicates the whole system, when only the trivial half is confirmed.
- **The honest vol test uses a matched random-day control**, not just a before/after
  comparison against the squeeze's own (definitionally low) baseline — the second is close to
  tautological. The random-day comparison is what tells you whether the squeeze *times* the
  expansion better than any other day; on this basket it does not (pooled Welch *t* = +0.12).
- **The directional test uses a matched random-timing, same-direction-mix control** — random
  dates carrying the identical multiset of long/short calls the real signals made, so the
  comparison isolates squeeze *timing* from the instrument's generic drift and from whatever
  long/short bias the sample happens to have.
- **Causal, look-ahead-free bands and direction.** Bollinger/Keltner bands use only trailing
  data; the breakout direction is the sign of a causal OLS slope over the trailing 20 bars,
  evaluated on the release bar itself. One documented execution lag: signal on the release bar's
  close, position entered at the next bar's open.
- **Parameter robustness**, not a single hand-picked configuration: BB std ∈ {1.5, 2.0, 2.5} ×
  KC multiplier ∈ {1.0, 1.5, 2.0} × hold days ∈ {5, 10, 20}, 27 combinations. 0 clear the bar.

## Data sources

- **SPY, QQQ, IWM, DIA, GLD daily adjusted OHLC** — yfinance (no key), cached under `_cache/`,
  2005-01-03 → 2026-06-30. Same five-ticker liquid-ETF basket used by sibling studies
  128-keltner-channel and 485-starc-bands, so results line up across the family.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).
- John Carter, *Mastering the Trade* (McGraw-Hill, 2006) — the TTM Squeeze's popular-press
  origin. John Bollinger, *Bollinger on Bollinger Bands* (McGraw-Hill, 2001). Linda Bradford
  Raschke & Laurence A. Connors, *Street Smarts* (1996) — the ATR-channel lineage feeding into
  the modern Keltner Channel.

## Related desk studies (the dedup map — what this study is NOT)

- [104-bollinger-reversion](../104-bollinger-reversion/) — buying the **lower Bollinger Band
  touch** as a mean-reversion signal (and its upper-band breakout contradiction). Single-band,
  single-indicator, snap-back framing. This study: **two indicators compared against each
  other** (BB vs KC geometry) and the **opposite** trade thesis — momentum continuation out of
  a contraction, not reversion off one band.
- [128-keltner-channel](../128-keltner-channel/) — the Keltner Channel's own contradictory
  breakout-vs-reversion folk rules, tested alone (no Bollinger comparison, no squeeze concept).
  This study uses Keltner only as the *second* geometry the Bollinger Band must sit inside —
  the squeeze is a relationship between two envelopes, not either envelope on its own.
- [483-zlema](../483-zlema/) *(brief's cited "483-starc-bands" is this desk's
  [485-starc-bands](../485-starc-bands/))* — STARC bands are a **third**, unrelated envelope
  (SMA ± ATR) testing a lower-band dip-buy, no relationship to Bollinger/Keltner geometry or to
  volatility contraction at all.
- [190-nr7](../190-nr7/) — Crabel's "narrowest range of 7 days" is a **single-bar range**
  contraction signal, refuted on this desk (range narrows *further* the next day, not wider).
  This study's squeeze is a **multi-bar, two-indicator geometric** contraction (BB inside KC)
  held for ≥ 5 bars — a different (and, per Test 1, similarly unhelpful) definition of
  "compressed," tested against a fair random-day control that NR7 did not use for its range
  claim.

None of the siblings test the specific TTM Squeeze geometry (Bollinger contracting inside
Keltner) or separate the mechanical vol-expansion claim from the directional-breakout claim the
way this study does.
