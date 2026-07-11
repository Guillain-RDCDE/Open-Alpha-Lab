# References & literature map — Study 666 (McClellan Summation Index)

## The claim under test

- **The folklore.** Sherman & Marian McClellan introduced the **McClellan Oscillator** in
  1969 (*Patterns for Profit*) as a breadth-momentum indicator: EMA₁₉(net advances) −
  EMA₃₉(net advances). They later extended it into the **Summation Index** — the running
  cumulative sum of the oscillator — explicitly framed as a *slower, level-based* companion:
  where the oscillator flags a single day's momentum shift, the Summation Index is meant to
  confirm a *regime* — a bull or bear market phase. The standard teaching (McClellan
  Financial Publications, and every breadth-trading site since): the Summation Index
  **crossing up/down through zero** confirms a new bull/bear phase, and extreme excursions
  (classically **±500 / ±1000** on the full-NYSE scale) mark overbought/oversold turning
  points ripe for reversal.
- **The academic anchor.** Market breadth as a market-timing input has a long, mixed
  academic history: Fosback (1976, *Stock Market Logic*) and Zweig (1986, *Winning on Wall
  Street*) popularized breadth thrusts and oscillators as trading tools; more rigorous
  academic tests of breadth-based timing rules (e.g. the advance-decline line, new-highs
  minus new-lows) have generally found weak or no out-of-sample forecasting power once a
  drift-matched baseline is imposed — the same finding this desk's sibling studies (491,
  493, 494, 168) independently reach for the oscillator, the new-highs/new-lows line, the
  bullish percent index, and the advance-decline line respectively.
- **What's specifically new here.** This study is the first on the desk to test the
  **integral**, not a single-day trigger: does *accumulating* the oscillator over time turn
  a noisy daily signal into a genuine, tradable multi-day *regime* gauge? A running sum could
  in principle smooth away noise that defeats a single-day trigger — that's the honest reason
  to test it separately rather than assume 491's "None" verdict on the oscillator
  automatically transfers to its integral.

## What we measure, and the honesty rails

- **Causal Summation Index.** `Summ_t = cumsum(EMA₁₉(net_adv) − EMA₃₉(net_adv))_t` — a plain
  running sum of a causal oscillator; no centering, no look-ahead.
- **The zero-cross, taken literally first.** We tested the textbook rule exactly as taught —
  and found it **cannot fire** on this real tape: the un-rebased cumulative sum climbs away
  from zero in the first few months (2005) and never returns, in either direction, across
  21.4 years and every basket/EMA-span variant tried. This is reported as a finding, not
  hidden as a null result — see `docs/results.md`.
- **The ±500 level, honestly rescaled.** The literature's numeric thresholds assume
  full-NYSE-exchange breadth (thousands of issues); a 10-name ETF proxy cannot reproduce them
  without rescaling. We use a **causal rolling z-score** (252-session trailing window, no
  look-ahead) and treat a ±1σ cross as the scale-appropriate analog — an explicit
  operationalization decision, stated up front rather than silently hard-coding "500" onto a
  series that never gets near it.
- **The Signal axis is always trigger/regime vs a drift-matched baseline** (random-entry
  Welch *t* for events; buy-and-hold HAC *t* for the regime timer) — never trigger-vs-zero,
  because an upward-drifting index makes *any* long exposure look good in isolation. This is
  the same discipline as sibling study 491.
- **One documented execution lag.** Events enter at the next close; the regime timer applies
  yesterday's close-of-day regime to today's return (`pos = regime.shift(1)`) — zero
  look-ahead either way.
- **Costs.** One-way × NAV, charged twice per event-study trade (in + out, 1 bp — cheap,
  liquid SPY/sector-ETF spreads) and once per regime switch (5 bps — the regime timer's
  round trips are far less frequent, so a heavier per-switch cost is the conservative
  choice).
- **Survivorship, named.** SPY and the 9 classic SPDR sector ETFs (live continuously since
  1998) are the *actual* traded universe throughout 2005→2026 — no membership changes, no
  delisted names dropped, unlike a current-constituent index panel. No survivorship
  adjustment needed on the Signal axis.

## Data sources

- **SPY daily total-return closes** and the **breadth-basket** (`SPY XLK XLF XLE XLV XLI XLY
  XLP XLU XLB`) — yfinance (no key), cached under `_cache/` as parquet, 2005-01-04 →
  2026-06-30.
- McClellan Financial Publications, "The McClellan Oscillator and Summation Index":
  https://www.mcoscillator.com/learning_center/weekly_chart/the_mcclellan_oscillator_and_summation_index/
- Fosback, N. (1976). *Stock Market Logic*. Institute for Econometric Research.
- Zweig, M. (1986). *Winning on Wall Street*. Warner Books.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [491-mcclellan-oscillator](../../491-mcclellan-oscillator/) — the **oscillator itself**
  (EMA₁₉ − EMA₃₉ of net advances) as a single-day up-cross trigger. This study tests its
  **running integral** as a multi-day regime gauge — a different object (a level, not a
  momentum spike) with its own failure mode (the un-rebased sum never revisits zero) that has
  no analog in 491's design.
- [494-bullish-percent-index](../../494-bullish-percent-index/) — % of a basket above its
  50-day SMA, an oversold/overbought **breadth-diffusion** gauge (Point & Figure column
  reversals), not a cumulative-sum indicator.
- [168-advance-decline](../../168-advance-decline/) — the **raw cumulative** advance-decline
  line (no EMA smoothing) tested as a price/breadth **divergence** signal, not a
  self-contained level-crossing regime rule.
- [493-new-highs-new-lows](../../493-new-highs-new-lows/) — the **52-week new-highs-minus-
  new-lows** breadth-thrust line, a different construction (extremes-based, not
  advance/decline-based) testing a "thrust" trigger, not a cumulative-sum regime gauge.
- **667** — a further breadth-family sibling in this lot (not yet published at the time of
  writing); once live it will be cross-referenced here.

None of the published siblings test the **Summation Index's own zero-cross / extreme-level /
long-flat-regime claim** — that is this study's own axis.
