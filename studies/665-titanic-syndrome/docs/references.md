# References & literature map — Study 665 (Titanic Syndrome)

## The claim under test

- **The folklore.** The **Titanic Syndrome**, created by **Bill Ohama in 1965**: if the
  market reaches a new high within the past **seven trading sessions**, and on that
  reading the number of NYSE issues making fresh **52-week lows** exceeds the number
  making fresh **52-week highs**, the rally's internal breadth has failed to confirm the
  new high — "the band is playing while the ship is already listing" — and a decline
  should follow within roughly the next year. Secondary sources: SentimenTrader
  ("A Titanic Syndrome alert — and two potential lifeboats"; "Hindenburg and Titanic
  Signals Are Clustering"), StockCharts ("Warning Signs for Stocks: The Hindenburg Omen
  and Titanic Syndrome Explained"), McClellan Financial ("Hindenburg and Titanic, OH
  MY!"), and a long-running Bogleheads thread ("Oh no! The Ohama Titanic Syndrome
  signal."). It is explicitly the Hindenburg Omen's older, cruder cousin — same premise
  (breadth divergence at a market peak), a looser trigger, and, per the same commentary
  that popularizes it, the same reputation for "crying wolf."
- **The academic anchor is thin by design.** Unlike the Hindenburg Omen (Miekka 1995's
  quantified thresholds) or the FOMC vol-crush (Amengual-Xiu 2018), the Titanic Syndrome
  has no peer-reviewed literature — it is pure technician folklore, carried entirely by
  financial-media secondary sources. That alone caps any claim at `WEAK` until the real
  tape says otherwise; this study lets the tape speak.

## What we measure, and the honesty rails

- **The signal, mechanically.** ^GSPC at a fresh trailing-252-session high on any of the
  trailing 7 sessions (Ohama's "within 7 days of a new high"; a 52-week high, not a
  literal all-time high — the free-data-computable proxy every breadth study on this
  desk uses, see the data-layer docstring), **AND** the Dow-30 new-52-week-low count
  exceeding the new-high count that same session. Consecutive signal sessions within 21
  calendar days are merged into one *cluster* (the same convention as
  [167-hindenburg-omen](../167-hindenburg-omen/) — no cherry-picking the "best" day of
  one listing episode).
- **Forward returns, three ways.** SPY forward return at 1/5/20/60 sessions, entered at
  the *next* close (one documented lag, zero look-ahead) — measured against (a) a
  drift-matched **random-entry** baseline of the same cluster count (the honest test on
  an upward-drifting tape, same convention as
  [493-new-highs-new-lows](../493-new-highs-new-lows/)) and (b) the plain unconditional,
  monthly-sampled mean. One-sample Newey-West (1987) HAC *t* and Welch (1947) *t* both
  reported.
- **False-alarm rate**, the Hindenburg-style crash-rate cross-check: the share of
  clusters followed within 60 sessions by a ≥5% peak-to-trough SPY drawdown, against the
  same rate on random dates, Welch *t* on the proportions.
- **The timer.** An actual equity-curve overlay (hold SPY, sit in *unremunerated* cash
  for 20 sessions after each cluster, 5 bps one-way cost per transition) — graded not
  just against buy-and-hold but against a **random-timer control** (same cluster count,
  same fixed exit window, dates drawn uniformly at random, 1,000 draws): the fair "is the
  *timing* worth anything beyond sitting out sometimes" test.

## Survivorship, named

The breadth basket is the **current (2026) 30-member Dow Jones Industrial Average** —
a coarse, backward-looking proxy for the thousands of NYSE-listed issues the 1965 rule
was built on. Members removed from the index across the sample window (GE, Pfizer,
Intel, Walgreens, ExxonMobil, Raytheon, DowDuPont, and others) are excluded by
construction, which plausibly **understates** the true new-lows count during any of
their idiosyncratic declines — the bias points **against** the signal firing more often,
the same direction named on the sibling 167-hindenburg-omen study's S&P panel. A basket
built on true historical NYSE membership would be needed to fully rule the bias out; the
desk's position (consistent with 167/493/168) is that a bias pointing against the
signal cannot manufacture the honest `NONE` this study finds.

## Data sources

- **Dow-30 adjusted closes**, **^GSPC daily close** and **SPY adjusted close** —
  yfinance (no key), cached under `_cache/` (`ts_dow30.csv`, `ts_gspc.csv`,
  `ts_spy.csv`), 2008-06-02 → 2026-06-30 (start chosen so every *current* Dow-30
  member — Visa's 2008-03-19 IPO is the youngest — has a full trading history before
  the first possible 252-session lookback).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).
- Ohama attribution and the operational definition (7-session window, 52-week
  highs/lows): SentimenTrader, StockCharts and McClellan Financial secondary sources
  cited above (the desk found no primary 1965 source; the rule has been carried entirely
  by financial-media retellings for six decades).

## Related desk studies (the dedup map — what this study is NOT)

- [167-hindenburg-omen](../167-hindenburg-omen/) — the **quantified, multi-condition**
  sibling (2.2% threshold, MA50 trend filter, McClellan-oscillator sign). This study is
  the **looser, older cousin**: no threshold, no trend filter, no oscillator — just
  "near a high, lows beat highs." Different trigger, same breadth-divergence premise,
  same false-alarm verdict.
- [493-new-highs-new-lows](../493-new-highs-new-lows/) — tests whether a **surge** in
  net new highs (a bullish *breadth thrust*) forecasts higher returns. This study tests
  the mirror-image bearish claim (a **collapse** relative to a fresh high) — opposite
  sign, same NH-NL breadth-line machinery, same random-entry control convention (reused
  directly here for methodological consistency).
- [168-advance-decline](../168-advance-decline/) — the **cumulative A/D line failing to
  confirm** a price high (a different breadth statistic entirely: net advances/declines,
  not 52-week extremes). Same "breadth doesn't confirm the high" family of claims,
  different construction and a different verdict mechanism (sign-flip on the lookback
  sweep there vs an outright null on both signal and false-alarm-rate here).

None of the siblings test Ohama's specific **7-session-window, 52-week-high/low**
construction — the Titanic Syndrome is this study's own object.
