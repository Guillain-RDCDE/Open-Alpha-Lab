# References & literature map — Study 731 (Wimbledon-Effect)

## The claim under test

- **The folklore.** During the Wimbledon fortnight (late June → mid-July) the City is
  supposed to empty out — everyone at the tennis, on the river, or on holiday — so
  trading volumes thin and the UK market settles into a quiet "summer lull" with its own
  distinctive drift, worth stepping aside for (or, in the trader-bar telling, worth
  fading). It is a British, tennis-flavoured instance of the broader summer-lull /
  *Sell-in-May* seasonal intuition, pinned to the most photogenic two weeks of the
  English calendar.
- **No academic anchor.** There is no peer-reviewed study of a "Wimbledon stock effect"
  that we are aware of — this is financial-media and City-desk folklore, not a tested
  claim. That itself is a data point: the desk starts here with a low prior, exactly as
  it did for [708-eurovision-effect](../../708-eurovision-effect/) (another
  no-academic-anchor cultural-calendar claim).
- **The nearest *real* seasonal literature (a different, broader claim).** The
  summer-lull family *does* have a serious anchor. Bouman & Jacobsen (2002, *The
  Halloween Indicator, "Sell in May and Go Away": Another Puzzle*, American Economic
  Review) document lower May–October equity returns across many markets, the UK among
  them. Kamstra, Kramer & Levi (2003, *Winter Blues: A SAD Stock Market Cycle*,
  American Economic Review) tie seasonal return patterns to daylight/mood. These concern
  a *six-month* seasonal, not a specific *two-week* window — the Wimbledon claim is a
  much narrower, much stronger version, and if a broad summer lull exists it would be
  removed here by the VGK Europe benchmark (which shares any pan-European summer effect),
  isolating anything genuinely *UK-and-fortnight-specific*.
- **The "other" Wimbledon effect (disambiguation).** In economics the phrase "Wimbledon
  effect" more often denotes the observation that the UK hosts a thriving financial /
  sporting arena won mostly by foreign players/firms (see e.g. discussions in UK
  financial-services policy). That is **not** this study — we test the literal
  stock-market summer-lull folklore, on the tradable UK equity ETF.

## What we measure, and the honesty rails

- **The calendar is hardcoded** (`data.py`, `WIMBLEDON`) from Wikipedia's "Wimbledon
  Championships" article and the per-year "<YEAR> Wimbledon Championships" pages: each
  fortnight's first-Monday start and second-Sunday (gentlemen's final) end. 2020 is
  **COVID-cancelled** (the first cancellation since the Second World War) and carries no
  market event. Every non-cancelled pair is **asserted at import** to be a Monday→Sunday
  span exactly 13 days apart — a fat-fingered date fires an `AssertionError` rather than
  silently poisoning the event study. Since 2015 the Championships begin a week later
  (three weeks after the French Open final), visible in the drift of start dates; the
  study tests explicitly whether that schedule shift matters (it doesn't).
- **No look-ahead, by construction.** The fortnight dates are published *years* in
  advance, so this is a **calendar-known** window: entry (last close before the first
  Monday) and exit (last close inside the fortnight) are both known ex ante. Per the
  desk's execution rule, a calendar-known window needs **no `shift`** and carries **zero**
  look-ahead — unlike an announcement/event study.
- **Why EWU, and why also VGK.** `EWU` (iShares MSCI United Kingdom) is the tradable UK
  equity vehicle and the natural home for a "FTSE summer lull" test. `VGK` (Vanguard
  FTSE Europe) is the benchmark for the *abnormal* cut (EWU − VGK): it shares any
  pan-European summer seasonal, so subtracting it isolates a *UK-and-fortnight-specific*
  effect from the generic summer drift. VGK's inception (2005-03-10) is the hard floor on
  the sample — a real constraint, disclosed, not patched with an index proxy.
- **Inference unit.** Each Championships year is one independent, non-overlapping event —
  the correct test is a **one-sample t** of the fortnight return across years, not a
  daily panel regression. A two-sided **random-window placebo** (many same-length windows
  drawn at random points in the *same* tickers' history) checks whether the observed mean
  sits outside the tickers' ordinary two-week noise; the folklore names no direction, so
  the placebo is two-sided.
- **Three axes, three claims.** The folklore contains two testable assertions — a
  directional *return* seasonal and a *volatility* lull — so the study carries a third,
  myth-check axis (**"A real lull?"**) that tests the literal "quiet window" claim via a
  realized-vol ratio, separately from the Signal/Tradability return axes.
- **Costs, labelled.** One-way × NAV per leg (5 and 10 bps sweeps). The market-neutral
  construction shorts VGK and **pays borrow** (0.50%/yr, pro-rated over the ~10-session
  hold). Gross and net are labelled everywhere; both ETF series are **total-return**
  (dividends reinvested), stated as such.

## Data sources

- **Daily adjusted (total-return) closes** for `EWU` and `VGK` — yfinance (no key),
  cached under `_cache/`.
- **Wimbledon Championships fortnight dates, 2005→2025** — hardcoded in
  [`data.py`](../wimbledon_effect/data.py). Source: Wikipedia, "Wimbledon Championships"
  (https://en.wikipedia.org/wiki/Wimbledon_Championships) and the per-year
  "<YEAR> Wimbledon Championships" pages, cross-checked against the gentlemen's-singles
  final dates.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [708-eurovision-effect](../../708-eurovision-effect/) — a national-mood **event**
  window on single-country ETFs, keyed to a broadcast result. Same family of
  no-academic-anchor cultural-calendar folklore, but an *event* (winner announced) with
  an execution lag, not a *pre-scheduled seasonal window*; and a per-country panel, not a
  single UK-vs-Europe series.
- [235-world-cup-effect](../../235-world-cup-effect/) — the Edmans-style sports-sentiment
  mechanism (football elimination shocks) on the S&P 500. A real academic anchor, a
  *shock* event, a US market — none of which this study shares.
- **The *Sell-in-May* / Halloween-indicator family** — the serious seasonal-lull
  literature (Bouman & Jacobsen 2002). This study is a narrow, two-week, UK-specific,
  benchmark-differenced instance of that family — and, unlike the six-month version,
  finds nothing once the pan-European summer drift is removed.

None of the siblings test a **UK-specific, pre-scheduled two-week calendar window against
a Europe benchmark, with a realized-volatility lull check as a third axis** — the
Wimbledon angle, including the "the fortnight isn't even quieter" finding, is this
study's own contribution.
