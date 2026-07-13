# References & literature map — Study 729 ("the ramen index" as a downturn tell)

## The claim under test

- **The folklore.** A recurring bit of recession lore that **instant-noodle sales are a
  downturn tell**: because instant ramen is a cheap, shelf-stable *inferior good*, sales
  climb as incomes fall — so a *rise in noodle demand leads a recession* and the noodle
  makers are a *defensive* place to hide. Popularised in press coverage of the 2008 crisis
  ("in a recession, Americans eat more ramen") and as a cousin of Leonard Lauder's **lipstick
  index**. The testable version: (H₁) noodle-demand growth **leads** a market downturn
  (negative lead-lag correlation at a positive lead); (H₂) the noodle stocks **out-return the
  market in recessions**; (H₃) that survives real-time implementation.
- **Where the belief comes from.** Instant noodles are a textbook **inferior good** (Giffen/
  inferior-good demand theory), so demand *should* be counter-cyclical in micro theory; and
  there are real 2008 anecdotes of US instant-ramen unit sales rising. But the aggregate,
  global record is more "secular Asian growth with commodity-price wobbles" than
  "counter-cyclical": world demand actually **fell in 2008–09** (a wheat/palm-oil price spike
  cut volumes) and **fell again in 2014–2016** with no recession, while the only clear jump
  was **2020** — a COVID lockdown/pantry-loading supply shock, not a business-cycle signal.

## The "ramen index" (the leading-indicator series)

- **World Instant Noodles Association (WINA) — global demand.** The authoritative figure for
  worldwide instant-noodle retail **servings per year** (billions), published annually by the
  industry association. Used here as the hardcoded, cited, **approximate** "ramen index"
  (2005–2024), a *labelled proxy* for the folklore's quantity — not a live feed.
  https://instantnoodles.org/en/noodles/demand/  (WINA "Global Demand" tables)
- **The publication lag (half the tradability kill).** WINA reports a calendar year's demand
  only in the *middle of the following year*, so even a working ramen tell reaches you ~6
  months after the year it describes.
- **US-only caveat.** The original anecdote is *American* instant-ramen sales (Nielsen/IRI
  scanner data), which can diverge from the global WINA figure. Study 729 tests the global
  series (the only comprehensive cited number) and flags the US-only tape as a Beat-7
  extension, not a hidden substitution.

## The recession dating (the event windows)

- **NBER Business Cycle Dating Committee — US business-cycle peaks and troughs.** The
  authoritative recession chronology. The three contractions inside the noodle-stock sample:
  **2001-03 → 2001-11** (dot-com), **2007-12 → 2009-06** (GFC), **2020-02 → 2020-04** (COVID).
  https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
- **The announcement lag (the other half of the kill).** The committee dates cycles
  **ex-post**: it announced the Dec-2007 peak in **Dec-2008** (~12 months late) and the
  Apr-2020 trough in **Jul-2021** (~15 months late).
  https://www.nber.org/news/business-cycle-dating-committee-announcements
  Combined with the WINA release lag, a ramen-conditional strategy is a **double look-ahead**.

## The tradable names (what a public investor can actually buy)

- **Nissin Foods Holdings (`2897.T`, Tokyo).** The inventor of instant noodles (Chikin Ramen,
  1958) and Cup Noodle (1971); owns Top Ramen/Cup Noodles in the US. The world's largest
  instant-noodle company and the archetypal "ramen stock."
- **Toyo Suisan (`2875.T`, Tokyo).** Maker of **Maruchan** — the best-selling instant noodle
  in the United States — plus seafood and frozen foods.
- **`^N225`** — the Nikkei 225, the home-market benchmark the "beats the market" claim is
  judged against.
- *(Beat 7 extensions:)* Nissin Foods (China) `1475.HK`; the staples cousins for the "lipstick
  index" family — discount retail `DG`/`WMT`, Spam maker Hormel `HRL`.

## Why the significant alpha is *not* the claim — the finance

- **Survivorship.** Ang et al. and the desk's own guard (METHODOLOGY → *Survivorship named on
  the Signal axis*) warn that a hand-picked *surviving winner* manufactures outperformance.
  `2897.T`/`2875.T` are the two noodle champions that survived and thrived; the selection
  points **for** the claim, so the CAPM alpha they show against the Nikkei is treated as
  suspect, not as evidence.
- **Low beta ≠ alpha ≠ counter-cyclical.** A β ≈ 0.2–0.3 staple that beats a moribund index is
  the **low-volatility / defensive-equity anomaly** (Baker, Bradley & Wurgler 2011, *Benchmarks
  as Limits to Arbitrage*; Frazzini & Pedersen 2014, *Betting Against Beta*), a priced,
  freely-replicable exposure — **not** a recession tell. Downside beta (Ang, Chen & Xing 2006,
  *Downside Risk*, *RFS*) is the right lens for "does it fall less"; the CAPM alpha (Jensen
  1968) with **Newey-West (1987)** HAC errors is the test for excess beyond beta.
- **Lipstick/inferior-good indices as folklore.** Lauder's lipstick index and its many cousins
  are widely cited and rarely robust; the counter-cyclicality of "small indulgences" is mostly
  anecdote (see press retrospectives on the lipstick index's mixed record). The desk's job is
  to separate the real (secular growth, low beta) from the sold (a leading recession signal).
- **Small samples & selection.** With 19 annual demand observations, 4 recession years and 3
  equity recessions, both the lead-lag and event-study tests are low-power and hostage to one
  window (COVID). White (2000)'s data-snooping warning and the desk's Reality-Check discipline
  apply: an "edge" that is one COVID data point is not a law.

## Method lineage (the desk's shared engine)

- **Lead-lag detection.** A cross-correlation of demand growth vs forward market returns with a
  small-sample correlation *t* ([`strategy.lead_lag_corr`](../ramen_recession/strategy.py)),
  plus an in-vs-out-of-recession Welch *t*
  ([`strategy.demand_in_vs_out_recession`](../ramen_recession/strategy.py)). `REAL` would need a
  negative, significant lead at $k\ge1$.
- **Risk/return primitives & defensiveness.** CAGR/vol/Sharpe/MDD
  ([`strategy.summarize`](../ramen_recession/strategy.py)) and bull/bear conditional beta
  ([`strategy.bull_bear_beta`](../ramen_recession/strategy.py)); defensive ⟺ β⁻ < β⁺ < 1.
- **Robust inference.** A **Newey-West (HAC)** *t* of the CAPM alpha vs the Nikkei
  ([`strategy.newey_west_alpha_t`](../ramen_recession/strategy.py)) and a paired recession-window
  excess *t* ([`strategy.recession_excess_t`](../ramen_recession/strategy.py)).
- **Deterministic synthetic controls.** A fixed-seed planted-*lead* generator and a planted-
  *defensive-beta* generator ([`data.synthetic_leading_index`](../ramen_recession/data.py),
  [`data.synthetic_defensive`](../ramen_recession/data.py)) proving the detectors recover a real
  signal — run with no network.

## Data sources used here

- **yfinance** (Yahoo Finance) month-end Adj Close for `2897.T`, `2875.T`, `^N225`, cached under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).
- **WINA** world instant-noodle demand (hardcoded, cited, approximate) and **NBER** recession
  dates (hardcoded, cited) as above — labelled facts/proxies, not feeds.

## Related desk studies

- **[Study 728 — Is beer recession-proof?](../../728-beer-recession/)** — the immediate sibling:
  a consumer-staple "defensive/counter-cyclical" claim tested on tradable single stocks vs a
  benchmark, same bull/bear-beta + recession-window machinery, same None/Mirage shape.
- **[Study 358 — Watches are an asset class?](../../358-watch-index/)** — the same
  labelled-proxy + tradable-leg construction and the same survivorship-narrated-as-system
  signature.
