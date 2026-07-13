# References & literature map — Study 770 (Concert-Economy)

## The claim under test

- **The folklore.** "The concert economy is booming — buy Live Nation before festival
  season." A recurring retail / financial-media trade idea: because Live Nation (LYV)
  makes most of its money in the summer (Coachella, the touring circuit, the amphitheatre
  season), the stock is supposed to *rally into* that window as the market front-runs the
  predictable revenue surge. The steelman is unusually strong for a folklore claim,
  because the underlying seasonality is genuinely large and genuinely predictable.
- **The fundamental anchor (real, and cited).** Live Nation's own filings show a heavily
  Q3-weighted revenue calendar — the summer touring quarter is by far the biggest.
  Segment- and quarter-level figures are in the **10-K / 10-Q** filings (SEC EDGAR,
  ticker LYV) and are the source for the LABELLED PROXY series in
  [`data.py`](../concert_economy/data.py): annual total revenue and an approximate
  quarterly share (Q1≈16% / Q2≈28% / **Q3≈37%** / Q4≈19%). These are reconstructed
  public figures used only to draw the *fundamental* backdrop — never presented as a live
  tape.
- **The efficient-markets prior.** A calendar everyone can see (Coachella's dates are
  announced each January; the touring season is the same every year) is exactly the kind
  of information a semi-strong-efficient market should already have in the price. The
  desk's prior is therefore that any "rally into" is already arbitraged away and the trade
  is folklore — see Fama (1970, *Efficient Capital Markets*, JF) for the canonical
  statement, and the seasonal-anomaly literature below for why known calendars rarely pay.

## The academic backdrop (calendar/seasonality anomalies)

- **Bouman & Jacobsen (2002), *The Halloween Indicator, "Sell in May and Go Away"*,
  AER** — the reference study for a *predictable-calendar* seasonal claim tested honestly
  across markets; the template for "is this known seasonal pattern actually tradable or
  already priced?"
- **Heston & Sadka (2008), *Seasonality in the Cross-Section of Stock Returns*, JFE** —
  documents robust seasonal patterns in individual-stock returns, and the debate over
  whether they are risk, mispricing, or data-snooping. The relevant caution for a
  single-name seasonal like LYV: with ~18 annual observations, one "significant" cut among
  several is the base rate of luck.
- **Edmans, García & Norli (2007), *Sports Sentiment and Stock Returns*, JF** — the real
  mood-to-market channel (elimination shocks in football). Cited here as the *contrast*:
  that is an event-driven surprise effect; the concert-economy claim is a
  *fully-anticipated calendar* effect, which efficient markets handle very differently.

## What we measure, and the honesty rails

- **The calendar is hardcoded** (`data.py`, `EVENTS`) from Wikipedia's per-year Coachella
  Valley Music and Arts Festival pages — weekend-1 Friday for every edition 2006→2025.
  2020 and 2021 were COVID-cancelled (no festival, no event) and are named, not hidden.
- **One execution convention, and it needs no surprise-day lag.** Coachella's opening date
  is announced months ahead, so the run-up window [anchor−K → anchor] is **calendar-known
  and zero-look-ahead by construction** — a "buy K sessions before, sell as it opens" rule
  could have been placed in advance every year. Signal and tradable capture are therefore
  the *same* window (gross vs net of 2× one-way cost × NAV), unlike a surprise-event study
  that must lag entry to the first post-news close.
- **Beta, named on the Signal axis.** LYV's full-sample daily beta to SPY is **1.35**, so a
  raw `LYV − SPY` difference is not clean alpha over long windows. This is the reason the
  one number that looks large — the +8.5% *in-season* drift — is not evidence of
  front-running: it is inside a random 4.5-month window's luck cloud (whose mean is itself
  positive) and survives beta-adjustment only weakly (*t* = 1.81, below the bar).
- **Price basis.** Both legs use total-return (dividend-adjusted) closes and are labelled
  as such — LYV pays no dividend but SPY does, and using `auto_adjust=True` on both keeps
  the comparison honest.
- **Inference unit.** Each festival year is one independent, non-overlapping event — the
  correct test is a **one-sample t** of the abnormal return across events, not a daily
  panel regression. A random-window placebo (many non-Coachella K-session windows on the
  *same* LYV/SPY pair) checks whether the observed mean sits outside LYV's ordinary
  tracking noise against SPY.

## Data sources

- **Daily adjusted (total-return) closes** for `LYV` and `SPY` — yfinance (no key), cached
  under `_cache/`.
- **Coachella weekend-1 dates, 2006→2025** — hardcoded in
  [`data.py`](../concert_economy/data.py). Source: Wikipedia, "Coachella Valley Music and
  Arts Festival" (https://en.wikipedia.org/wiki/Coachella_Valley_Music_and_Arts_Festival)
  and the individual per-year festival pages.
- **Live Nation revenue seasonality (LABELLED PROXY)** — reconstructed from Live Nation
  Entertainment's SEC filings (10-K annual revenue; 10-Q quarterly splits), SEC EDGAR
  ticker `LYV` (https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=LYV).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [708-eurovision-effect](../../708-eurovision-effect/) — the same event-study machinery
  (hardcoded cultural calendar, one-sample *t* across years, random-window placebo) applied
  to a *surprise* result (a Saturday-night winner) on a per-country ETF panel. This study
  differs on the two axes that matter: a single high-beta stock (not a panel), and a
  *fully-anticipated* calendar (not a surprise), which is why the execution convention here
  needs no post-news lag.
- [150-sad-effect](../../150-sad-effect/) — a seasonal *mood* anomaly (seasonal affective
  disorder and equity returns). Same "known calendar" family; different mechanism and a
  market-wide (not single-name) test.
- [234-olympic-year](../../234-olympic-year/) — "stocks rally in Olympic years," a
  macro-frequency calendar claim on a broad index. Concert-Economy is a single-week,
  single-stock event window, not an annual-frequency macro claim.
- [358-watch-index](../../358-watch-index/) — the "collectibles as an asset class" study
  whose LABELLED-PROXY pattern (a small, cited, hardcoded fundamental series standing in
  for a non-API data source) this study reuses for the touring-revenue backdrop.

No sibling tests **a single live-events stock's front-running of its own predictable
revenue seasonality** — the concert-economy angle, including the "the seasonality is real
but the front-run isn't, and the one big number is just beta" finding, is this study's own
contribution.
