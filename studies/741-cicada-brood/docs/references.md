# References & literature map — Study 741 (Cicada-Brood)

## The claim under test (and why it is deliberately silly)

- **There is no serious "cicada indicator."** This study is a **spurious-pattern demo**,
  built and labelled as such: it invents the strongest possible version of a folklore
  calendar signal and shows it is noise. The genre it parodies is real, though — the
  "market almanac" tradition of pinning returns to fixed, evocative calendars: the
  *Stock Trader's Almanac* (Hirsch, annual) popularised "Sell in May and Go Away", the
  Santa-Claus rally, the January barometer and the Super Bowl indicator, the last of
  which (Krueger & Kennedy 1990, *Journal of Finance*, "An Examination of the Super Bowl
  Stock Market Predictor") is the canonical academic write-up of an impressive-looking,
  utterly mechanism-free calendar coincidence. A "17-year cicada bull" is the same joke
  with insects: two marquee emergences (Brood X in 2004 and 2021) happened to fall in up
  markets, and folklore does the rest.
- **The 13/17-year cicada clock is real and famous, which is exactly why it makes a good
  null.** Periodical cicadas (*Magicicada* spp.) emerge on rigid 13- or 17-year cycles
  that are known *decades* in advance — so unlike almost every other event study on this
  desk, there is **zero look-ahead risk**: you could have scheduled the trade at the
  previous emergence. If even a perfectly-foreseeable, media-saturated fixed calendar
  produces no edge, the lesson about data-snooped calendar patterns is as clean as it
  gets.

## The cicada science (the calendar's provenance)

- **UConn / Cooley *Magicicada* brood chart** — magicicada.org, maintained by John
  Cooley and colleagues (University of Connecticut), the canonical academic schedule of
  the extant periodical-cicada broods (active 17-year broods I–X, XIII, XIV; active
  13-year broods XIX, XXII, XXIII), with mapped emergence years and geographic ranges.
  This is the primary source for the hardcoded `BROODS` table.
- **US Forest Service, "Periodical Cicada" (Eastern Region)** — the agency overview of
  *Magicicada* life history, emergence triggers (soil temperature ~18 °C at ~20 cm,
  early-to-mid May in the core range) and the six-week above-ground adult window that
  sets this study's May-June event window.
- **Marlatt, C. L. (1907), "The Periodical Cicada"** (USDA Bureau of Entomology Bulletin
  71) — the historical monograph that assigned the Roman-numeral brood numbering still in
  use, cited for the broods' identity and cycle lengths.
- **Marquee emergences cross-checked against contemporary national coverage** — Brood X
  (Great Eastern, 2004 & 2021), the Northern-Illinois Brood XIII (2007), the East-Coast
  Brood II "Swarmageddon" (2013), and the rare 2024 Brood XIII × Brood XIX 17-/13-year
  dual co-emergence, each of which was front-page science news and would be the raw
  material for any "cicada indicator" folklore.

## What we measure, and the honesty rails

- **Constant-mean market model** (Brown & Warner 1985, *Journal of Financial Economics*,
  "Using daily stock returns: The case of event studies"): the "normal" return is SPY's
  full-sample mean daily return, so the abnormal return is the demeaned series and a
  spring CAR is measured *above the market's ordinary drift*, not "stocks go up".
- **The event unit is the emergence YEAR** — independent, non-overlapping — so the
  primary statistic is a **one-sample *t*** across years, not a daily-panel regression
  (which would badly overstate significance by treating autocorrelated daily returns as
  independent observations). No HAC correction is needed once each event is summarised to
  one number.
- **The random-year placebo** draws the same number of random years and averages the
  same May-June window, so the ordinary seasonal ("Sell in May") baseline is common to
  events and placebo alike — the test isolates *cicada* springs from *random* springs,
  not spring from other seasons. The up-rate carries a **Wilson (1927) interval**.
- **Alpha, not beta, on the tradability axis.** The tradable overlay is graded on its
  **excess over the unconditional every-spring baseline**, never on a one-sample *t* of
  the raw return vs zero — because a two-month equity window is positive on average for
  reasons that have nothing to do with cicadas (the desk's "normalise before you marvel"
  / "alpha vs beta" rule). Costs are one-way × NAV per leg, one round trip per year,
  long-only.
- **Zero look-ahead, stated as the study's one execution convention** — the emergence
  year is fixed by the brood clock and known since the previous emergence, so entering
  at the last April close before the window is fully foreseeable, not a peek.

## Data sources

- **SPY** daily **total-return** close (`auto_adjust=True`) — yfinance (no key), cached
  under `_cache/cicada_spy.csv`, 1993-01-29 → 2026-06-30. SPY is a real tradable
  instrument, so no proxy series is used anywhere in this study.
- **30 hardcoded periodical-cicada brood emergences (24 distinct years), 1996 → 2025**,
  in [`cicada_brood/data.py`](../cicada_brood/data.py) — from the UConn/Cooley brood
  chart and USFS/USDA sources above.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [707-plane-crash-effect](../../707-plane-crash-effect/) and
  [708-eurovision-effect](../../708-eurovision-effect/) — the nearest siblings: a
  hardcoded, cited event calendar; an event study on a tradable instrument around each
  event; one-sample *t* across independent events; a random placebo; a costed timer.
  Same machinery, different (equally folkloric) trigger. Those test *sentiment* shocks;
  this one tests a **pure fixed-calendar coincidence** with no sentiment channel even
  proposed — it is the most explicitly null member of the family.
- The market-almanac cousins on the desk — turn-of-month, "sell in May", day-of-week and
  holiday-calendar effects — are the *serious* calendar studies; Cicada-Brood is their
  deliberately absurd control, included to show the same rigorous apparatus returns
  "nothing here" when there is, by construction, nothing to find.
