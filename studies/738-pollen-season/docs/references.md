# References & literature map — Study 738 (Pollen-Season)

## The claim under test

- **The folklore.** "Sell in May" has a mirror-image cousin traders repeat every spring:
  buy the allergy names *before* pollen season. The reasoning is a clean
  demand-seasonality story — every March through May, tens of millions of US hay-fever
  sufferers restock antihistamines and nasal sprays, a spike visible in
  retail-scanner/OTC-cough-cold-allergy sales data (IRI/Nielsen category reports,
  quarterly cited by the OTC industry body **CHPA**, the Consumer Healthcare Products
  Association). The tradable corollary: the listed owners of the big allergy brands
  should carry a repeatable spring seasonal in their share price relative to the market.
- **Why the window is a labelled calendar rule, not a data feed.** US spring pollen
  (tree first, then early grass) runs roughly **March→May**. There is no free,
  machine-readable national pollen-index *price series* to trade against, so — exactly as
  [`358-watch-index`](../../358-watch-index/) and
  [`708-eurovision-effect`](../../708-eurovision-effect/) do with their hand-built
  calendars — the season is encoded as a small, clearly-cited **calendar window**
  (last session of February → last session on/before May 31), never presented as a real
  tape. Sources for the window: the **Asthma and Allergy Foundation of America (AAFA)**
  annual *Allergy Capitals* report; the **American Academy of Allergy, Asthma &
  Immunology (AAAAI)** / National Allergy Bureau (NAB) regional pollen calendars; and
  Pollen.com/IQVIA station data — all of which put the tree/grass pollen build-up across
  these three months.
- **The basket.** Five currently-listed owners of household allergy brands: **Bayer**
  (Claritin/loratadine), **Sanofi** (Allegra/fexofenadine, plus the Regeneron-partnered
  biologic Dupixent), **Perrigo** (the largest US private-label maker of the generic
  antihistamines and nasal sprays), **Kenvue** (Zyrtec/cetirizine + Benadryl, spun off
  from Johnson & Johnson in 2023) and **Haleon** (Flonase/fluticasone, spun off from GSK
  in 2022). Brand ownership cross-referenced to each company's product pages and the
  respective spin-off prospectuses (Kenvue S-1 2023; Haleon demerger 2022).

## The shared method (the desk's event/calendar-study gauntlet)

- **The unit of analysis is one spring window per year** — an independent,
  non-overlapping event — so the primary statistic is a **one-sample *t*** across the
  yearly abnormal returns, **not** a daily-panel regression whose thousands of
  autocorrelated rows would grossly overstate the degrees of freedom. This is the same
  "independent-events, not a daily panel" discipline as
  [`707-plane-crash-effect`](../../707-plane-crash-effect/) and `708-eurovision-effect`.
- **Abnormal return** = basket total-return over the window minus the benchmark's
  total-return over the same window (a difference-in-returns market model; the
  window-return analogue of the constant-mean/market model of **Brown & Warner 1985**,
  *Journal of Financial Economics*, "Using daily stock returns: The case of event
  studies").
- **Hit rate carries a Wilson (1927) score interval**; the mean carries a
  **block-bootstrap** percentile CI (events resampled with replacement).
- **Random-window placebo** — the falsification design shared across the desk's
  event studies: recompute the same statistic on same-length windows anchored at random
  *non-spring* dates, many seeds, and read the observed value's tail position. Here it
  doubles as a **teaching case**: the placebo's borderline *p* disagrees with the
  vs-zero *t* precisely because the two null different quantities (spring-vs-random-window
  vs spring-vs-zero), and the write-up spells out why.
- **Costs, borrow, gross/net.** Costs are one-way × NAV per leg; the long/short timer
  charges both legs and the **short SPY leg pays borrow** (the bench's recurring
  self-inflicted wounds are double-charged financing and a short that borrows for free);
  gross and net are labelled separately at every horizon.
- **Survivorship named on the Signal axis.** The basket holds only *currently-listed*
  brand owners; two are recent spin-offs that enter only post-listing (coverage tracked
  per year, never back-filled), and a full-history 3-name core basket cross-checks that
  the spin-offs don't drive the result — the caveat travels with the stamp.

## Sibling seasonality / sentiment studies (the dedup map — what this is NOT)

- **"Sell in May" / Halloween indicator** (Bouman & Jacobsen 2002, *American Economic
  Review*, "The Halloween Indicator, 'Sell in May and Go Away'") — a *market-wide*
  calendar seasonal. This study is the opposite polarity on a *single demand-driven
  sector*: a spring **long** in allergy names, not a summer market exit.
- **Weather/mood seasonals** — Kamstra, Kramer & Levi (2003, *AER*, seasonal-affective
  SAD effect) and Hirshleifer & Shumway (2003, *Journal of Finance*, sunshine) move
  *sentiment*; the pollen claim is a *fundamental demand* seasonal (real unit sales),
  a different mechanism entirely.
- On this desk, [`708-eurovision-effect`](../../708-eurovision-effect/) and
  [`707-plane-crash-effect`](../../707-plane-crash-effect/) share the exact
  event-study machinery (one-sample *t* across independent events + random placebo +
  costed timer + synthetic control), and [`358-watch-index`](../../358-watch-index/)
  shares the labelled-proxy-calendar pattern used for the pollen window.

## Data sources

- **BAYRY / SNY / PRGO / KVUE / HLN** and benchmarks **SPY / XLP** — daily total-return
  closes (`auto_adjust=True`), yfinance (no key), cached under `_cache/`
  (`pollen_<ticker>.csv`), 1996-01-02 → 2026-06-30.
- **The pollen-season window** — the hardcoded calendar rule in
  [`pollen_season/data.py`](../pollen_season/data.py) (`SEASON_START_MMDD`,
  `SEASON_END_MMDD`), sourced to AAFA/AAAAI as above.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).
