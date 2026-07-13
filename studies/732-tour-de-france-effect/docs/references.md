# References & literature map — Study 732 (Tour-de-France-Effect)

## The claim under test

- **The folklore.** French equities enjoy a July "Grande Boucle" seasonal — a feel-good,
  summer-holiday bump while the whole country follows the Tour de France from the Grand
  Départ to the Champs-Élysées. It is the summer-holiday, home-crowd cousin of
  sports-sentiment folklore, recast as a three-week *calendar window* on the CAC 40 / EWQ
  rather than a surprise result. Like most such seasonals it circulates as financial-media
  / social-media colour ("markets take a summer break while France watches the bikes"),
  not as a tested academic claim.
- **The academic anchor (a different sport, a real effect).** Edmans, García & Norli
  (2007, *Sports Sentiment and Stock Returns*, Journal of Finance 62(4):1967–1998) find a
  robust next-day national-market decline after a country is **eliminated** from soccer's
  World Cup — a genuine mood-to-market channel, identified off a *loss* shock in the sport
  with the deepest national following. Ashton, Gerrard & Hudson (2003, *Economic impact of
  national sporting success on the London Stock Exchange*, Applied Economics Letters
  10:783–785) find a same-day England-football-result effect on the FTSE. The Tour borrows
  the *mood → risk-appetite* mechanism but swaps the trigger for a **calendar window**
  (a three-week festival with no win/loss shock for "France" as a whole) — a much weaker
  a-priori channel.
- **The confounding seasonal it collides with — "Sell in May".** The Tour runs in July,
  inside the best-documented calendar *weakness* in equities. Bouman & Jacobsen (2002,
  *The Halloween Indicator, "Sell in May and Go Away": Another Puzzle*, American Economic
  Review 92(5):1618–1635) document systematically lower May–October returns across 37
  markets, France included. Any "buy French stocks during the Tour" seasonal is therefore
  running straight into a known summer headwind, and a raw July bump has to be measured
  *against Europe* (an abnormal return) before it can be called French at all — the crux
  of this study's third axis.
- **What nobody has published.** There is no peer-reviewed study of a Tour de France stock
  effect that we are aware of — this is pure folklore. As with study 708 (Eurovision), the
  desk starts with a low prior: a real mechanism (sports sentiment) borrowed from a much
  bigger, shock-driven setting and stretched onto a gentle three-week calendar festival.

## What we measure, and the honesty rails

- **The calendar is hardcoded** (`data.py`, `EVENTS`) from Wikipedia's "List of Tour de
  France editions," cross-checked per year for the exact Grand Départ and final-stage
  dates. Two named quirks: **2020** was pushed by COVID-19 from July to Aug 29 → Sep 20
  (measured on the actual race dates — a natural "race vs calendar month" probe), and
  **2024** finished in Nice rather than Paris because of the Olympics (no market effect).
- **One execution convention, and it needs no information lag.** Unlike a surprise
  announcement (study 708's Eurovision winner drops on a non-trading Saturday night), the
  Tour dates are public a *year* in advance. This is a **calendar-known window**: a
  believer can be positioned at the last close before the Grand Départ with zero
  look-ahead, and there is no un-tradable weekend jump to strip out. Entry = last close
  before the Grand Départ; exit = first close on/after the final stage (the Champs-Élysées
  Sunday is non-trading). Signal = gross window return; Tradability = the same window net
  of 2× one-way cost × NAV.
- **Raw vs abnormal, and price-only vs total-return, labelled.** `EWQ` (iShares MSCI
  France) and `VGK` (Vanguard FTSE Europe) are **total-return** (dividends reinvested);
  `^FCHI` (CAC 40) is a **price index** with no dividend reinvestment, used only as a
  long-history cross-check on the raw seasonal and never mixed into the total-return
  abnormal test. The **raw** seasonal (EWQ during the Tour) is what a believer earns; the
  **abnormal** (EWQ − VGK) is the only cut that can separate a genuine *French* sentiment
  bump from ordinary pan-European summer beta — and it is the cut that decides the verdict.
- **Inference unit.** Each Tour edition is one independent, non-overlapping annual event —
  the correct test is a **one-sample t** of the window return across editions, not a daily
  panel regression. A random-window placebo (drawing many same-length windows from *else-
  where* in EWQ's own history) checks whether the July-Tour window is unusual or just an
  ordinary three weeks; here it is ordinary (left-tail *p* = 0.167), exactly the kind of
  non-event the desk exists to document.

## Why VGK, not FEZ

The abnormal test needs a Europe benchmark that a French index can be measured *against*.
FEZ (Euro Stoxx 50) is Eurozone-only; VGK (Vanguard FTSE Europe) spans euro and non-euro
Europe and is the broader, fairer counterfactual for "is France doing anything Europe as a
whole is not." VGK's inception (2005-03-10) caps the abnormal test to 2005 → 2025 (21
editions); the raw EWQ seasonal and the CAC price cross-check reach back to EWQ's 1996
inception — a real constraint, disclosed rather than patched.

## Data sources

- **Daily closes** for `EWQ` (total-return), `^FCHI` (CAC 40, price-only) and `VGK`
  (total-return) — yfinance (no key), cached under `_cache/`.
- **Tour de France Grand Départ and final-stage dates, 1996→2025** — hardcoded in
  [`data.py`](../tour_de_france_effect/data.py). Sources: Wikipedia, "List of Tour de
  France editions" (https://en.wikipedia.org/wiki/List_of_Tour_de_France_editions) and the
  per-year "<YEAR> Tour de France" pages for the exact Grand Départ / final-stage dates.
- All headline numbers are pinned in [`results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [708-eurovision-effect](../../708-eurovision-effect/) — the same "national feel-good
  bump" folklore, but keyed to a **surprise result** (the Eurovision winner) with a real
  information lag, across a per-country ETF panel. This study is a **calendar window** on a
  **single country** (France) with no surprise and no lag — a cleaner seasonal, and an even
  emptier result.
- [235-world-cup-effect](../../235-world-cup-effect/) — the Edmans-style national-sentiment
  effect for **football World Cup** windows (the closest real academic anchor to the
  mechanism), tested on the S&P 500. A win/loss shock, not a three-week festival window.
- [158-super-bowl](../../158-super-bowl/) and
  [709-world-series-effect](../../709-world-series-effect/) — US single-sport folklore
  indicators on a single national index. Different sport, different market; no summer-beta
  confound.
- **Any "Sell in May" / Halloween-indicator study on the bench** — this study is the mirror
  image: it asks whether a July window is *special for France*, and finds only the ordinary
  pan-European summer that "Sell in May" already describes. The Tour angle — a calendar
  sports-festival window collapsing into plain summer beta once you net out Europe — is
  this study's own contribution.
