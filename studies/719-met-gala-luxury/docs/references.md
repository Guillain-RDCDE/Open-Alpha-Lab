# References & literature map — Study 719 (Met-Gala-Luxury)

## The claim under test

- **The folklore.** The Met Gala (the Costume Institute Benefit at the Metropolitan
  Museum of Art) is "fashion's Super Bowl": a single first-Monday-in-May night on which
  the world's cameras point at exactly the brands owned by four listed European houses —
  LVMH (Dior, Louis Vuitton, Fendi, Loewe…), Kering (Gucci, Saint Laurent, Balenciaga,
  Bottega Veneta), Hermès, and Richemont (Cartier, Van Cleef & Arpels). Markets-meet-
  culture pieces argue that the enormous, free brand exposure ought to lift the luxury
  complex in the days around the event.
- **The academic cousin (a different trigger, a real effect).** There is no peer-reviewed
  study of a "Met Gala stock effect." The nearest published anchors are the
  sentiment-and-attention literature: Edmans, García & Norli (2007, *Sports Sentiment and
  Stock Returns*, Journal of Finance) show national markets fall after World-Cup
  elimination — mood genuinely moves prices, identified off a *loss* shock; Da, Engelberg
  & Gao (2011, *In Search of Attention*, Journal of Finance) show retail attention
  (Google search volume) predicts short-run price pressure; Barber & Odean (2008, *All
  That Glitters*, Review of Financial Studies) show attention-grabbing events drive
  retail buying. The Met Gala folklore borrows this "attention is priced" mechanism and
  swaps in a *scheduled, anticipated* advertisement — which efficient-markets logic says
  should already be in the price (Fama 1970).
- **What nobody has published.** No formal test of a Met-Gala luxury effect exists that
  we are aware of — this is pure media/social folklore, not a tested academic claim. The
  desk therefore starts with a low prior, and a *predictable* calendar-known attention
  event is precisely the kind that should produce **no** abnormal return.

## What we measure, and the honesty rails

- **The calendar is hardcoded** (`data.py`, `EVENTS`) from Wikipedia's "Met Gala" article
  and the Costume Institute per-year coverage, with each exact Monday date. Three years
  had **no gala** (2000 — planned Chanel exhibition cancelled; 2002 — cancelled after
  9/11; 2020 — COVID-19), and **2021 was held in September** (2021-09-13) as the make-up
  for 2020 — all named quirks, not smoothed over. The first-Monday-in-May slot firmly
  holds from 2005 onward; 2001/2003/2004 were April galas and predate the convention.
- **Selection, named on the Signal axis.** The `VGK` benchmark's inception (2005-03-10)
  is the binding floor: of the 23 held galas, **20** fall inside the VGK window and are
  tested (2005–2025, excluding COVID-cancelled 2020). This floor coincides with the first
  reliably-first-Monday year, so the tested sample is exactly the "modern, calendar-
  regular" Met Gala — disclosed, not patched with a pre-2005 index proxy.
- **One documented execution lag.** The gala runs **Monday evening in New York** (~7pm ET
  red carpet), while Euronext Paris and SIX Swiss — where all four names list — close
  ~17:30 CET, ~7–8 hours *earlier*. So day(-1) = the gala-Monday European close (cannot
  yet know the gala); day(0) = the next European close (first to reflect it). The
  **signal** measurement runs day(-1)→day(-1)+k (the full reaction, including the
  un-tradable overnight jump); the **tradability** measurement enters at day(0)'s close —
  zero look-ahead by construction.
- **Inference unit.** Each gala year is one independent, non-overlapping event — the
  correct test is a **one-sample t** of the abnormal return across events (like study
  708's per-edition t-test), not a daily panel regression. A random-window placebo
  (drawing many non-gala k-session windows from the *same* basket) checks whether the
  observed mean sits outside the basket's own ordinary tracking noise against VGK — here
  it sits dead centre, which is exactly the kind of null the desk exists to surface.
- **Basket construction.** Equal-weighted daily returns of the four names, requiring all
  four to trade that day (`dropna(how="any")`), so a single-exchange holiday (May 1
  Labour Day shuts Euronext, not SIX) can never let one name masquerade as the basket and
  distort an anchor. Total-return (dividends reinvested) on both the basket and VGK — the
  honest like-for-like, price-only vs total-return never mixed.

## Why VGK, not FEZ or the CAC 40

FEZ (Euro Stoxx 50) is Eurozone-only and the CAC 40 (`^FCHI`) is France-only and
price-only; Richemont is Swiss (non-euro) and the basket is a France+Switzerland mix. VGK
(Vanguard FTSE Europe) spans euro *and* non-euro Europe on a total-return basis and is the
only benchmark that is a fair counterfactual for every name in the basket — the same
choice, for the same reason, as study 708. VGK's 2005 inception is a real constraint,
disclosed rather than papered over.

## Data sources

- **Daily adjusted (total-return) closes** for `MC.PA` (LVMH), `KER.PA` (Kering),
  `RMS.PA` (Hermès), `CFR.SW` (Richemont) and the `VGK` Europe benchmark — yfinance (no
  key), cached under `_cache/`.
- **Met Gala dates, 2000→2025** — hardcoded in
  [`data.py`](../met_gala_luxury/data.py). Source: Wikipedia, "Met Gala"
  (https://en.wikipedia.org/wiki/Met_Gala), cross-checked per year for the exact Monday
  date and for the 2000/2002/2020 cancellations and the September-2021 make-up edition.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [708-eurovision-effect](../../708-eurovision-effect/) — a per-country abnormal-return
  panel keyed to a cultural-contest night (does a country's market pop when it wins/hosts
  Eurovision). Same event-study machinery (hardcoded calendar, VGK benchmark, one-sample
  *t*, random-window placebo, synthetic control) — different trigger (a contest result vs
  a fashion advertisement) and different instrument (single-country ETFs vs a luxury-
  sector basket). This study is the sector-basket sibling.
- [235-world-cup-effect](../../235-world-cup-effect/) — the Edmans-style national-
  sentiment effect for football World Cup windows on the S&P 500: a single market against
  a single tournament window, not a sector basket keyed to a scheduled advertisement.
- [158-super-bowl](../../158-super-bowl/) — the "Super Bowl Indicator" on the S&P 500: a
  folklore calendar signal on a single national index, no sector panel.
- [707-plane-crash-effect](../../707-plane-crash-effect/) — a sentiment *shock* (aviation
  disasters) with the opposite sign and an unscheduled trigger; the mirror-image case to a
  fully-anticipated calendar advertisement.

None of the siblings test a **luxury-sector abnormal-return panel keyed to a fashion
event** — the Met-Gala angle, including the "a predictable advertisement is already in the
price" null, is this study's own contribution.
