# References & literature map — Study 742 (Friday-17th / *Venerdì 17*)

## The claim under test

- **The folklore.** In Italy the unlucky day is **Friday the 17th**, not the 13th
  (*Venerdì 17*; the fear itself is *eptacaidecafobia*). The origin is Roman: the numeral
  for 17, **`XVII`**, is an anagram of the Latin **`VIXI`** — "I have lived", i.e. "I am
  dead / my life is over" — a formula found on Roman tombstones. The superstition is live
  and mainstream: Italian airlines (Alitalia historically) skipped seat row 17, buildings
  skip the 17th floor, and the 2000 Renault Vel Satis was rebranded away from "17" for the
  Italian market. The market claim is the folk-finance corollary: a whole country in a
  faintly darker mood on Venerdì 17 should leave the FTSE MIB trading weak that day.
- **The academic anchor (a different day, a different index, a different era).**
  Kolb & Rodriguez (1987, *"Friday the Thirteenth: 'Part VII' — A Note"*, Journal of
  Finance 42(5), 1385–1387) reported a small **negative** Friday-**13th** anomaly in the
  Dow Jones over 1940–1987. Lucey (2000, *"Friday the 13th and the Philosophical Basis of
  Financial Economics"*, Applied Economics Letters 7(11), 727–730; and 2001 follow-ups
  across international indices) finds the effect fragile and largely absent out of sample.
  The desk's own [study 163](../../163-friday-13th/) finds **nothing** on the S&P 500 —
  Friday the 13th is, if anything, a slightly *above*-average day. There is **no published
  study of a Venerdì-17 stock effect** that we are aware of: this is pure superstition /
  folk-finance, tested here for the first time on the Italian tape. The desk therefore
  starts with a low prior — the closest real anchor (Kolb–Rodriguez) is itself unreplicated
  and concerns a *different* unlucky day.

## What we measure, and the honesty rails

- **The calendar is pure date arithmetic** (`data.is_friday_17th`): a date is a Venerdì 17
  iff `weekday() == 4` (Friday) and `day == 17`. No hardcoded table, no fetch — deterministic
  and independently verifiable. ~1.7 events per year → 49 events on the MIB tape (1998→2026),
  52 on the longer EWI tape (1996→2026).
- **One documented execution convention, and it is *no lag at all*.** The label "today is
  Venerdì 17" is a calendar fact known **before the open**, so there is no estimation or
  announcement lag to apply (calendar-known rules need none — see METHODOLOGY.md, *"One
  execution lag, documented exactly"*). The outcome is that session's close-to-close
  log-return. The tradability short is therefore established at the **prior** close and
  covered at the 17th's close — zero look-ahead by construction.
- **Inference unit — the event, not the day.** Each Venerdì 17 is one independent,
  non-overlapping calendar date. The correct primary statistic is a **one-sample *t*** of
  the per-event return (like study 163's per-event test, and the 707/708 event studies), not
  a daily panel regression — there is no within-event serial correlation to cluster once each
  event is one number. The matched control is **all other Fridays** (a Welch contrast holds
  any generic Friday-of-the-week effect fixed), and the **random-calendar placebo** redraws
  many random other-Friday sets from the same tape to ask whether the observed mean sits
  outside the ordinary Friday-to-Friday noise.
- **Look-elsewhere, corrected.** The 17th was chosen by folklore, not pre-registration, so a
  snooper testing the neighbouring "middle Friday" slots (17 ± 7k = {3, 10, 17, 24, 31}) and
  quoting the most extreme would inflate the false-positive rate fivefold. `strategy.dom_sweep`
  reports raw **and** Bonferroni-corrected (k = 5) *p*-values — the same kill shot study 163
  applies to {6, 13, 20, 27}.
- **Two instruments, both labelled.** `FTSEMIB.MI` is the FTSE MIB index in **EUR,
  price-only** (no dividend adjustment on Yahoo's index series) — the purest read of Italian
  sentiment, priced by Italians in euro. `EWI` (iShares MSCI Italy) is **USD, total-return** —
  the vehicle a foreigner could actually trade, but USD-denominated, so it blends the Italian
  tape with the EUR/USD cross. Price-only vs total-return, EUR vs USD, are carried into every
  table; neither is ever sold under the other's banner (METHODOLOGY.md, *"Gross is labeled
  gross…"*).
- **Costs one-way × NAV, shorts pay borrow.** The tradability timer shorts the unlucky day;
  one round trip charges the one-way cost twice against NAV **plus** one day of short borrow
  (breakeven 12 bps). Gross and net are both reported.

## Why the FTSE MIB (and why also EWI)

A superstition about *Italian* mood should show up first on the *Italian* tape, in the
currency Italians price in — hence the FTSE MIB (EUR) is the primary instrument, not a
USD proxy. But the FTSE MIB is not directly buyable by a US-based retail investor, and Yahoo
carries it only as a price-only index; so `EWI` (USD, total-return) is included as the
tradable cross-check and as the surface the "could you trade it?" short runs on. The two
disagreeing only in the third decimal (both faintly green, both insignificant) is itself the
robustness result: the non-effect is not an artefact of one instrument's quirks.

## Data sources

- **Daily closes** for `FTSEMIB.MI` (FTSE MIB, EUR, price-only index) and `EWI` (iShares
  MSCI Italy, USD, total-return) — yfinance (no key), cached under `_cache/`.
- **Venerdì-17 dates** — derived by pure date arithmetic in
  [`data.py`](../friday_17th/data.py); no external table.
- **The superstition** — general reference: Wikipedia, "17 (number) § Cultural
  significance" and "Heptadecaphobia / *eptacaidecafobia*"; the `XVII` → `VIXI` anagram is
  the standard etymology.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [163-friday-13th](../../163-friday-13th/) — the direct Anglo cousin: Friday the **13th**
  on the **S&P 500**, HAC *t* + Friday-6th placebo + a {6,13,20,27} Bonferroni sweep. Same
  shape, a different unlucky day and a different market. Study 742 is its Latin twin: the
  **17th** on the **FTSE MIB**, in EUR. Both land None/Busted, both faintly green — the
  headline cross-study finding.
- [158-super-bowl](../../158-super-bowl/) — the "Super Bowl Indicator" on the S&P: a folklore
  calendar signal tested with a permutation null. Different trigger (a game outcome, not a
  fixed date), same honesty rails.
- [608-friday-news-dump](../../608-friday-news-dump/) — a *Friday*-specific effect, but about
  information timing (news released late Friday), not a superstition about a day-of-month.
- [707-plane-crash-effect](../../707-plane-crash-effect/) / [708-eurovision-effect](../../708-eurovision-effect/)
  — the sentiment-**event** siblings (one-sample *t* + random-calendar placebo + costed
  timer), but keyed to irregular real-world events, not a deterministic calendar slot.

None of the siblings test the **Italian 17th on the Italian tape** — the Venerdì-17 angle,
and the two-superstitions-two-countries-one-answer comparison with study 163, is this study's
own contribution.
