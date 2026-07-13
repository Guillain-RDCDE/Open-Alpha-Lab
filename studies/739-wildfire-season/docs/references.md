# References & literature map — Study 739 (Wildfire-Season)

## The claim under test

- **The folklore.** Every California fire season, financial media and utility-sector
  analysts revive the same idea: a major wildfire is a repeating, tradable risk event
  for the state's investor-owned utilities and property insurers — sell (or short) the
  exposed names on the ignition headline, and underweight the basket heading into the
  July→December fire window. The steelmanned version has a genuine mechanism for the
  utilities: California's doctrine of **inverse condemnation** (Cal. Const. art. I, §19;
  *Barham v. Southern Cal. Edison* 74 Cal.App.4th 744, 1999; *Pacific Bell Tel. Co. v.
  Southern Cal. Edison*, 208 Cal.App.4th 1400, 2012) holds a utility strictly liable for
  property damage its equipment causes **even without negligence** — so a fire traced to
  a utility line is a direct, large, and legally near-automatic hit to that utility's
  equity. This is not merely folklore: PG&E (`PCG`) filed for Chapter 11 in January 2019
  citing an estimated **$30 bn** of wildfire liability after the 2017 North Bay fires
  and the 2018 Camp Fire (PG&E Corp. Form 8-K, 2019-01-14; CPUC investigations), and
  Edison International (`EIX`) fell sharply when its equipment became the leading suspect
  in the January 2025 Eaton Fire.
- **The insurer corollary.** Believers extend the claim to property/casualty insurers —
  Allstate (`ALL`), Travelers (`TRV`), Mercury General (`MCY`, the most
  California-concentrated homeowner insurer of the four) and Chubb (`CB`) — on the logic
  that they carry the claims. California's 2017–2018 and 2025 fire seasons produced
  record insured losses (Aon and Munich Re catastrophe reports; the January 2025
  Los Angeles fires are among the costliest insured wildfire events on record), and the
  resulting non-renewals and FAIR-Plan stress are a standing news story.
- **The seasonal corollary.** The strongest form claims a *calendar* edge: the fire
  window itself (roughly Jul→Dec, peak burning plus autumn Santa Ana / Diablo wind
  events) should carry a systematically worse basket return — a "sell in July" for
  California risk, in the tradition of calendar-anomaly folklore.

## What this study does, and how it differs from just telling the PG&E story

The honest question for this desk is not "did PG&E's stock fall after the Camp Fire"
(it obviously did) but **"is there a systematic, same-day, basket-wide, seasonal,
tradable pattern across California fire events?"** We test the strongest tradable form —
an event study on the combined utility+insurer basket around 14 major fires, a
utility-vs-insurer leg split, a basket-vs-market extra-drop test, the Jul→Dec seasonal,
and a costed short-the-headline timer — and grade each on the desk's inference bar.

## What we measure, and the honesty rails

- **Ignition-day (day 0) abnormal basket return** — a constant-mean market model (Brown
  & Warner 1985, *Journal of Financial Economics*, "Using daily stock returns: The case
  of event studies"): the "normal" return is the sample mean, the abnormal return is the
  demeaned daily return. One-sample *t* across the 14 independent, non-overlapping event
  dates (the planned primary; events are far apart in time, so no HAC correction is
  needed the way a daily-panel regression would).
- **Event window [−1..+5]** with each offset's own one-sample *t*, read honestly as a
  **multiple-comparison** exercise — 7 offsets, so a spurious |*t*| ≥ 2 bar is expected
  by chance (offset −1 in this run, on the *wrong* side of the event, is flagged as
  exactly that).
- **The [+1..+5] "liability window"** — the study's crux: a big drop here with a flat
  day 0 is the signature of a *delayed fundamental repricing* (who-caused-it news
  landing over days), not a same-day sentiment flinch.
- **Outlier discipline.** The [+1..+5] mean carries an **event bootstrap** 95% CI
  (Efron 1979; events resampled with replacement) and a **leave-one-out jackknife** of
  the *t*-stat, because a 14-event mean dominated by two megafires is exactly the kind
  of number that looks significant until you drop one event.
- **Utility vs insurer legs** and a **paired basket-vs-SPY** extra-drop test — pairing
  removes the common market move, isolating the CA-specific reaction.
- **The seasonal test** carries a **random-6-month-window placebo** (multi-seed): a real
  fire-season gap must sit in the tail of the distribution of gaps that *any* 6-month
  slice of the calendar produces.
- **Hit rate carries a Wilson (1927) interval**; the event placebo is a 20-seed ×
  1,000-draw random-calendar null (the same falsification design as
  `707-plane-crash-effect` and `313-geopolitical-shock`).

## Why the timer is a short, and how it's costed

- The tradable overlay **shorts** the basket at the ignition-session close (fires are
  bad news for the basket, so the folklore trade is a short) and holds `hold` sessions —
  a single documented execution lag (the ignition is public before that session's
  close). Costs are one-way × NAV per leg (5 bps), one round trip per event, and the
  **short pays a borrow fee** (300 bps/yr, prorated over the hold) — the desk rule that
  shorts are not free. Gross and net are both reported, and the *median* is reported
  next to the *mean* precisely because the mean is a jackpot artifact.

## Data sources

- **EIX, PCG, ALL, TRV, MCY, CB** and **SPY** daily total-return closes
  (`auto_adjust=True`) — yfinance (no key), cached under `_cache/` (`wfs_eix.csv` …
  `wfs_spy.csv`), 2003-01-02 → 2026-06-30. PG&E's 2019 bankruptcy and dilution are
  *inside* the total-return `PCG` series, not survivored out.
- **14 hardcoded major California wildfire events, 2003 → 2025**, in
  [`wildfire_season/data.py`](../wildfire_season/data.py), each with a `utility_linked`
  cause flag. No free, machine-readable "major California wildfire index" exists, so this
  is a hand-built table cross-referenced against the **Cal Fire incident archive**, CPUC
  and utility 8-K liability disclosures, and contemporary reporting.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [707-plane-crash-effect](../../707-plane-crash-effect/) — the sibling event study
  (Kaplanski & Levy 2010 aviation-disaster sentiment): same event-study machinery, same
  random-calendar placebo, but a *sentiment* claim on a broad index, not a
  *liability/fundamental* claim on a sector basket. Wildfire is the mirror image: here
  the exposed names have genuine fundamental exposure (inverse-condemnation liability),
  and the finding is that even *that* isn't a clean same-day or seasonal tradable signal.
- [313-geopolitical-shock](../../313-geopolitical-shock/) — hardcoded shock calendar +
  random-calendar placebo on the broad market; same falsification design, different
  trigger.
- [300-sports-sentiment](../../300-sports-sentiment/) — the Edmans, Garcia & Norli
  (2007) loss effect; a pure mood channel with no fundamental exposure, the opposite end
  of the spectrum from a utility's strict-liability fire loss.

No sibling tests **what a California wildfire does to the state's utilities and property
insurers, and whether the fire season is a tradable calendar** — that is this study's
own axis.
