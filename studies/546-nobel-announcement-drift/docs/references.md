# References & literature map — Study 546 (Nobel-Announcement-Drift)

## The claim

This is **news-attention folklore**, not a peer-reviewed anomaly: the retail/press notion that
when the Nobel Prizes are announced each October, the sectors thematically tied to the *science*
prizes catch an attention bid and **drift** — pharma/biotech after **Medicine**, tech/semis after
**Physics** & **Chemistry**. There is no canonical paper claiming a tradable Nobel-announcement
sector drift; the study tests the folklore directly with a textbook event study.

## The methods this study rests on

- **Event-study methodology / cumulative abnormal returns (CARs).** Brown & Warner (1985),
  *"Using Daily Stock Returns: The Case of Event Studies,"* J. Financial Economics 14; MacKinlay
  (1997), *"Event Studies in Economics and Finance,"* J. Economic Literature 35. The market-model
  abnormal return $r_i - (\alpha_i + \beta_i r_m)$ estimated on a pre-event window, summed over a
  post-event window — exactly the CAR machinery here (SPY as the market, a trailing 120-day
  rolling beta).
- **Attention and returns.** Barber & Odean (2008), *"All That Glitters: The Effect of Attention
  and News on the Buying Behavior of Individual and Institutional Investors,"* Review of Financial
  Studies 21 — the mechanism the folklore invokes (attention-driven buying). Da, Engelberg & Gao
  (2011), *"In Search of Attention,"* J. Finance 66 — measuring attention shocks. Neither predicts
  a *persistent* post-Nobel sector drift; both are about transient attention.
- **Post-announcement drift, the real thing.** Ball & Brown (1968) and Bernard & Thomas (1989) on
  post-*earnings*-announcement drift — the genuine drift phenomenon, driven by a cash-flow surprise
  a Nobel prize does not carry. The contrast is the point.

## The event calendar

- **nobelprize.org** press-release / announcement dates. The three science prizes are announced on
  the Monday (Physiology/Medicine), Tuesday (Physics) and Wednesday (Chemistry) of the first full
  week of October. The Peace, Literature and Economics prizes are excluded — no clean sector map.
  The hardcoded ``NOBEL_DATES`` (2001-2024) are these announcement dates.

## Neighbours on this bench (the dedup map)

- **[Study 517 — Pre-FOMC-Drift](../../517-pre-fomc-drift/)** — the same event-study / placebo
  spirit on a *scheduled macro* calendar (FOMC decision days). Study 546 is a *thematic sector*
  reaction to a *cultural* calendar with no cash-flow channel.
- **[Study 158 — Super-Bowl](../../158-super-bowl/)** / **[Study 95 — Holiday-Cheer](../../95-holiday-cheer/)**
  — calendar/folklore indicators with no fundamental mechanism; the same "attention/ritual moves
  markets?" question, different ritual.
- **[Study 299 — Keynote-Drift](../../299-keynote-drift/)** / **[Study 363 — PEAD](../../363-pead-drift/)**
  / **[Study 515 — Earnings-Announcement-Premium](../../515-earnings-announcement-premium/)** — the
  *real* announcement-drift family (product keynotes, earnings). Study 546 is the folklore cousin
  where the "announcement" carries no tradable surprise.

## Shared method

- **Welch / one-sample *t*** — the CAR mean vs 0.
- **Random-relocation placebo** (Fisher 1935; Good 2005) — relocate each event to a random October
  day of the same year and ask how often a random October week matches the observed mean CAR.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust
  *t* ≥ 2 on the real tape plus a placebo null and seed-robustness), one documented execution lag,
  gross/net labelled, and the small-event-count / thematic-mapping limitations named on the Signal
  axis.
