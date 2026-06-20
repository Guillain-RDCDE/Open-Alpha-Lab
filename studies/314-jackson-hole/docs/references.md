# References & literature map — Study 314 (Jackson-Hole)

## The claim under test

- **The "Jackson Hole drift / Jackson Hole effect."** A piece of market folklore that
  the S&P 500 drifts predictably around the Kansas City Fed's late-August **Jackson Hole
  Economic Symposium** — either anticipating a dovish keynote in the run-up, or trending
  after the Fed Chair's Friday-morning speech. The belief is seeded by a few unusually
  market-moving addresses (see below), not by a documented, replicated calendar anomaly.
  We make it falsifiable as an event study and test whether one observation per year can
  clear an inference bar.

## The event itself

- **Kansas City Fed, *Jackson Hole Economic Symposium* archive**
  (kansascityfed.org/research/jackson-hole-economic-symposium/). The conference has been
  held in Grand Teton National Park every late August since 1978; the headline event is
  the **opening keynote by the Fed Chair**, traditionally Friday morning. The symposium is
  **not** an FOMC meeting and carries **no policy decision** — it is a speech-driven media
  event, which is exactly why a "drift" around it is a priori suspect.
- **The memorable keynotes that seeded the belief.** Ben Bernanke, *The Economic Outlook
  and Monetary Policy* (Aug 27, 2010) — widely read as foreshadowing QE2, and followed by
  a strong equity rally. Jerome Powell, *Monetary Policy and Price Stability* (Aug 26,
  2022) — an eight-minute hawkish address that sent the S&P down ~3.4% on the day. These
  two salient, opposite-signed reactions are the anecdotal basis of the folk effect.

## Why an annual single-event calendar is treacherous

- **Multiple testing across the event window.** Harvey, Liu & Zhu (2016),
  *…and the Cross-Section of Expected Returns* (Review of Financial Studies) — the
  multiple-comparisons problem. Sweeping eleven offsets (−5…+5) around one date and
  quoting the largest is precisely the data-snooping the paper warns against; the maximum
  HAC *t* here (+1.75) is the max-over-offsets, not an honest single test.
- **Thin samples and over-fitting calendar dates.** Sullivan, Timmermann & White (2001),
  *Dangers of Data Mining: The Case of Calendar Effects in Stock Returns* (Journal of
  Econometrics) — calendar anomalies routinely vanish once corrected for the search over
  many candidate dates. A once-a-year event yields ~33 observations across the SPY era;
  the standard error swamps any plausible point estimate.
- **Event-study methodology.** MacKinlay (1997), *Event Studies in Economics and
  Finance* (Journal of Economic Literature); Brown & Warner (1985), *Using Daily Stock
  Returns* (Journal of Financial Economics) — the abnormal-return-around-an-event design
  this study implements, and its well-known low power on small event counts.

## The real Fed-related effects this is NOT

- **Pre-FOMC announcement drift.** Lucca & Moench (2015), *The Pre-FOMC Announcement
  Drift* (Journal of Finance) — equities drift up in the ~24h before *scheduled FOMC
  statements* (eight per year). Tested on this desk as **[Study 67 — Fed-Drift](../../67-fed-drift/)**.
  Jackson Hole is a different, single, annual, non-FOMC event.
- **The FOMC even-week cycle.** Cieslak, Morse & Vuolteenaho (2019), *Stock Returns over
  the FOMC Cycle* (Journal of Finance) — returns accrue in even weeks of the ~6-week
  inter-meeting cycle. Tested as **[Study 135 — FOMC-Cycle](../../135-fomc-cycle/)**.
  Again distinct: that is a meeting-cycle effect, not a symposium-day event.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy._hac_t`](../jackson_hole/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1992), *A circular block-resampling
  procedure for stationary data* — the CI on the event-day mean
  ([`strategy.block_bootstrap_ci`](../jackson_hole/strategy.py)), which respects any
  cross-year clustering instead of assuming i.i.d. events.
- **Post-publication decay framing.** McLean & Pontiff (2016), *Does Academic Research
  Destroy Stock Return Predictability?* (Journal of Finance) — the lens the desk applies
  to every published calendar effect.

## Data sources used here

- **Shared SPY daily total returns** (`_cache/last_call_spy.parquet`, the desk's
  `last_call_spy` pull via `yfinance`, auto-adjusted), 1993-02-01 → 2026-06-11. Same tape
  Study 67 uses. Pinned with an as-of date and content fingerprint (see
  [`docs/results.md`](results.md)). The offline reproducible core and tests run on the
  deterministic [`data.synthetic_world`](../jackson_hole/data.py) generator, never the
  network.

## Related desk studies

- **[Study 67 — Fed-Drift](../../67-fed-drift/)**: the pre-FOMC announcement drift — a
  *real* (now decayed) Fed-event effect, the honest cousin of this null.
- **[Study 135 — FOMC-Cycle](../../135-fomc-cycle/)**: the FOMC even-week cycle — another
  Fed-calendar anomaly, stamped Weak/Mirage.
- **[Study 287 — Easter-Effect](../../287-easter-effect/)** and the desk's wider calendar
  zoo: the same thin-sample, annual-event design — most land None/Mirage for the same
  reason this one does.
