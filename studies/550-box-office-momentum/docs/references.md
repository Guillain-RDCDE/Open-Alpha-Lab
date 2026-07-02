# References & literature map — Study 550 (Box-Office-Momentum)

## The claim, at full strength

- **Alt-data folklore / practitioner claim.** Weekend box-office receipts are pitched as a
  real-time **consumer-sentiment leading indicator**: strong theatrical weekends signal confident,
  spending consumers, which should foreshadow gains for media/studio stocks and even the broad tape.
  The tradable version is a *box-office momentum* signal (this weekend's gross vs its trailing norm)
  feeding a predictive regression on forward returns. This study steelmans that claim and tests it.
- **Box Office Mojo / The Numbers** — the public weekend-gross series the folklore points to. There
  is no free, rate-limit-friendly, survivorship-honest API for the history, and the series carries
  definitional breaks (3-day vs 4-day weekends, holiday shifts, the 2020-21 shutdown, the
  streaming-era decline) — the data-availability limitation named on the SIGNAL axis.
- **MPAA / MPA Theatrical & Home Entertainment Market reports (2015-2023)** — the qualitative arc
  (plateau → pandemic collapse → partial recovery → structural decline) the curated illustrative
  index in [`data.py`](../box_office_momentum/data.py) traces (levels are stylised, never fitted).

## Why alt-data "sentiment leads returns" claims usually fail the honest test

- **Da, Engelberg & Gao (2015)**, *"The Sum of All FEARS: Investor Sentiment and Asset Prices,"*
  *Review of Financial Studies* 28(1). Search-based sentiment indices *look* predictive but the
  effect is fragile once common macro/market factors are controlled — the template for how a
  co-moving alt-data series manufactures a spurious lead.
- **Tetlock (2007)**, *"Giving Content to Investor Sentiment: The Role of Media in the Stock
  Market,"* *Journal of Finance* 62(3). Media/sentiment content correlates with returns
  contemporaneously; the *predictive* content is small and mostly reverses — the co-movement-vs-lead
  distinction this study makes.
- **Novy-Marx (2014)**, *"Predicting Anomaly Performance with Politics, the Weather, Global Warming,
  Sunspots, and the Stars,"* *Review of Financial Studies*. The canonical warning that *any*
  plausible external series will regress "significantly" on returns by chance/confound — the reason
  this study insists on a confound control plus a structural (not merely placebo) test.
- **Common-factor confounding.** Two series that both load on a persistent aggregate factor
  (consumer confidence / the market) will show a spurious cross-lag; the honest fix is to control for
  the contemporaneous factor, which this study does (media slope-*t* falls +1.53 → +0.65).

## Neighbours on this bench (the dedup map)

- **[Study 257 — AAII-Sentiment](../../257-aaii-sentiment/)** / **[Study 300 —
  Sports-Sentiment](../../300-sports-sentiment/)** / **[Study 335 — Buzz-Sentiment-ETF](../../335-buzz-sentiment-etf/)**
  / **[Study 392 — Glassdoor-Sentiment](../../392-glassdoor-sentiment/)** — the sentiment/alt-data
  leading-indicator family. Study 550 is the **box-office** variant, specifically framed around the
  common-consumer-factor confound and a synthetic-only, survivorship caveat.
- **[Study 271 — Cardboard-Box](../../271-cardboard-box/)** / **[Study 269 —
  Baltic-Dry](../../269-baltic-dry/)** — real-economy "leading indicator" series vs the tape; the
  same trap (a macro-co-moving series looks predictive) in a different alt-data dress.
- **[Study 273 — Lego-Returns](../../273-lego-returns/)** / **[Study 275 —
  Whisky-Cask](../../275-whisky-cask/)** / **[Study 276 — Sneaker-Resale](../../276-sneaker-resale/)**
  — the synthetic-only / curated-index alt-data studies whose data-availability caveat and
  positive-control discipline this study matches.

## Shared method

- **Predictive regression + circular-shift placebo** (Politis & Romano 1994, the stationary/circular
  bootstrap idea) — the block-preserving null used here; the study also shows *why a placebo alone is
  not enough* when a persistent common factor is present.
- **Confound-controlled OLS** — adding the contemporaneous market return to separate a genuine lead
  from shared-factor co-movement.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a real signal
  needs *t* ≥ 2 on a **real** tape; synthetic-only is capped at `WEAK`/`NONE`), one execution lag
  (the signal at week *t* is entered at that close and held over week *t+1*), costs one-way × NAV,
  and the seed-robust (≥ 20 seeds) synthetic positive control.
