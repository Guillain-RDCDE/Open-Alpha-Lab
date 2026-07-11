# References & literature map — Study 647 (PBoC RRR Effect)

## The claim under test

- **The folklore.** "Chinese equities pop when the PBoC cuts the Reserve Requirement Ratio" —
  a staple of financial-press coverage every time the People's Bank of China acts: freeing up
  bank lending capacity system-wide is read as a "stimulus" green light, and headlines like
  *"China stocks rally after PBoC cuts reserve ratio"* recur across Reuters, Bloomberg and
  CNBC after nearly every cut since 2015. Unlike the FOMC-vol-crush or BoJ-announcement-effect
  siblings, this claim's anchor is mostly **practitioner narrative, not a peer-reviewed
  event-study literature** — worth naming honestly, because it means the claim enters this
  desk with a thinner presumption of truth than a claim with a dedicated academic paper behind
  it.
- **The closest academic anchor.** Fernald, Spiegel & Swanson (2014, *Monetary Policy
  Effectiveness in China*, Journal of International Money and Finance) study the transmission
  of Chinese monetary-policy tools — including the RRR — into activity and prices; it is about
  macro transmission lags (months), not a same-day equity event-study, so it does not directly
  test (or refute) the "pop on the day" claim this study runs.
- **The mirror claim.** If cuts are stimulus (bullish), by the same logic hikes should be
  tightening (bearish). The folklore rarely states this half explicitly, but it's the natural
  falsifiable complement — and this study tests it directly (cuts vs hikes) rather than taking
  the cut-only framing at face value.

## What we measure, and the honesty rails

- **FXI/MCHI log return on the RRR announcement's mapped trading day**, cut and hike split
  separately, vs every other trading day. Welch *t* (Welch 1947) is the planned primary; a
  **Newey-West (1987)** 5-lag *t* on the event-day dummy regression is the autocorrelation-robust
  cross-check.
- **Broad-based moves only.** The PBoC also runs frequent *targeted/structural* RRR relief
  (carve-outs for rural credit cooperatives, inclusive-finance lending programmes) that isn't
  what "stimulus rally" commentary is about — those are excluded by construction, the same
  filtering logic sibling study 637-fomc-vol-crush applies to scheduled-vs-emergency FOMC
  actions (there: temporal filter; here: scope filter).
- **One-sided placebo, pre-committed sign.** The claim predicts a direction (cuts up, hikes
  down), so the random-calendar placebo (20 seeds × 1,000 draws) tests the corresponding tail
  — right for cuts, left for hikes — rather than a two-sided test that would understate how
  specific the claim actually is.
- **Cuts vs hikes, directly.** The single most decisive test on this desk's own terms: if
  "cut = pop" is a real, direction-driven mechanism, cut-day and hike-day returns should differ
  by a lot. A Welch *t* of the difference answers that without leaning on either side's noisy
  small-sample split.
- **Era split (2015-01-01) is justified, not snooped:** the RRR's last hike was June 2011; the
  2008-2012 cuts were acute crisis pivots (GFC, Euro-crisis panic), while every cut from 2015
  onward has been well telegraphed days ahead by State Council meetings — a genuine regime
  change in *how* cuts arrive, tested as a difference (not eyeballed).

## Why the tradable echo is graded separately

- **FXI's structure is worth naming.** iShares' FXI has historically combined direct H-share/
  ADR holdings with total-return swaps to replicate mainland A-share exposure under China's
  QFII quota system — a structure that has, at points in its history, produced meaningful
  premium/discount and tracking noise relative to NAV, independent of whatever the event-study
  measures. MCHI (a broader A/H/ADR blend, inception 2011) is used throughout as a
  cross-check specifically to catch results that are an artefact of FXI's own construction
  rather than genuine to "Chinese equities."
- Costs are charged one-way × NAV per leg (5 bps; a "buy the cut" round trip = 2 legs); the
  entry is the prior close relative to the announcement's mapped trading day (see the
  execution-lag convention in `docs/results.md` — zero look-ahead by construction, Beijing's
  12-13-hour lead on New York means the news is already public before the mapped US session
  opens).

## Data sources

- **FXI and MCHI daily raw OHLC + adjusted closes** — yfinance (no key), cached under
  `_cache/` (`pboc_fxi.csv`, `pboc_mchi.csv`), 2008-01-02 → 2026-06-30 (MCHI from its
  2011-03-29 inception).
- **48 hardcoded broad-based PBoC RRR announcements, 2008 → 2025** (31 cuts, 17 hikes), in
  [`data.py`](../pboc_rrr_effect/data.py). Compiled from the PBoC's official announcement
  archive (pbc.gov.cn, "Required Reserves") cross-checked against contemporaneous Reuters/
  Xinhua/Caixin coverage and the CEIC / Wikipedia RRR-level time series. Named honestly: the
  2008-2012 multi-category cuts carry more day-level uncertainty than the independently
  double-sourced 2015-2025 dates (see the data-quirk note in `docs/results.md`); no single
  date drives the headline verdict.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [620-a-h-premium](../620-a-h-premium/) — the **structural price gap** between a Chinese
  company's Shanghai (A) and Hong Kong (H) share lines. A cross-sectional valuation anomaly
  with no event calendar at all; this study is a **time-series event-study around a monetary-
  policy calendar**. No overlap in claim, mechanism or instrument.
- [313-geopolitical-shock](../313-geopolitical-shock/) — the closest methodological cousin: a
  hardcoded-calendar event-study on SPY, with a placebo, a block-bootstrap and a synthetic
  positive control, same shared skeleton (`quantlab`-style event-study machinery). But the
  claim, the market and the calendar are entirely different — wars/attacks on the S&P 500, not
  PBoC RRR moves on Chinese equities. Worth naming because both studies land on the same
  honest conclusion (`None x Mirage`) via the same kind of test, which is itself a small piece
  of evidence that this shared protocol isn't manufacturing false positives by construction.

No sibling study tests what the **PBoC RRR calendar does to Chinese equities** — this study's
own axis.
