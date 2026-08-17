# References & literature map — Study 924 (First Cut)

## The claim under test

- **The folk trade.** "When the Fed starts cutting, extend duration." It is one of the most
  repeated lines in macro strategy: the first cut of an easing cycle is said to mark the
  moment when the front end is anchored, the curve bull-steepens and long Treasuries begin a
  multi-quarter rally, so the trade is to buy TLT (or IEF) on the announcement and hold. The
  testable version — the one this study runs — is: *does duration, bought the session after
  the first cut of a cycle and held 1/3/6/12 months, beat cash by more than a random start
  date would?*
- **The steelman.** Easing cycles begin when growth is deteriorating; term premia and
  expected short rates both fall; and cuts arrive in clusters, so the first one carries
  information about the ones to follow. If any single calendar moment should pay a duration
  holder, this is the candidate.
- **The catch, stated in advance.** There have been five arguable cycle-start cuts since
  2001, of which four are measurable with a live ETF tape. That is the whole sample. No
  amount of care in the accounting fixes N = 4.

## Why the effect might be zero *before* any data

- **Only the surprise moves the curve.** Kuttner (2001), *Monetary Policy Surprises and
  Interest Rates: Evidence from the Fed Funds Futures Market*, Journal of Monetary Economics
  47(3) — decomposing announcements into anticipated and unanticipated components, yields
  respond to the *surprise* alone; the expected part is already in the price. A hardcoded
  calendar of cut dates contains no surprise measure by construction.
- **The path matters more than the level.** Gürkaynak, Sack & Swanson (2005), *Do Actions
  Speak Louder Than Words? The Response of Asset Prices to Monetary Policy Actions and
  Statements*, International Journal of Central Banking 1(1) — a second "path" factor drives
  long yields far more than the target change itself.
- **The information effect can flip the sign.** Nakamura & Steinsson (2018),
  *High-Frequency Identification of Monetary Non-Neutrality: The Information Effect*,
  Quarterly Journal of Economics 133(3) — a cut also reveals the Fed's private pessimism.
  Cieslak & Schrimpf (2019), *Non-Monetary News in Central Bank Communication*, Journal of
  International Economics, make the same point across announcement types. A cut that reads
  as "we know something bad" and a cut that reads as "we are done fighting inflation" have
  different consequences for the long end. Our 2024-09-18 event (−10.7% over 12 months, as
  long yields *rose* through a resilient economy) is exactly the second case.
- **Expected easing is already in forwards.** Under any version of the expectations
  hypothesis with a term premium — Fama & Bliss (1987), *The Information in Long-Maturity
  Forward Rates*, AER; Cochrane & Piazzesi (2005), *Bond Risk Premia*, AER — the anticipated
  path of the funds rate is embedded in today's curve, so a scheduled cut on a known date
  cannot be a free duration bid.
- **Cuts are a bad-news announcement.** Bernanke & Kuttner (2005), *What Explains the Stock
  Market's Reaction to Federal Reserve Policy?*, Journal of Finance 60(3) — the asset-price
  response to policy is dominated by what the action implies about future risk premia and
  cash flows, not by the mechanical rate change.

## The methodological problem this study is really about

- **Small-N event studies.** MacKinlay (1997), *Event Studies in Economics and Finance*,
  Journal of Economic Literature 35(1), and Kothari & Warner (2007), *Econometrics of Event
  Studies* (Handbook of Corporate Finance) — event-study test statistics rely on
  cross-sectional averaging across many, ideally independent, events. With four overlapping
  macro events sharing a single business cycle, the effective sample is closer to one or
  two, and the nominal *t* is badly oversized.
- **Randomisation inference beats a *t* here.** The placebo — random start dates matched in
  count and horizon — is a Fisher-style permutation test, the standard remedy when the
  asymptotics behind a *t* are unavailable. Bertrand, Duflo & Mullainathan (2004), *How Much
  Should We Trust Differences-in-Differences Estimates?*, Quarterly Journal of Economics
  119(1), is the canonical demonstration that serially correlated outcomes plus few
  treatment events manufacture spurious significance unless the null is simulated.
- **Overlapping windows.** Twelve-month windows on adjacent cuts share days, so per-event
  returns are not independent. This is why the study's inferential spine is the *daily*
  conditional excess series with a Newey-West *t* and a circular block bootstrap, not the
  four-observation mean.

## Related desk studies (dedup)

- **[Study 67 — Fed-Drift](../../67-fed-drift/)** and **[Study 517 — Pre-FOMC-Drift](../../517-pre-fomc-drift/)**:
  the *equity* drift into scheduled FOMC announcements (Lucca-Moench). Same calendar,
  different asset and a *pre*-announcement window; Study 924 is *post*-announcement and
  measures duration.
- **[Study 135 — FOMC-Cycle](../../135-fomc-cycle/)**: the even-week Fed-cycle pattern in
  equities (Cieslak-Morse-Vuolteenaho) — a fortnightly seasonality across *all* meetings,
  not a conditional trade on the direction of a policy action.
- **[Study 322 — FOMC-Blackout](../../322-fomc-blackout/)** and
  **[Study 637 — FOMC Vol Crush](../../637-fomc-vol-crush/)**: communications-window and
  implied-vol effects around meetings; neither conditions on a cut, and neither buys bonds.
- **[Study 647 — PBoC-RRR-Effect](../../647-pboc-rrr-effect/)**: the closest cousin in
  spirit — a hardcoded easing calendar for another central bank, on equities — and it lands
  in the same place: the market reacts to the central bank showing up, not to the direction.
- **[Study 59 — Downhill](../../59-downhill/)**, **[Study 581 — Term-Premium](../../581-term-premium/)**,
  **[Study 625 — Starting-Yield](../../625-starting-yield-bond-decade/)**: *unconditional*
  or yield-conditioned reasons to own duration (roll-down, estimated term premium, entry
  yield). Study 924 conditions on a policy *event* instead, and it is precisely the gap
  between those two that this study measures — the conditional leg earns less per day
  invested than the unconditional one.
- **[Study 826 — Treasury Duration BAB](../../826-treasury-duration-bab/)** and
  **[Study 884 — Convexity Barbell](../../884-convexity-barbell/)**: cross-sectional trades
  *within* the Treasury curve (leverage-adjusted beta, duration-matched barbell). Study 924's
  curve leg (long TLT / short SHY) is a directional-timing expression on an event date, not a
  standing structural position.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica 55(3) —
  [`strategy.newey_west_t`](../first_cut/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA
  89(428) — [`strategy.block_bootstrap_mean_ci`](../first_cut/strategy.py).
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of
  slice and content fingerprint quoted in [results.md](results.md).

## Data sources

- **TLT** (20y+ Treasuries), **IEF** (7-10y), **SHY** (1-3y), **BIL** (1-3 month T-bills,
  the cash leg) — daily **total-return** closes via `yfinance` (`auto_adjust=True`). Bond
  ETFs distribute most of their return as coupon, so a price-only series would understate
  every leg; nothing here is price-only.
- **`^IRX`** (13-week Treasury bill discount rate) — used only to build a **PROXY** cash
  accrual index for the pre-BIL era, as a cross-check on the cash leg. It is a discount
  rate, not a bond-equivalent yield, and carries no fund fee; it never produces a headline
  number.
- **The FOMC cut calendar is hand-typed** — see [results.md](results.md) for the full list
  and its truncation rule. It is the study's single non-market input and is labelled an
  ASSUMPTION wherever it appears.
- **As-of 2026-06-30**, the last complete month; the partial current month is dropped and no
  window is allowed to extend past it. TLT's 2002-07-30 inception, not the cash leg,
  is what removes the 2001-01-03 event from the sample.
- **Survivorship:** none. Every leg is a single, still-listed ETF and the event calendar is
  fixed in advance of the returns it selects, so there is no cross-sectional selection to
  bias the Signal axis. The selection risk here is of a different kind and is named on the
  Signal axis instead: *which* cuts count as "first" was chosen by hand.
