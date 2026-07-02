# References & literature map — Study 564 (Short-Report-Event)

## The claim under test

- **The mechanics.** An **activist short-seller** (Muddy Waters, Hindenburg Research, Citron
  Research, Kerrisdale, Spruce Point, Grizzly, …) takes a short position, then *publishes* a
  detailed public report alleging fraud, accounting manipulation, channel-stuffing or gross
  over-valuation. Unlike a Schedule 13D (a legally-mandated *long* disclosure — see the cousin
  study 390), a short report is a **voluntary publication**: the firm is talking its own book,
  and the report is the moment the market learns the thesis. The 2020–2023 SPAC/EV boom produced
  a wave of these (Nikola, Lordstown, Clover, Mullen …).
- **The folklore.** Financial media frame the short report as a *sell/short signal*: the stock
  **craters** on the report and then **keeps falling** for months as the allegations sink in,
  regulators circle, and the thesis plays out — the textbook stories (Nikola → fraud charges,
  Luckin Coffee → delisting, Wirecard-era names, Valeant → collapse). The pitch is a free ride
  for anyone who shorts when the report drops.

## What the academic / practitioner literature actually finds

- **Ljungqvist & Qian (2016), *How Constraining Are Limits to Arbitrage?* (Review of Financial
  Studies).** Study small-arbitrageur short campaigns that publicise their theses: targets
  experience large, persistent price declines after the report is made public — the strongest
  steelman for a real post-report drift, but on campaigns *selected* by the arbitrageurs to be
  winnable.
- **Zhao (2020) and Brendel & Ryans (2021), on activist short-seller reports.** Document a
  significant negative announcement return and some continued drift, concentrated in
  small/illiquid, high-short-interest names — and note the sharp **reversals** (squeezes) that
  make the *average* far noisier than the median.
- **Appel & Fos (2019), *Short Selling and Activism.*** Situate activist shorts within the broader
  activism literature; the abnormal returns are real around the report but the tradable,
  outside-shorter edge is far weaker and highly skewed.
- **The short-squeeze / limits-to-arbitrage tradition** (Shleifer & Vishny 1997, *The Limits of
  Arbitrage*, JF; the 2021 GME episode). The defining risk of activist shorting is the **asymmetric
  payoff**: bounded gains (the stock can only fall to zero) against unbounded losses (a squeeze can
  multiply the position). This is the mechanism that destroys the *mean* even when the *median*
  short works — exactly what we reproduce (CVNA +318% excess).

## Why a hardcoded basket — and the biases it carries

- **No clean feed.** There is no free, deduplicated, ticker-mapped panel of *every* activist short
  report with genuine first-publication dates. We therefore use a **transparent, hardcoded table of
  32 famous campaigns** (Hindenburg, Muddy Waters, Citron and adjacent firms, 2015–2023) — and name
  the consequences on the Signal axis.
- **Selection bias runs *for* the claim.** Short reports enter folklore because they were prominent
  and often *worked* — so our basket is the *easiest* possible test of "short reports crater the
  stock." A bias that points *for* the effect and still leaves the mean *t* below the bar is
  decisive against a deployable edge. (House rule: survivorship/selection is named on the **Signal**
  axis, not buried in Tradability — see METHODOLOGY.)
- **Delisting bias runs *against* the claim.** The most spectacular short wins (Nikola, Lordstown,
  Mullen, XL Fleet, …) **delisted or were acquired** and drop from the free yfinance feed — **10 of
  32 targets are gone**, and they are disproportionately the biggest winners. So the *surviving*
  22-name tape is biased *against* the short even as the roster is biased *toward* it. Both are named
  in [`docs/results.md`](results.md).

## Why ~22 events cannot be a clean edge — the statistics

- **Small-sample inference / power.** With *k* ≈ 22 events, the standard error of a conditional-mean
  estimate is large; a violently right-skewed distribution (a few squeezes) makes the *mean* even
  harder to pin down. We test the mean excess against zero with a one-sample **Welch *t*** and,
  because *k* is tiny and skewed, with a **placebo / randomization test** — drawing the same-horizon
  excess from *random* dates on the *same targets* (Fisher's randomization logic; Efron & Tibshirani,
  *An Introduction to the Bootstrap*, 1993).
- **Mean vs median.** The **median** short works decisively (−42.7% at 6m) while the **mean**
  (−14.6%) is dragged toward zero by squeezes — the classic mean/median divergence of a skewed
  payoff. Reporting both, and the hit-rate, is the honest way to show "the short usually works but
  the average doesn't pay."
- **Multiple testing / selection on famous cases.** Harvey, Liu & Zhu (2016), *…and the
  Cross-Section of Expected Returns* (Review of Financial Studies), formalise why a pattern selected
  ex-post from memorable wins needs a far higher bar than a naive *t*-stat.

## Method lineage (the desk's shared engine)

- **Crater vs drift split.** [`strategy.report_day_moves`](../short_report_event/strategy.py) (the
  uncapturable day-0 news) vs [`strategy.event_excess_drift`](../short_report_event/strategy.py) (the
  tradable, 1-day-lagged, excess-of-SPY short leg) — the two objects "the stock craters and stays
  down" fuses.
- **Welch t + placebo p-value.** [`strategy.welch_t_vs_zero`](../short_report_event/strategy.py) and
  [`strategy.placebo_pvalue`](../short_report_event/strategy.py) — the mean excess against zero, and
  a 20,000-draw randomization null sized to the event count, drawing from the same targets to control
  for their own drift, one-sided toward the short claim (random basket at least as *negative*).
- **Short costs + borrow.** [`strategy.short_net_of_costs`](../short_report_event/strategy.py) — the
  short book pays a round-trip cost **and a punitive annual borrow**, pro-rated over the hold (a
  crashed, allegation-hit name is the hardest borrow).
- **Deterministic synthetic control.**
  [`data.synthetic_events`](../short_report_event/data.py) injects a known number of events with a
  planted *negative* post-report drift; the offline core runs with no network. Averaged over 25
  seeds, the control confirms the engine recovers a real short edge (t → −2.9 at −20%/6m, −6.1 at
  −35%) and stays flat at the null.

## Data sources used here

- **yfinance** daily adjusted closes for SPY + 22 surviving short-report targets, 2010-01-04 →
  2026-06-26, cached under `_cache/event_prices.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by re-running [`short_report_event/`](../short_report_event/).

## Related desk studies

- **[Study 390 — Activist-13D](../../390-activist-13d/)** — the **long-side cousin**: the
  post-*13D-filing* drift when an activist takes a large *long* stake. Same event-study shape (a
  public date, a pop/crater, and the question of a *tradable* drift), landing in the same place: a
  real announcement effect you can't capture and a post-event drift that a small, selected sample
  can't certify. Study 564 flips the sign (a short report, a negative drift) and adds the
  short-specific pathology — the **squeeze** — that no long-side study faces.
