# References & literature map — Study 390 (Activist-13D)

## The claim under test

- **The rule (Schedule 13D mechanics).** Section 13(d) of the Securities Exchange Act of
  1934 (and SEC Regulation 13D-G) requires any person acquiring beneficial ownership of
  **more than 5%** of a public company's voting equity *with an intent to influence
  control* to file a **Schedule 13D** with the SEC. The 2023 SEC amendments shortened the
  filing window (to 5 business days from the older 10-day window). The filing is public on
  EDGAR and names the activist, the stake, and the "purpose of transaction" — the moment the
  market learns an activist has arrived.
- **The folklore.** Financial media and activist-fund marketing frame the 13D as a buy
  signal: the stock *pops* on the filing and then *keeps drifting up* for months as the
  activist agitates for buybacks, spin-offs, board seats or a sale (the textbook stories:
  Icahn → Apple buyback, Elliott → Salesforce/Pinterest, Trian → P&G board seat, Third
  Point → Yum! split). The pitch is a free ride for anyone who buys when the 13D hits EDGAR.

## What the academic literature actually finds

- **Brav, Jiang, Partnoy & Thomas (2008), *Hedge Fund Activism, Corporate Governance, and
  Firm Performance* (Journal of Finance).** The canonical study: a **positive abnormal
  announcement return** of roughly **+7–8%** in the (−20, +20)-day window around 13D
  filings, with **no reversal** over the next year on average. This is the strongest
  steelman of the claim — but it is a *large-sample* (1,000+ event) result on the *event
  window centred on the filing*, dominated by the day-0 pop, not a clean test of a tradable
  post-filing drift on a buy-able entry.
- **Clifford (2008), *Value creation or destruction? Hedge funds as shareholder activists*
  (Journal of Corporate Finance)** and **Greenwood & Schor (2009), *Investor activism and
  takeovers* (Journal of Financial Economics).** Both localise the abnormal return: most of
  the activist "alpha" is concentrated in targets that are subsequently **acquired**. Absent
  a takeover, the post-filing drift is far weaker — i.e. the average is carried by a tail of
  outcomes, the cross-sectional fragility we reproduce on a small basket.
- **Bebchuk, Brav & Jiang (2015), *The Long-Term Effects of Hedge Fund Activism* (Columbia
  Law Review).** Argue activism is *not* followed by long-run underperformance (rebutting the
  "short-termism" critique) — but again on a large panel, and the question of a *tradable*
  post-announcement excess for an outside buyer is distinct from "are targets worse off."

## Why a hardcoded basket — and the bias it carries

- **EDGAR coverage.** Every Schedule 13D is public on `sec.gov/cgi-bin/browse-edgar`, but a
  clean, deduplicated, ticker-mapped panel of *all* 13D filings (with the genuine first-
  disclosure date, not amendments) is not available on a free per-ticker feed like yfinance.
  We therefore use a **transparent, hardcoded table of 25 famous campaigns** — and name the
  consequence on the Signal axis.
- **Selection / survivorship bias (named on the Signal axis).** Campaigns enter folklore
  because they were *prominent and often profitable*. Selecting on fame biases the sample
  **toward** the claim — so our basket is the *easiest* possible test of "13Ds drift up,"
  and it still fails to clear significance. A bias that points *for* the effect and yet
  leaves it insignificant is decisive evidence *against* a deployable edge. (House rule:
  survivorship is named on the Signal axis, not buried in Tradability — see METHODOLOGY.)

## Why ~25 events cannot be an edge — the statistics

- **Small-sample inference / power.** With *k* ≈ 25 events, the standard error of a
  conditional-mean estimate is large; a few-percent excess over a noisy multi-month return
  cannot be distinguished from luck. We test the mean excess against zero with a one-sample
  *t* and, because *k* is tiny, with a **placebo / randomization test** — drawing the same
  horizon excess from *random* dates on the *same targets* (Fisher's randomization logic;
  Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **Post-earnings/announcement drift analogue.** The "drift" framing borrows from
  post-earnings-announcement drift (Bernard & Thomas, 1989) — but PEAD is a large-sample,
  cross-sectional effect; a 25-name event basket has none of that statistical mass.
- **Multiple testing / selection on famous cases.** Harvey, Liu & Zhu (2016), *…and the
  Cross-Section of Expected Returns* (Review of Financial Studies) formalise why a pattern
  selected ex-post from memorable wins needs a far higher bar than a naive *t*-stat.

## Method lineage (the desk's shared engine)

- **Pop vs drift split.** [`strategy.announcement_pops`](../activist_13d/strategy.py) (the
  uncapturable day-0 news) vs [`strategy.event_excess_drift`](../activist_13d/strategy.py)
  (the tradable, 1-day-lagged, excess-of-SPY leg) — the two objects "never wrong" fuses.
- **Welch t + placebo p-value.** [`strategy.welch_t_vs_zero`](../activist_13d/strategy.py)
  and [`strategy.placebo_pvalue`](../activist_13d/strategy.py) — the mean excess against
  zero, and a 20,000-draw randomization null sized to the event count, drawing from the same
  targets to control for their own drift.
- **Deterministic synthetic control.**
  [`data.synthetic_events`](../activist_13d/data.py) injects a known number of events with a
  planted post-announcement drift; the offline core runs with no network. The control
  confirms the engine is faithful *and* that ~25 events cannot reach significance unless the
  planted edge is implausibly large.

## Data sources used here

- **yfinance** daily adjusted closes for SPY + 25 activist targets, 2010-01-04 →
  2026-06-18, cached under `_cache/event_prices.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- Event-driven cousins on this bench measure the *post-announcement drift* of other discrete
  corporate events (splits, dividend initiations, spin-offs, index inclusions). The activist
  13D is the same shape — a public corporate-event date, a pop, and the question of whether a
  *buy-able* drift follows — and lands in the same place: a real announcement effect you
  can't capture, and a post-event drift inside the noise on a small, selected sample.
