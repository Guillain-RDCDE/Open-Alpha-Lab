# References & literature map — Study 736 (Sportsbook-Playoffs)

## The claim under test

- **The folklore, as its believers state it.** A recurring theme in sell-side notes,
  retail-investor threads and betting-industry trade press every autumn and January:
  sportsbook and iGaming stocks are "seasonal" and rally *into* the big US betting
  windows — NFL season and its January playoffs, and the March-Madness men's basketball
  tournament — because anticipation of a wall of betting handle (new-customer promos,
  deposit surges, record parlay volume) should be priced *ahead* of the games. The
  tradable corollary: accumulate DraftKings and the betting basket a couple of weeks
  before the first game and ride the anticipation. Representative statements of the
  "betting stocks are seasonal / the playoffs are a catalyst" thesis appear routinely in
  outlets such as *Legal Sports Report*, *Sportico*, *Front Office Sports* and broker
  previews around the NFL playoffs and Selection Sunday.
- **Why the premise is at least half-true.** The *handle* seasonality is real and
  well-documented: US commercial sports-betting handle peaks in the Sep→Jan NFL window
  and bumps again in March, then troughs over the summer, per the **American Gaming
  Association** (AGA) *Commercial Gaming Revenue Tracker* and the monthly handle
  releases of state regulators (e.g. the New Jersey DGE, Pennsylvania PGCB and Michigan
  MGCB monthly reports). The study encodes this shape as a small **labelled proxy**
  (`HANDLE_SEASONALITY` in `data.py`) — used only to motivate the claim, never traded.
  The open question this study answers is the *second* leap: whether that known,
  calendar-fixed activity seasonality translates into a tradable *stock-price* rally in
  the weeks before the games — or whether an efficient market has already priced a
  schedule everyone can read a year in advance.

## What we measure, and the honesty rails

- **Run-up cumulative abnormal return** — a constant-mean market model (Brown & Warner
  1985, *Journal of Financial Economics*, "Using daily stock returns: The case of event
  studies"): the "normal" return is the sample mean, the abnormal return is the demeaned
  daily return. The run-up statistic is the sum of the abnormal returns over the 10
  sessions ending the session before the first game — one number per event, one-sample
  *t* across the 12 **independent, non-overlapping** events (the correct unit; a daily
  panel would over-count autocorrelated days, so no HAC correction is the right call
  once each event is summarised to a single run-up figure).
- **No look-ahead, stated as a decision.** Both the NFL playoff schedule and the NCAA
  tournament dates are **public months in advance**, so the entry and exit sessions of a
  "buy N days before the first game" rule are known ex-ante — this is a **calendar-known
  rule** (like the turn-of-month effect), and applies **no execution lag** at all, by
  design. Documented once, applied once.
- **The placebo is right-tailed on purpose.** The claim predicts a *positive* run-up, so
  the falsification test asks how often a random calendar of 12 dates produces a run-up
  *this large or larger* (MacKinlay 1997, *Journal of Economic Literature*, "Event
  Studies in Economics and Finance", for the event-study framing; the random-calendar
  null mirrors this desk's `313-geopolitical-shock` and `707-plane-crash-effect`).
- **Hit rate carries a Wilson (1927) interval**; the placebo is a 20-seed × 1,000-draw
  random-calendar null; the point estimate carries an event-bootstrap CI.
- **Beta≈1 market-adjust robustness.** Because betting names are very high-beta, a raw
  run-up could just be the market rising; a DKNG-minus-SPY market-adjusted run-up (a
  blunt beta=1 adjust, no fitted beta to avoid look-ahead) checks that the result is not
  a market-timing artifact.
- **Survivorship named on the Signal axis.** The 5-name basket is *current, still-listed*
  US betting names — survivors of the 2021-22 SPAC-era sector wipeout. This tilts the
  basket toward the names that survived, which biases a "rally" test **upward** (dead
  names can't rally), so the null result is the conservative reading. Coverage is full
  5/5 for every event (all members trade before the first event), so no member is
  silently backfilled.

## Data sources

- **DKNG, PENN, CZR, MGM, RSI, BETZ, SPY** daily total-return closes (`auto_adjust=True`)
  — yfinance (no key), cached under `_cache/`, 2019 → 2026-06-30. **DKNG floored at its
  2020-04-24 SPAC-merger close** (the earlier tape is the Diamond Eagle / DEAC cash
  shell, not the operating DraftKings — named, not hidden).
- **12 hardcoded flagship betting-season starts** (6 NFL Wild-Card weekends, 6
  March-Madness Round-of-64s, 2021 → 2026) in
  [`sportsbook_playoffs/data.py`](../sportsbook_playoffs/data.py), from Pro-Football-
  Reference / NFL.com playoff schedules and NCAA.com tournament brackets, cross-checked
  against contemporary reporting for the exact first-game date.
- **US sports-betting handle seasonality PROXY** — a small hardcoded 12-month multiplier
  shape (mean 1.0) approximating AGA / state-regulator monthly handle releases; a
  **labelled illustration**, never priced, never traded.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [707-plane-crash-effect](../../707-plane-crash-effect/) — the same event-study
  machinery (constant-mean abnormal returns, one-sample *t* across independent events,
  random-calendar placebo, costed timer, synthetic positive control), applied to a
  *sentiment/shock* trigger rather than a *scheduled, anticipated* one.
- [708-eurovision-effect](../../708-eurovision-effect/) — a "national-pride bump" event
  study with the same country-ETF-mapping and labelled-proxy discipline; the sibling
  that shares this study's exact file shape.
- [300-sports-sentiment](../../300-sports-sentiment/) — the Edmans, Garcia & Norli
  (2007, *Journal of Finance*, "Sports Sentiment and Stock Returns") "loss effect": a
  national team's elimination souring the *home* market. That study tests sports *result*
  sentiment on a broad market; this one tests a *scheduled betting-season* anticipation
  trade on the betting stocks themselves — different trigger, different instruments,
  different mechanism (anticipation, not outcome).

None of the siblings test whether **the betting stocks themselves rally into the betting
calendar** — that scheduled-anticipation axis is this study's own.
