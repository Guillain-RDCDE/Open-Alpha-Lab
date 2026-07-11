# References & literature map — Study 645 (ECB Announcement Effect)

## The claim under test

- **The folklore.** "The euro area reacts to the ECB the way the US reacts to the Fed" —
  European desks trade around the Governing Council's 13:45 CET decision and 14:30 CET
  (14:15 since 2012) press conference the same way US desks trade around FOMC day: expecting a
  systematic equity drift or reaction, and a vol crush once the statement is out.
- **The academic anchor.** Lucca & Moench (2015, *The Pre-FOMC Announcement Drift*, JF) is the
  canonical version of this claim for the **Fed** — see [517-pre-fomc-drift](../../517-pre-fomc-drift/),
  which finds the effect real (in the archive) and decayed (post-2012). Amengual & Xiu (2018,
  *Resolution of policy uncertainty and sudden declines in volatility*, Journal of Econometrics)
  is the vol-crush analog — see [637-fomc-vol-crush](../../637-fomc-vol-crush/). Whether either
  pattern transplants to the ECB's own decision calendar is an open, distinct empirical
  question — a different central bank, a different currency bloc, a different investor base,
  and (until 2015) a different, monthly meeting cadence.
- **ECB-specific evidence.** Rosa (2011, *The high-frequency response of exchange rates to
  monetary policy actions and statements*, Journal of Banking & Finance) and Ehrmann & Fratzscher
  (2009, *Purdah — On the rationale for central bank silent periods*) study high-frequency
  (intraday, minute-level) reactions around ECB communication — a materially higher-resolution
  lens than the **daily** close-to-close test this study runs. A study finding a real intraday
  reaction and this study finding no *daily* directional drift are not in conflict: they answer
  different questions at different horizons.

## What we measure, and the honesty rails

- **FEZ decision-day log return vs all other days** — close-to-close, Welch *t* (single,
  non-overlapping events) plus a Newey-West (1987) 5-lag cross-check on the decision-day dummy
  regression, and a two-sided 20-seed × 1,000-draw random-calendar placebo.
- **Realized range** — FEZ (H−L)/prev-close on the same days: is a "loud" reaction actually
  there in the tape (mechanical, priced-in vol) even where the direction is not.
- **EURUSD absolute return** — the FX leg is the most direct pricing of a monetary-policy
  surprise; if euro-area equities aren't reacting, does the currency react instead?
- **Event window [−5..+3] and cumulative pre-meeting run-up** — the Lucca-Moench-style
  pre-announcement drift analog, with a per-meeting one-sample *t* (not just eyeballed offsets).
- **Era split at 2015-01-01** — the Governing Council's own structural change from a monthly to
  a 6-week decision cycle (announced 2014-07-03/17), a genuinely *ex ante* justified split, not
  a snooped one.
- **Costs charged one-way × NAV per leg** on the third-axis timing rule; the entry is the prior
  session's close (the ECB calendar is public months in advance, so this is a zero-look-ahead
  scheduled entry, the study's single documented execution convention).

## Data sources

- **FEZ raw OHLC** and **EURUSD=X OHLC** — yfinance (no key), cached under `_cache/`
  (`eae_fez.csv`, `eae_eurusd.csv`), 2005-01-03 → 2026-06-30.
- **208 scheduled ECB Governing Council monetary-policy decision dates, 2005 → 2026**,
  hardcoded in [`data.py`](../ecb_announcement_effect/data.py). Sourced from the ECB's own
  year-ahead schedule press releases for the monthly era (2005-2014:
  ecb.europa.eu/press/pr/date/2004/html/pr040617.en.html,
  .../2006/html/pr060602.en.html, .../2007/html/pr070330.en.html,
  .../2009/html/pr090508.en.html, .../2011/html/pr110420.en.html,
  .../2013/html/pr130517.en.html) and the ECB's monetary-policy-statement archive for the
  6-week era (2015-2026: ecb.europa.eu/press/press_conference/monetary-policy-statement/…).
  Every decision that also changed a key rate is cross-checked against the effective-date jump
  in the ECB's own key-rate series (FRED `ECBDFR`), which always lands 5-7 calendar days after
  the Governing Council meeting that decided it — an independent numerical check on the
  hand-compiled calendar.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [637-fomc-vol-crush](../../637-fomc-vol-crush/) — the **Fed's** implied-vol (^VIX) crush on
  FOMC decision day. Different central bank, different instrument (an index, not an ETF), and
  this study finds no equivalent "crush" story for the ECB — only a realized-range bump.
- [517-pre-fomc-drift](../../517-pre-fomc-drift/) and [67-fed-drift](../../67-fed-drift/) — the
  **Fed's** pre-announcement equity return drift (SPY) and its decay. Same *kind* of test as
  this study's event-window analog, applied to the Fed, not the ECB — and this study finds no
  drift at all, where 517 finds one (real pre-2012, decayed since).
- [135-fomc-cycle](../../135-fomc-cycle/) — the **Fed's** week-parity cycle across the whole
  inter-meeting period. Calendar-cycle returns, not a single decision-day event study.
- [322-fomc-blackout](../../322-fomc-blackout/) — the **Fed's** pre-meeting blackout/quiet-period
  window. A communication-regime study, not the announcement day itself.
- [606-opec-announcement-effect](../../606-opec-announcement-effect/) — the closest sibling in
  *shape*: another scheduled-decision calendar tested for vol vs. drift on a non-Fed
  policymaker (OPEC/OPEC+), landing on the same **Mixed** structure (real vol bump, no signed
  drift) this study lands on for the ECB — independently, on a different asset class and a
  different institution.
- [314-jackson-hole](../../314-jackson-hole/) — an **unscheduled-content, scheduled-date**
  central-bank *speech* (not a rate decision) and its own reaction window; a different kind of
  event entirely.

None of the siblings test what euro-area equities and EURUSD do around the **ECB's own**
scheduled decision calendar — this study is that corner of the desk's monetary-policy-event map.
