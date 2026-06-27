# References & literature map — Study 517 (Pre-FOMC-Drift)

## The claim under test

- **The seminal paper.** David O. Lucca & Emanuel Moench, *The Pre-FOMC Announcement Drift*
  (2015, Journal of Finance, 70(1), 329–371). Their headline: since 1994, US equities have
  earned **large excess returns in the 24 hours before scheduled FOMC announcements** — so large
  that the pre-FOMC window accounts for the bulk of the realized equity premium over their
  sample, with the drift appearing **only around scheduled meetings** (not unscheduled actions)
  and concentrated in the hours **before** the 2:15 p.m. statement.
- **The folklore version.** "Stocks always drift up into Fed day — buy the close before the
  meeting." The retail compression of a genuinely striking academic finding.
- **Why it matters.** If a third of the equity premium is earned on 3% of days clustered around a
  *pre-scheduled* event, that is a profound puzzle for efficient markets — there is no news in
  the window, the meeting date is known months ahead, yet prices drift up.

## Distinct from the desk's neighbouring FOMC studies

- **vs Study 67 (Fed-Drift).** The original SPY single-day write-up of Lucca-Moench with the
  pre/post-2011 split. This study (517) re-derives it on an independent hardcoded calendar and
  adds (a) an **overnight-vs-intraday decomposition** (the third axis: is it the run-up *into*
  the release, as the paper claims?), (b) a **random-calendar placebo**, and (c) a **survivor
  basket** as cross-sectional colour. The two studies agree (Real × Fragile) — a deliberate
  replication check.
- **vs Study 135 (FOMC-Cycle).** Cieslak, Morse & Vissing-Jorgensen (2019, *Stock Returns over
  the FOMC Cycle*, Journal of Finance) document a different pattern — returns concentrated in
  **even weeks** of the 6-week inter-meeting cycle. That is a *cycle* claim over the whole
  inter-meeting period; ours is the *single pre-announcement session*. Different windows,
  different mechanism, separate verdict.
- **vs Study 322 (FOMC-Blackout).** The pre-meeting communications-blackout "calm" trade — again
  a different window (the ~10 days before the meeting), stamped None × Mirage.

## Why a big *t* still needs a placebo + a decay split

- **Publication decay.** R. David McLean & Jeffrey Pontiff, *Does Academic Research Destroy Stock
  Return Predictability?* (2016, Journal of Finance) — documented anomalies lose ~58% of their
  return out-of-sample after publication. The pre-FOMC drift is a textbook case: the post-2012
  Welch *t* collapses to 0.28. Subsequent work (e.g. the New York Fed's own follow-ups) finds the
  drift weakened or reversed after the paper circulated.
- **Multiple testing / data-mining.** Harvey, Liu & Zhu (2016, *…and the Cross-Section of
  Expected Returns*, RFS) on the t > 3 hurdle for "discovered" effects. Our placebo
  (`strategy.placebo_pvalue`) relocates the same number of event days at random 20,000 times — the
  honest "could a random calendar of this size have looked this good?" null.
- **Welch t.** B. L. Welch (1947, *The generalization of "Student's" problem when several
  different population variances are involved*, Biometrika) — the unequal-variance two-sample test
  used for pre-FOMC vs other-day means.

## The overnight-vs-intraday angle

- **Overnight anomaly.** Cooper, Cliff & Gulen and the broader literature (e.g. *Return
  Differences between Trading and Non-Trading Hours*) show the overnight session carries a
  disproportionate share of the equity premium. Lucca-Moench's mechanism is specifically about the
  **intraday run-up** into the 2:15 release; our decomposition (`strategy.overnight_intraday`)
  tests whether the pre-FOMC excess is really intraday or partly overnight — it turns out to be
  split roughly evenly, qualifying the clean run-up story.

## Method lineage (the desk's shared engine)

- **Event-vs-rest + one-sample / Welch t.** [`strategy.event_vs_rest`](../pre_fomc_drift/strategy.py)
  — pre-FOMC mean vs other-day mean, one-sample *t* of the event days vs 0, and the share of
  cumulative return on the event days.
- **Random-calendar placebo.** [`strategy.placebo_pvalue`](../pre_fomc_drift/strategy.py) — 20,000
  random same-size calendars; the honest small-effect null.
- **Decay split.** [`strategy.split_summary`](../pre_fomc_drift/strategy.py) — the pre/post-2012
  publication-era comparison.
- **Deterministic synthetic control.** [`data.synthetic_world`](../pre_fomc_drift/data.py) plants a
  known pre-meeting drift; with the edge set to zero the inference must NOT manufacture
  significance (seed-averaged over 40 seeds). The offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted closes for SPY + a fixed 30-name large-cap basket, plus raw SPY
  Open/Close for the overnight-vs-intraday split, 1993-02-01 → 2026-05-29, cached under
  `_cache/pfd_prices.csv` and `_cache/pfd_ohlc.csv`.
- **FOMC scheduled-announcement calendar** — hardcoded in [`data.FOMC_DATES`](../pre_fomc_drift/data.py)
  from the Federal Reserve's historical FOMC calendars (scheduled meetings only, 1994–2026). All
  headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 67 — Fed-Drift](../67-fed-drift/)** (the original SPY write-up; Real × Fragile — we
  replicate and agree).
- **[Study 135 — FOMC-Cycle](../135-fomc-cycle/)** (the even-week cycle; Weak × Mirage).
- **[Study 322 — FOMC-Blackout](../322-fomc-blackout/)** (the pre-meeting blackout; None × Mirage).
