# References & literature map — Study 320 (Russell-Reconstitution)

## The claim under test

- **Front-running the Russell reconstitution.** The folklore (and a long line of sell-side
  "Russell rebalance trade" notes): once a year FTSE Russell rebuilds the Russell 1000 /
  2000 / 3000 from a late-May rank-day snapshot, with the new membership effective after the
  close of the late-June reconstitution Friday. Because every index fund must hold the new
  weights by that close, a giant, forced, one-directional flow lands in that day's closing
  auction — one of the highest-volume sessions of the US year. The date is known years ahead,
  so (the claim goes) you can *front-run* it: buy the small-cap index in the run-up window
  and sell on/after the event. We test this on the index ETF itself (IWM, with IWO as a
  breadth check) — a calendar event study, not a name-by-name inclusion study.

## The microstructure the claim leans on — predictable index-rebalancing flow

- **Index-reconstitution price pressure.** FTSE Russell, *Russell US Indexes Reconstitution*
  (annual methodology / recon calendar) — the rank-day → effective-Friday mechanics.
- **Demand curves for stocks slope down.** Shleifer (1986), *Do Demand Curves for Stocks
  Slope Down?* (Journal of Finance); Harris & Gurel (1986), *Price and Volume Effects
  Associated with Changes in the S&P 500 List* (Journal of Finance) — the founding evidence
  that forced index demand moves single-stock prices (mostly the *additions*, with partial
  reversal). The open question this study asks is whether that single-name pressure
  aggregates into a tradable move in the *index* around a *calendar* date.
- **Russell reconstitution specifically.** Madhavan (2003), *The Russell Reconstitution
  Effect* (Financial Analysts Journal); Chen, Noronha & Singal (2006), *Index Changes and
  Losses to Index Fund Investors* — document predictable single-name additions/deletions
  price pressure and the cost it imposes on mechanical index funds, and note that the
  predictability invites front-running that *competes the effect away over time*.
- **Decay of a known anomaly.** McLean & Pontiff (2016), *Does Academic Research Destroy
  Stock Return Predictability?* (Journal of Finance) — once an effect is published and
  trades crowd in, it shrinks. A reconstitution date known years in advance is the extreme
  case: the flow is transparent, so the index-level drift is the first thing arbitraged out.

## Why the index-level test comes up empty — aggregation and tiny-n

- **Single-name pressure ≠ index move.** Even if individual additions pop and deletions sag,
  the Russell 2000 holds ~2000 names; adds and drops roughly offset at the index level, and
  the cap-weighting dilutes the small-name moves that dominate the reconstitution. There is
  no a-priori reason the *index* should drift up into the event.
- **The small-n problem.** There is one reconstitution per year, so 2000–2025 yields just
  n = 26 event windows. At ~1.3% daily small-cap vol, a five-session window's mean has a
  standard error of order ~0.5%, so a front-run edge would have to be very large (~200 bps
  cumulative) to clear |t| = 2 — see the synthetic power sweep in
  [`docs/results.md`](results.md). Harvey, Liu & Zhu (2016), *…and the Cross-Section of
  Expected Returns* (Review of Financial Studies) — the multiple-testing / low-power trap
  that makes thin calendar samples easy to over-read.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../russell_reconstitution/strategy.py).
- **Event study with the right null.** We test the event window's *excess* over the
  matched unconditional rolling-window baseline (and a same-month June control), not its raw
  return, and confirm with a permutation null — the discipline shared with the desk's other
  calendar studies.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), adjusted-close, IWM back to 2000 (IWO as a
  breadth check on an explicit fetch). All headline numbers are pinned with an as-of date and
  content fingerprint (see [`docs/results.md`](results.md)). The offline reproducible core
  and test-suite run on the deterministic [`data.synthetic_daily`](../russell_reconstitution/data.py)
  generator, never the network.

## Related desk studies

- **[Study 249 — Index-Inclusion](../../249-index-inclusion/)**: the *single-name* S&P 500
  inclusion pop — a cross-section of individual additions, with an announce→effective window
  and a post-inclusion give-back. Study 320 is the *calendar / whole-index* counterpart: the
  Russell ETF on a date fixed years ahead, no name-picking and no survivorship panel. The
  two are deliberately distinct angles on the same "forced index flow" microstructure.
- **[Study 287 — Easter-Effect](../../287-easter-effect/)**: the same calendar-event-study
  machinery (event window vs unconditional + same-period control, HAC t, permutation,
  synthetic control) applied to a once-a-year date — there the signal was Real; here it is
  not, which is the point of running the identical apparatus.
