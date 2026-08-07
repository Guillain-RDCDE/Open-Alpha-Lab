# References & literature map — Study 836 (Rebalance Timing Luck)

## The claim, at full strength

Every backtest of a "monthly" or "annual" rules-based strategy makes a silent, arbitrary
choice: *on which day of the period do you rebalance?* The first trading day? The last?
The third Friday? Index and smart-beta providers each pick one — and the choice is
supposed to be immaterial, a detail. It is not. The **same** rule, rebalanced on a
different day of the cycle, traces a **materially different equity curve** and prints a
**materially different Sharpe ratio**. That spread is **rebalance timing luck**: phantom
dispersion driven entirely by *when* you rebalance, not by any difference in signal,
universe, or skill. It is the reason two funds tracking the "same" index can diverge by
hundreds of basis points a year, and the reason a backtest's Sharpe can be quietly
cherry-picked by trying rebalance dates.

## The source paper (the claim under test)

- **The source paper.** Corey **Hoffstein**, Nathan **Faber** & Steven **Braun** (Newfound
  Research), *"Rebalance Timing Luck: The Difference Between Hired and Fired"* (2019), and the
  companion *"Rebalance Timing Luck: The (Dumb) Luck of Smart Beta"* (Hoffstein, Sober &
  Vezeris, *Journal of Index Investing*, 2020). Building indices from the identical
  methodology but rebalanced in different months, they show the choice of rebalance date
  induces a large, **unrewarded** dispersion in realized returns and Sharpe — a strategy's
  "performance" can hinge on the calendar accident of when it reconstitutes. The prescribed
  cure is **portfolio tranching / overlapping portfolios** (Blitz, van der Grient & van Vliet
  2010): split the capital into `k` sub-portfolios each rebalanced on a different offset, so
  the arbitrary timing averages out and the dispersion collapses.
- **The overlapping-portfolio construction.** David **Blitz**, Bart **van der Grient** & Pim
  **van Vliet** (2010), *"Fundamental Indexation: Rebalancing Assumptions and Performance."*
  *Journal of Index Investing* 1(2). The formal treatment of overlapping portfolios as the
  fix for rebalance-frequency/date sensitivity — one slice of the book rebalanced each period,
  averaged into a single, timing-luck-free curve.
- **The methodology sensitivity backdrop.** **Arnott, Hsu & Moore (2005)** and the fundamental-
  vs cap-weight indexation debate first surfaced how much an index's measured "premium"
  depends on construction choices that are supposed to be innocuous — the same fragility this
  study isolates down to the single rebalance-date knob.

## What we measure, and the honesty rails

- **One book, every offset.** A cross-sectional 6-month momentum long-short (top-30% minus
  bottom-30%, dollar-neutral) rebalanced every 21 trading days — run once for each rebalance
  **offset** 0…20. Every offset trades the *identical* rule on the *identical* tape; only the
  day of the cycle differs.
- **Point-in-time, one documented lag.** On each rebalance day `d` the book is formed from the
  trailing-return signal **known at the close of `d-1`** and held fixed until the next
  rebalance. Zero look-ahead.
- **Luck, not skill — the persistence check.** If the offset dispersion were information, the
  lucky offset would stay lucky. We rank offsets by Sharpe in the first half of the sample and
  the second half and read the **Spearman rank correlation** — ≈ 0 means the winner is
  unforecastable, the signature of pure luck.
- **The fix, measured.** The tranched / overlapping portfolio averages all 21 offset books
  into **one** curve; there is nothing left to be lucky about, so the dispersion is zero by
  construction. Its Newey-West (HAC) *t* grades whatever genuine content remains.
- **Synthetic-only, capped at NONE.** The tape is built so a momentum sort has **zero** genuine
  edge on the null (`mom_edge = 0`); real free data can never *certify* "zero edge", so there is
  no real-tape stamp and the study is capped at `NONE` on the Signal axis (as with the desk's
  other method demos). A seeded positive control (`mom_edge > 0`) plants a real momentum
  premium to prove the machinery detects real edge and is not itself the artefact.
- **The timer is graded separately.** The tranched book pays a one-way × NAV cost on the slice
  it rotates each day, plus borrow on the short leg — the honest test of whether anything
  survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the tranched daily return series).
- **Spearman, C. (1904)** — the rank correlation used for the out-of-sample offset-persistence
  test.
- **Wilson, E. B. (1927)** — score interval for a binomial share (a shared inference primitive).
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a synthetic
  control is a machinery proof, never market evidence; `REAL` needs a robust *t* ≥ 2 on a real
  tape, which a synthetic-only demo can never provide), gross/net labelling with costs one-way ×
  NAV, and the ≥ 20-seed rule for any synthetic-dependent claim.

## Neighbours on this bench (the dedup map)

- **[Study 349 — Regime-Dependence](../../349-regime-dependence/)** — how a strategy's measured
  edge depends on the *sample/regime* you evaluate it in. Timing luck is a *sharper, orthogonal*
  fragility: the sample is fixed and the *rule* is fixed — only the arbitrary rebalance **day**
  moves, and it still swings the Sharpe. Different knob, same lesson that construction choices
  masquerade as signal.
- **[Study 102 — Free-Rebalance](../../102-free-rebalance/)** — the *diversification/rebalancing
  premium* (the return earned by mechanically rebalancing to fixed weights). That is about the
  **economic** effect of rebalancing back to target; this study is about the **statistical
  artefact** of *which day* you do it — no premium claimed, just phantom dispersion.
- **[Study 604 — Month-End-Rebalancing-Flows](../../604-month-end-rebalancing-flows/)** — a
  *calendar* effect driven by real predictable **flows** around month-end reconstitution (a
  tradable-in-principle market impact). Timing luck is the opposite: not a real flow to trade,
  but a *measurement* dispersion with nothing underneath — the lucky offset is unforecastable.

None of the siblings isolates the **rebalance-date offset** as the sole moving part and shows
its Sharpe dispersion is (a) material and (b) pure luck, collapsed by tranching — this study's
own axis.

## Data sources

- **No real data.** A deterministic, seeded synthetic return panel
  ([`timing_luck/data.py`](../timing_luck/data.py)); the null (`mom_edge = 0`) carries zero
  momentum edge by construction, the control (`mom_edge > 0`) a planted premium. All headline
  numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).
