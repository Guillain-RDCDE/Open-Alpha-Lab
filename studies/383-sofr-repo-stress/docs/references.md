# References & literature map — Study 383 (SOFR-Repo-Stress)

## The claim under test

- **The September 2019 repo spike.** On 16–17 September 2019 the overnight Treasury
  general-collateral (GC) repo rate spiked to roughly **10%** intraday and the Secured
  Overnight Financing Rate (SOFR) jumped to about **5.25%**, several multiples of the
  Fed's target range. The Federal Reserve Bank of New York restarted overnight (and then
  term) repo operations for the first time since the financial crisis. See the **New York
  Fed**, *Statement Regarding Repurchase Operations* (Sept 2019) and the Fed's later
  post-mortems on reserve scarcity. This is the founding event of the "watch the repo
  market" macro narrative.
- **The folklore.** After Sept 2019, a spike in overnight repo / SOFR (or the SOFR-to-OIS
  / SOFR-to-fed-funds spread) became a widely-repeated *early warning* for risk assets
  across macro commentary and financial media: the idea that when the funding *plumbing*
  seizes, equities and credit are about to crack. The "never miss a repo spike" reputation
  is precisely what invites a sample-size audit — there are only a handful of named
  episodes, several of which were *not* followed by a sell-off.
- **Official analysis of the spike's causes.** Afonso, Cipriani, Copeland, Kovner,
  La Spada & Martin (2020/2021), *The Market Events of Mid-September 2019* (Federal Reserve
  Bank of New York **Staff Report 918** / *Economic Policy Review*) — the canonical
  decomposition (corporate-tax date, Treasury settlement, reserve scarcity), arguing the
  spike was a *mechanical funding* event, not a solvency signal about risk assets.
  Bank for International Settlements (2019), *September stress in dollar repo markets*
  (BIS Quarterly Review box).

## Why true repo/SOFR stress is not on yfinance — and what we do instead

- **No free clean daily funding-stress tape.** SOFR and its percentiles are published by
  the **New York Fed** (and FRED series `SOFR`, `SOFR99` etc.), and the GC repo / tri-party
  rates by the **Office of Financial Research**; but a clean, gap-free *daily stress series*
  and the historical intraday repo prints are not available through the free yfinance
  endpoint, which serves per-ticker OHLCV only. We therefore work from a **hardcoded,
  sourced table of named repo-stress episodes** (`data.REPO_EPISODES`) assembled from
  contemporaneous Fed/ press coverage, and measure the forward reaction of the risk assets
  the folklore says it warns. This is a methodological choice, named on the Signal axis:
  the *signal* is a curated event list, not a mechanical threshold on a live feed.
- **The risk assets.** SPY (S&P 500 ETF) for equities; HYG (iShares iBoxx high-yield) and
  LQD (iShares iBoxx investment-grade) for credit — the assets the "funding stress → risk
  off" thesis predicts should weaken. All public adjusted closes via yfinance.

## Why ~a dozen named episodes cannot be an edge — the statistics

- **Small-sample inference / power.** With *k* ≈ 13 (and only ~3 truly *systemic*) events,
  the standard error of a conditional-mean estimate is large; a forward return of a few
  percent over a noisy base cannot be distinguished from luck. We test the conditional mean
  against the unconditional mean with a **Welch two-sample t** (Welch, 1947, *The
  generalization of "Student's" problem when several different population variances are
  involved*) and, because *k* is tiny, with a **placebo / randomization test** — the share
  of random same-size date draws at least as bearish as the episode set (Fisher's
  randomization logic; Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **Confounding / event overlap.** The one episode universally remembered as a crash —
  **March 2020** — is a COVID shock that *also* seized funding markets; attributing the
  equity move to repo is a textbook confound (the spike and the crash share a common cause,
  and in fact March 2020 marked a *bottom*, not a top). Likewise **SVB (March 2023)** is a
  bank-run event. Disentangling "repo caused it" from "a bigger shock caused both" is
  impossible with a handful of overlapping episodes.
- **Base rates and the warning-rate illusion.** US equities rise in the large majority of
  rolling windows, so even a real funding shock is typically *backstopped* and followed by a
  rebound; a high post-signal *down*-rate would be needed to beat the base rate, and it
  isn't there past a week — the classic base-rate fallacy (Kahneman & Tversky, 1973, *On the
  psychology of prediction*).
- **Multiple testing / selection on a famous event.** Narratives that survive into folklore
  are selected on a single vivid in-sample case (Sept 2019); Harvey, Liu & Zhu (2016),
  *…and the Cross-Section of Expected Returns* (Review of Financial Studies) and Bailey &
  López de Prado (2014), *The Deflated Sharpe Ratio*, formalise why an ex-post "it always
  warns" rule needs a far higher bar than a naive *t*-stat.

## Method lineage (the desk's shared engine)

- **Welch t + placebo p-value.** [`strategy.welch_t`](../sofr_repo_stress/strategy.py) and
  [`strategy.placebo_pvalue`](../sofr_repo_stress/strategy.py) — the Signal-axis tests:
  conditional vs unconditional forward returns, and a 20,000-draw left-tail randomization
  null sized to the event count.
- **Deterministic synthetic control.**
  [`data.synthetic_tape`](../sofr_repo_stress/data.py) plants a known number of stress dates
  and (optionally) a known forward edge; the offline core runs with no network. The control
  confirms the inference is faithful *and* that ~a dozen events cannot reach significance
  unless the planted edge is implausibly large.
- **Forward-return measurement with execution lag.**
  [`strategy.event_returns`](../sofr_repo_stress/strategy.py) enters one day after the
  episode (no look-ahead) and holds a fixed horizon; costs applied in
  [`strategy.net_of_costs`](../sofr_repo_stress/strategy.py).

## Data sources used here

- **yfinance** daily adjusted closes for SPY + HYG + LQD, 2010-01-04 → 2026-06-18, cached
  under `_cache/risk_assets.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).
- **Repo-stress episode table** (`data.REPO_EPISODES`): hardcoded dates from NY Fed
  operations announcements, BIS/Fed post-mortems, and contemporaneous press coverage.

## Related desk studies

- **[Study 115 — Credit-Spreads](../115-credit-spreads/)**: whether the credit cycle itself
  carries a forward warning for risk — the slow-moving cousin of the funding-stress thesis.
- **[Study 111 — VIX-Term-Structure](../111-vix-term-structure/)**: another "stress gauge"
  said to flash before sell-offs; same question of whether a stress signal beats the base
  rate net of when it actually fires.
- **[Study 317 — Fed-Balance-Sheet](../317-fed-balance-sheet/)**: reserves and QT — the
  slow-moving backdrop behind why repo gets tight at quarter-ends in the first place.
