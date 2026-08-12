# References & literature map — Study 878 (Economic Policy Uncertainty)

## The claim under test

- **The source paper.** Scott R. **Baker, Nicholas Bloom & Steven J. Davis**, *"Measuring
  Economic Policy Uncertainty"* (Quarterly Journal of Economics, 2016, 131(4):1593–1636).
  They construct a newspaper-based **Economic Policy Uncertainty (EPU)** index by counting
  articles in major US papers that jointly mention *economy*, *policy*, and *uncertainty*
  (plus components from tax-code expirations and forecaster disagreement). The index spikes
  around debt-ceiling standoffs, elections, wars, and crises.
- **The two stories the index is sold on.** (1) **Real/vol story** — high policy uncertainty
  precedes falling investment and **higher stock-market volatility** (Bloom, *"The Impact of
  Uncertainty Shocks"*, Econometrica 2009, is the theoretical parent: uncertainty shocks look
  like second-moment shocks and raise realized vol). (2) **Risk-premium story** — if
  uncertainty is a priced state variable, high-EPU periods should be compensated with
  **higher forward equity returns**. This study tests both legs directly.
- **The specific test here.** Predictive regressions of the H-month-ahead SPY **return** and
  the H-month-ahead **realized volatility** on the uncertainty **level** and its **change**,
  with a Newey-West HAC *t*, a block-shuffle placebo, a two-era cut, a costed timer, and a
  seeded synthetic positive control. The desk's prior — stated up front — is that uncertainty
  indices are mostly **contemporaneous**: they co-move with drawdowns and vol, and do not
  earn a forward return premium.

## Data-honesty note — the signal is a labelled VIX proxy

The intended signal is the **free Baker-Bloom-Davis US EPU** series
(`https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.csv`; FRED mirror
`USEPUINDXM` monthly / `USEPUINDXD` daily). `epu/data.py`'s `fetch_epu()` tries each endpoint
in order (4 retries, a real User-Agent) and caches the real series when it is reachable.

From the environment this study was built in, **only Yahoo Finance was network-reachable** —
`policyuncertainty.com` and `fred.stlouisfed.org` did not resolve. Rather than **fabricate** a
newspaper series (forbidden by the desk's data-honesty rule), `load_uncertainty()` falls back
to a **market-based proxy built from real VIX** (CBOE implied volatility, a fetchable Yahoo
series), returning `source == "vix_proxy"` so every published number carries its provenance.
This mirrors [387-economic-surprise-index](../../387-economic-surprise-index/), which proxies
the proprietary Citi CESI with public data and labels it as a proxy everywhere. **VIX is not
EPU** — it is market-implied, not text-based; the two co-move (~0.4–0.6 in the literature) but
the vol-leg magnitudes here reflect VIX's near-mechanical link to realized vol, and only the
**return-leg null** transfers cleanly (a null risk-premium is a null either way).

## What we measure, and the honesty rails

- **Predictive regressions, HAC inference.** OLS of the forward outcome on the predictor with
  a **Newey-West (Bartlett, 6-lag)** standard error on the slope — overlapping forward
  outcomes and a persistent regressor make plain OLS *t* far too optimistic.
- **One documented execution lag.** The predictor is known at month-end `t`; the forward
  outcome is realised strictly after `t`. Zero look-ahead.
- **A block-shuffle placebo.** Both the regressor and the outcome are heavily
  autocorrelated, so we shuffle the predictor in 12-month blocks (preserving its short-run
  autocorrelation) and re-fit the slope 1,000 times — the honest check that a slope isn't a
  lucky alignment of two persistent series.
- **A two-era cut.** A slope that lives in one regime and dies in the other is an artefact,
  not an edge — the sub-era bar the return leg fails.
- **The timer is graded separately.** A long/flat SPY rule keyed off the uncertainty level,
  net of a one-way cost, raced against buy-and-hold — the honest test of whether the (absent)
  edge would pay.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on every slope).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Bloom, N. (2009)** — *"The Impact of Uncertainty Shocks"* (Econometrica) — the
  second-moment-shock mechanism behind the vol leg.

## Data sources

- **^VIX + SPY daily closes** (yfinance, `auto_adjust=True`), 1993-02-28 → 2026-06-30, cached
  under `_cache/`. The VIX proxy stands in for the (unreachable) Baker-Bloom-Davis EPU feed.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [567-uncertainty-word-count](../../567-uncertainty-word-count/) — counts the word
  "uncertainty" in **single-firm 10-K filings** (a firm-level text signal for that firm's
  own stock). This study uses a **market-wide, macro** uncertainty index and asks whether it
  times the **aggregate** (SPY) — a different unit of analysis and a different text corpus.
- [318-election-volatility](../../318-election-volatility/) — the vol pattern around
  **US election dates** specifically (an event-window effect). This study is a continuous
  monthly predictive regression across the whole tape, not an event study.
- [313-geopolitical-shock](../../313-geopolitical-shock/) — reactions to discrete
  **geopolitical shock events**. EPU is a slow-moving continuous index, not a shock dummy.
- [255-fear-greed-index](../../255-fear-greed-index/) — a composite **sentiment**
  contrarian gauge. This study tests a **policy-uncertainty** index's forward-vol and
  risk-premium legs, not a sentiment mean-reversion timer.

None of the siblings run the **forward-vol AND forward-return predictive regression of a
macro policy-uncertainty index on the aggregate market** — this study's own axis. (And the
signal here is a labelled **VIX proxy**, disclosed above, because the newspaper EPU feed was
unreachable in-environment.)
