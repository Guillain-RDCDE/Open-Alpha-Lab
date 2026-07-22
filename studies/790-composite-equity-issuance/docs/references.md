# References & literature map — Study 790 (Composite Equity Issuance)

## The claim under test

- **The source paper.** **Daniel, K. and Titman, S. (2006), *"Market Reactions to Tangible and
  Intangible Information," Journal of Finance* 61(4), 1605–1643.** They introduce **composite
  equity issuance** — the growth in a firm's market equity *not* attributable to its stock
  return:

      iota(t) = log( ME(t) / ME(t-5) ) - r(t-5, t),

  where `ME` is market capitalisation and `r(t-5,t)` is the cumulative log stock (total) return
  over the same five years. It captures **all** the ways a firm changes its equity base — SEOs,
  employee stock, share-financed M&A, buybacks — in one log measure, not just the raw
  share-count change. Their finding: firms with **high** composite issuance subsequently
  **underperform**, and those with **negative** issuance (net repurchasers) **outperform**, over
  the following years.
- **The broader net-issuance literature.** **Pontiff, J. and Woodgate, A. (2008), *"Share
  Issuance and Cross-Sectional Returns," Journal of Finance* 63(2), 921–945** show the
  predictive power of net share issuance directly, and **Fama, E. and French, K. (2008),
  *"Dissecting Anomalies," Journal of Finance* 63(4)** and Daniel-Titman (2006) both fold
  issuance into the modern anomaly set. **Loughran, T. and Ritter, J. (1995)** (the "new issues
  puzzle") and **Ikenberry, Lakonishok, Vermaelen (1995)** (buyback drift) are the discrete-event
  ancestors. The consensus is that the effect is real and robust in the **full**, point-in-time,
  survivorship-free US cross-section (thousands of names).
- **What we test here.** Does the **5-year log composite** measure — not the 1-year raw
  share-count change — replicate on a **small, transparent large-cap survivor basket** built
  from free public data (EDGAR + yfinance), with point-in-time shares, one execution lag and
  realistic costs? A wrong-sign or insignificant result is the *expected* small-sample /
  survivorship outcome, and is the honest finding to publish.

## What we measure, and the honesty rails

- **Point-in-time shares.** Every EDGAR XBRL fact carries a `filed` date. The share count used
  at each formation Dec 31 is the latest fact **filed on or before** that date — never a filing
  that had not yet reached EDGAR. This is the study's single execution-timing discipline; the
  composite window [t−5, t] is fully public at *t*, and the position is held *t → t+1* (one lag).
- **The composite measure cancels splits.** `log(shares) + log(raw price)` is split-neutral in
  the market-cap leg, and the adjusted-close ratio is split-neutral in the return leg, so a
  stock split cannot masquerade as issuance (the exact trap Study 519 handles for the raw
  share-count change).
- **Survivorship is named on the Signal axis, with its direction argued.** A point-in-time,
  survivorship-free issuance universe is a CRSP/Compustat product, not a free feed. We fix a
  36-name large-cap survivor basket; delisted issuers are absent, which biases the sort
  **against** the claim (the surviving high-issuers are disproportionately the ones whose
  dilution funded growth that worked). We reason about that direction in
  [`docs/results.md`](results.md) rather than burying it.
- **Inference.** One-sample *t* of the annual long-short vs zero (primary), a **Newey-West**
  HAC *t* (the 5-year formation windows overlap, inducing serial dependence), a Wilson interval
  on the win-rate, and a **20,000-draw label-shuffle placebo** (permute which name carries which
  composite-issuance value, preserving the marginal issuance distribution and the realised
  cross-section of forward returns). Costs are one-way × both legs × turnover + a short-leg
  borrow. A **25-seed synthetic control** proves the engine recovers a planted low-issuance edge
  (positive *t*) and refuses to manufacture significance from a null world.

## Data sources

- **Shares** — SEC EDGAR XBRL `companyconcept` API: `us-gaap:CommonStockSharesOutstanding`
  primary, `dei:EntityCommonStockSharesOutstanding` fallback. Cached under `_cache/shares.csv`
  (point-in-time year-end grid). Fetch pattern adapted from
  [`tools/fetch_altdata.py`](../../../tools/fetch_altdata.py).
- **Prices** — yfinance (no key), raw + adjusted daily closes, year-end grid
  (`_cache/price_raw.csv`, `_cache/price_adj.csv`).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [519-net-share-issuance](../../519-net-share-issuance/) — the **1-year** pure split-adjusted
  **share-count change** (Pontiff-Woodgate framing), sorted annually. This study is the
  **5-year log composite** measure of Daniel-Titman, which also absorbs the price/dividend and
  intangible-information components, not just the raw count. (Both land `NONE` on a survivor
  basket — 519 inside its noise at *t* = −0.86, this one *significantly* wrong-signed at
  *t* = −2.29 — which is itself the interesting contrast.)
- [250-reverse-split](../../250-reverse-split/) — a **corporate-action event** (reverse splits),
  a discrete signal about distress, not a continuous cross-sectional issuance measure.
- [368-buyback-drift](../../368-buyback-drift/) — the **announcement event** (Ikenberry et al.):
  the drift after a discrete buyback *announcement*. This study measures the **realised** 5-year
  composite issuance from filed share counts and market caps, with no announcement calendar.

None of the siblings test the **5-year log composite** issuance measure on a point-in-time
filed-shares panel — that is this study's own axis.
