# References & literature map — Study 23 (Broken-Tether)

## The source — where this study came from

- **Zura Kakushadze & Juan Andrés Serur, *151 Trading Strategies* (Palgrave Macmillan, 2018).**
  SSRN [3247865](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865); arXiv
  [1912.04492](https://arxiv.org/abs/1912.04492). The relevant entry is **strategy §3.8 (pairs
  trading)** — form a dollar-neutral position on the spread between two related assets and trade its
  mean reversion.

## The claim under test — the steelman

- **Pairs trading as relative-value arbitrage.** Evan Gatev, William Goetzmann & K. Geert Rouwenhorst,
  *"Pairs Trading: Performance of a Relative-Value Arbitrage Rule"*, **Review of Financial Studies**
  19(3), 2006: the classic study showing a simple distance/spread rule on cointegrated US equities
  earned a meaningful, market-neutral excess return over 1962–2002.
- **Cointegration.** Robert Engle & Clive Granger, *"Co-integration and Error Correction"*,
  **Econometrica** 55(2), 1987 — the formal basis: two I(1) series are cointegrated if a linear
  combination is stationary, which is what makes a spread mean-revert.

## The honest counters — why the verdict is `WEAK` / `MIRAGE` / `Breaks`

- **The edge has decayed.** Binh Do & Robert Faff, *"Does Simple Pairs Trading Still Work?"*, **Financial
  Analysts Journal** 66(4), 2010: the Gatev et al. profitability falls sharply after 2002 as the trade
  became crowded — consistent with the thin, fragile result here on modern liquid ETFs.

- **Cointegration is unstable out of sample.** A relationship estimated on the past need not persist;
  the `decompose.in_sample_vs_oos` split and the `extension.hedge_ratio_drift` measure exactly this — a
  hedge ratio that wanders means the spread reverts toward a moving anchor.

- **Selection / multiple testing.** Scanning a universe for "cointegrated" pairs surfaces false
  positives: a fraction of *independent* random walks show a tradable-looking spread half-life by chance
  (`decompose.spurious_pairs`). The broader hazard is the data-snooping literature — Ryan Sullivan, Allan
  Timmermann & Halbert White (*Journal of Finance* 1999); Bailey–Borwein–López de Prado–Zhu, *"Pseudo-
  Mathematics and Financial Charlatanism"* (*Notices of the AMS* 2014) — on the inflation of backtested
  performance by searching over many candidates.

## The desk's own method — engine and reproducibility

- **Causal estimation.** The hedge ratio and z-score are trailing-window (no full-sample look-ahead —
  the trap [Study 22](../../22-crystal-ball/) dissects). Half-life from an AR(1) on the spread.
- **Reproducibility.** Headline numbers are pinned with [`quantlab.repro`](../../../quantlab/repro.py)
  (an as-of date + a content fingerprint of the basket closes).

## Caveats stated in the open (house rule)

- **Split-only closes, log space.** Hedge ratio and spread are estimated on log-prices; both legs are
  charged the same series.
- **Scanned ETF pairs, not economically-linked ones.** Liquid ETFs are already heavily arbitraged and
  lack a *structural* tether (unlike dual share classes, ADRs, or a fund vs its NAV) — a stated beat-7
  fork, where a durable pair might still live.

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
