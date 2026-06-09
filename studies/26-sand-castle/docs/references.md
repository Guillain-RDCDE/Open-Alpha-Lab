# References & literature map — Study 26 (Sand-Castle)

## The source — where this study came from

- **Zura Kakushadze & Juan Andrés Serur, *151 Trading Strategies* (Palgrave Macmillan, 2018).**
  SSRN [3247865](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865); arXiv
  [1912.04492](https://arxiv.org/abs/1912.04492). The relevant entry is **strategy §3.18 (statistical
  arbitrage – optimization)** — express expected P&L, variance and Sharpe in terms of dollar holdings
  and the sample covariance, and solve for the mean-variance-optimal weights ``w ∝ C⁻¹ E``.

## The claim under test — the steelman

- **Mean-variance optimization.** Harry Markowitz, *"Portfolio Selection"*, **Journal of Finance** 7(1),
  1952 — the foundation: with expected returns `E` and covariance `C`, the maximum-Sharpe dollar-neutral
  portfolio is `w ∝ C⁻¹E`. The promise here is that accounting for cross-stock correlations should beat a
  naive signal-weighting.
- **The reversion signal.** The short-horizon mean reversion the optimizer trades is the contrarian
  effect of Bruce Lehmann (*QJE* 1990) and Andrew Lo & Craig MacKinlay (*RFS* 1990); residual / market-
  neutral stat-arb is the classic implementation (Avellaneda & Lee, *"Statistical Arbitrage in the U.S.
  Equities Market"*, **Quantitative Finance** 2010).

## The honest counters — why the verdict is `REAL` / `MIRAGE` / `Busted`

- **Optimization is "error-maximization".** Richard Michaud, *"The Markowitz Optimization Enigma: Is
  'Optimized' Optimal?"*, **Financial Analysts Journal** 45(1), 1989: mean-variance optimization
  over-weights assets with estimation error in their inputs, so an *estimated* `C⁻¹` produces extreme,
  unstable, out-of-sample-poor weights. The near-singular sample covariance here (condition number
  ~10¹⁷) is exactly that pathology — `weight_instability` measures it, and the optimized book
  underperforming the naive one is the consequence.
- **Covariance shrinkage — the fix that converges to naive.** Olivier Ledoit & Michael Wolf, *"Honey, I
  Shrunk the Sample Covariance Matrix"*, **Journal of Portfolio Management** 2004: shrinking `C` toward a
  structured target stabilises the inverse. The beat-7 complement applies it and finds it only lets the
  optimizer climb back *toward* the naive book — at full shrink (a diagonal `C`) the two are identical.
- **Reversion profits are eaten by the spread.** As in [Study 19](../../19-rubber-band/), a daily
  contrarian book turns over ~daily, so a small gross edge is consumed by transaction costs (the bid-ask
  bounce literature, Roll 1984; Lo–MacKinlay 1990). The `gross_vs_net` and cost figures make this concrete.

## The desk's own method — engine and reproducibility

- **Causal estimation.** Residualisation, the reversion signal and the covariance are all trailing-window
  (no full-sample look-ahead — the trap [Study 22](../../22-crystal-ball/) dissects).
- **Reproducibility.** Headline numbers are pinned with [`quantlab.repro`](../../../quantlab/repro.py);
  the cross-section is built with [`quantlab.universe`](../../../quantlab/universe.py).

## Caveats stated in the open (house rule)

- **Universe capped to estimable.** The sample covariance of N stocks from a window of T days is singular
  for N≳T and unstable well before that, so we cap the universe at the longest-history names — the study
  is *about* that instability, surfaced honestly, not hidden by drowning in 500 names.
- **Flat average-cost haircut.** Cost is charged as the average per-day turnover fee; the conclusion (any
  realistic cost erases the gross edge) is robust to the exact haircut. Survivorship/total-return caveats
  as in the other panel studies.

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
