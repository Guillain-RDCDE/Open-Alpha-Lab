# References & literature map — Study 22 (Crystal-Ball)

## The source — where this study came from

- **Zura Kakushadze & Juan Andrés Serur, *151 Trading Strategies* (Palgrave Macmillan, 2018).**
  SSRN [3247865](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865); arXiv
  [1912.04492](https://arxiv.org/abs/1912.04492). The relevant entry is **strategy §8.1 (moving
  averages with an HP filter)** — detrend a price with a Hodrick–Prescott filter and trade the moving
  average / mean reversion of the resulting cycle.

## The tool under test

- **The Hodrick–Prescott filter.** Robert Hodrick & Edward Prescott, *"Postwar U.S. Business Cycles: An
  Empirical Investigation"*, **Journal of Money, Credit and Banking** 29(1), 1997 (circulated 1980s).
  The standard trend/cycle decomposition `τ = (I + λD′D)⁻¹ y`. It is, by construction, a **two-sided**
  smoother: the trend at each date is a weighted average of the *entire* series, future included.

## Why this is a trap, not an edge — `NONE` / `MIRAGE` / `BUSTED`

- **The HP filter's endpoint / look-ahead problem is well known in macro-econometrics.** James Stock &
  Mark Watson and others have long warned that the HP filter is unreliable at the sample endpoint and
  uses future data; most pointedly, James Hamilton, *"Why You Should Never Use the Hodrick–Prescott
  Filter"*, **Review of Economics and Statistics** 100(5), 2018, documents the spurious dynamics it
  manufactures. Trading the two-sided cycle turns that endpoint problem into a full-blown look-ahead
  bias at *every* date.

- **Look-ahead / data-snooping bias more broadly.** The general failure mode — using information not
  available at decision time — is the central hazard of backtesting (e.g. David Bailey, Jonathan Borwein,
  Marcos López de Prado & Qiji Zhu, *"Pseudo-Mathematics and Financial Charlatanism"*, **Notices of the
  AMS** 2014; López de Prado, *Advances in Financial Machine Learning*, 2018, on backtest overfitting).
  The HP case is a clean, demonstrable instance: the same family includes full-sample z-scores,
  two-sided band-pass filters, and any regression fit on the whole sample.

- **The decisive tests are mechanical, not statistical.** (i) Run the strategy on a **random walk**,
  where no tradable structure exists — any positive Sharpe is artefact. (ii) **Perturb a future price**
  and check whether a past signal value moves (it does, for any two-sided transform). (iii) Correlate
  the signal with **future** returns. `decompose.future_leakage` and the tests implement all three.

## The honest control

- **The one-sided (causal) HP filter** — the endpoint of an HP filter on a trailing window — uses only
  past data and *is* tradable. On a genuinely mean-reverting tape it recovers a small real edge (so it
  is not broken); on a random walk and on real ETFs it earns nothing, which is the verdict.

## The desk's own method — engine and reproducibility

- **HAC / Newey–West inference.** Newey & West, *Econometrica* 1987 — the *t* on each book's stream.
- **Reproducibility.** Headline numbers are pinned with [`quantlab.repro`](../../../quantlab/repro.py)
  (an as-of date + a content fingerprint of the basket closes).

## Caveats stated in the open (house rule)

- **Split-only closes, log space.** The filter and rule act on the log-price path; both the two-sided
  and one-sided books are charged the identical series, so the *only* difference between them is what
  data the filter is allowed to see — which is the whole point.
- **The one-sided HP is a rolling-window endpoint**, not the Kalman-filtered local-linear-trend; a
  state-space causal HP would be smoother. We bet no real edge appears either way — a stated fork.

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
