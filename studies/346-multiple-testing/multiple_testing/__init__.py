"""Study 346 — Multiple-Testing: the p-value-correction bake-off.

If you test 100 calendar effects, some will clear *t* > 2 by luck alone. This study is a
**methodology demo**: it puts the four ways of dealing with that — the naive |*t*| > 2
count, **Bonferroni** (control the family-wise error rate), **Holm** (the uniformly more
powerful step-down FWER procedure), and **Benjamini-Hochberg** (control the *false
discovery rate*) — on the same battery of effects and asks *which discoveries you can
trust*.

Distinct from its lot-siblings:

- **[Study 343 — Data-Mining-Roulette](../../343-data-mining-roulette/)** spins random
  *rules* and runs Bonferroni + White's Reality Check on the *single best* of N. It asks
  "does luck mimic skill?" on the maximum statistic.
- **[Study 344 — Backtest-Overfitting](../../344-backtest-overfitting/)** grid-searches one
  strategy's parameters and uses the Deflated Sharpe Ratio + PBO.

Study 346's distinct angle is the **correction-procedure comparison itself**, on a
*battery of many named, interpretable effects* with a *known mix of true positives and
nulls* — the FWER-vs-FDR trade-off, not the best-of-N maximum. The output is a
methodology verdict: how many of the naive "winners" survive each correction, and which
procedure you should actually report.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
