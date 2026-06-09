# References & literature map — Study 21 (Fools-Gold)

## The source — where this study came from

- **Zura Kakushadze & Juan Andrés Serur, *151 Trading Strategies* (Palgrave Macmillan, 2018).**
  SSRN [3247865](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865); arXiv
  [1912.04492](https://arxiv.org/abs/1912.04492). The relevant entries are **strategies §3.11–3.13
  (one, two and three moving averages)** — the moving-average-crossover family, of which the 50/200
  "golden cross" is the canonical instance.

## The claim under test — the steelman

- **Moving-average timing.** The strongest academic case for crossover/MA timing is Meb Faber, *"A
  Quantitative Approach to Tactical Asset Allocation"*, **Journal of Wealth Management** 2007: a simple
  10-month (≈200-day) moving-average rule that is long when price is above the average and in cash
  otherwise has historically matched or beaten buy-and-hold with much shallower drawdowns, across asset
  classes. This is the steelman — a *risk-managed*, long/flat trend filter, not a magic buy signal.

## The honest counters — why the verdict is `WEAK` / `MIRAGE` / `NOT SUPPORTED`

- **A long/flat rule lowers risk by holding cash — not by skill.** Any rule that sits in cash part of
  the time has a sub-1 beta, so its lower vol and drawdown are mechanical. The fair benchmark is a
  *constant* position at the same average exposure (a cash blend), which `decompose.risk_matched`
  computes — and most of the crossover's calm survives in that blend with no trading.

- **The crossover "works" mainly on the trending index everyone cites.** The golden-cross success
  stories are overwhelmingly about the S&P 500, which had a strong secular uptrend; the same rule is a
  coin flip or worse across other instruments. This is selection: pick the asset after seeing the
  result.

- **And mainly at hand-picked parameters.** `extension.param_grid` shows the fraction of (fast, slow)
  pairs that beat buy-and-hold swings from near-total on the trending index to zero elsewhere — the
  signature of a data-mined parameter, not a robust effect. The broader critique of technical-rule
  data-mining: Ryan Sullivan, Allan Timmermann & Halbert White, *"Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap"*, **Journal of Finance** 1999 (and White's Reality Check, 2000).

- **Even on the S&P, it's a bet on crash timing.** Its long-run edge is dominated by dodging one or two
  bear markets; through the long bull since 2009 it *lagged* buy-and-hold by sitting out the dips. A
  single regime, not a repeatable signal.

## The desk's own method — engine and reproducibility

- **HAC / Newey–West inference.** Newey & West, *Econometrica* 1987 — the golden−death spread *t* and
  the alpha *t* (`cross.signal_value`, `decompose.spread_tstat`, `decompose.vs_buy_hold`).
- **Reproducibility.** Headline numbers are pinned with [`quantlab.repro`](../../../quantlab/repro.py)
  (an as-of date + a content fingerprint of the basket closes).

## Caveats stated in the open (house rule)

- **Split-only closes.** The crossover acts on the price path; the benchmark is charged the same series,
  so the comparison is apples-to-apples. A total-return variant is a fork.
- **A single random walk can drift.** The synthetic null shows that one realisation can spuriously
  trend, making a crossover "work" — which is the data-mining lesson the parameter grid formalises on
  real data, not a bug.

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
