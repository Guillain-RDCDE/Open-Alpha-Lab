# Study 531 — Enterprise-Multiple

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Is EV/EBITDA — the practitioner's favourite value multiple — a real edge: do cheap-multiple firms beat expensive ones?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a low EV/EBITDA predict returns? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Long-cheap/short-expensive hedge earns **+8.8%/yr** but at HAC *t* = **+1.11** (\|*t*\| < 2), and a within-month label-shuffle places it at **p = 0.30** — well inside the null. The IC (+0.050, *t* +1.32) has the right sign but is insignificant, and the mean is not stable across the half-split (1st-half *t* −0.07). **Basket = names still trading in 2026 — survivorship-biased upper bound, named here.** |
| **Tradability** — does the spread survive costs? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Turnover is tiny (signal updates annually) so cost is only **4.8 bps/month**: net is **+8.2%/yr at *t* 1.04** — barely dented, but there was no significant gross edge to protect. |
| **Cheap beats expensive?** | ![Mixed](https://img.shields.io/badge/Mixed-8b949e?style=flat-square) | As a *ranking* the value direction shows: the cheap leg out-Sharpes the expensive leg **1.19 vs 0.53** and the IC is positive — but the long-short spread that would monetise it can't be told from zero on this window. |

> **In one sentence:** the enterprise multiple sorts the right way — cheap-EV/EBITDA names out-Sharpe expensive ones 1.19-to-0.53 — but the long-short trade that would cash that in earns a thin +8.8%/yr the tape can't separate from luck (HAC *t* 1.11, shuffle p 0.30), so Loughran-Wellman's value signal lands None × Mirage on a 38-month large-cap survivor basket, exactly where the large-cap factor-decay literature says it should.

## What we tested

**Loughran & Wellman (2011):** rank stocks by the **enterprise multiple** EV/EBITDA
(EV = market cap + total debt − cash; EBITDA from annual fundamentals), go **long the cheap
(low-multiple) tercile and short the expensive tercile**, equal-weight, rebalance monthly. The
multiple is computed per fiscal year and made public only after a 4-month reporting lag (no
look-ahead), with one execution lag. Panel: a fixed **35-name large-cap survivor basket**
(5 banks drop out — EBITDA is meaningless for financials), yfinance fundamentals + monthly
prices, **2023-03 → 2026-05 (38 months)**. We report the hedge with one-sample and HAC *t*,
a 500-draw within-month label-shuffle placebo, costs (5 bps/leg × turnover + 50 bps/yr borrow),
a tercile/quintile sweep and a half-split, and a seed-averaged synthetic positive control that
proves the engine recovers a planted premium. *Distinct from [530 Book-to-Market](../530-book-to-market-value/)
(book equity anchor) and [124 Cash-Flow-Yield](../124-cash-flow-yield/) (OCF/market-cap):
EV/EBITDA adds net debt and uses operating earnings.* **Data limitation:** yfinance carries
only ~4 fiscal years of EBITDA, so the usable window is short — a real constraint on power.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what EV/EBITDA is in plain language, why "buy the cheap multiple" sounds compelling, the cheap-vs-expensive leg race, and the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the monthly hedge with HAC *t*, the label-shuffle placebo, costs + borrow, the tercile/half-split robustness, the monthly IC, and the seed-averaged positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run
(35 names, 2023-03→2026-05, fp `7ba034f010a2`): [docs/results.md](docs/results.md).

---

*Engine: [`enterprise_multiple/`](enterprise_multiple/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
