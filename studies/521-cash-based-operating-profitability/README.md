# Study 521 — Cash-Based Operating Profitability

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Cash-OP hedge **+10.63%/yr**, one-sample *t* = **+1.75** (|*t*| < 2) on only **3 usable years**, and it **fails** the label-shuffle placebo (*p* = **0.39**). Encouraging sign + strong literature prior (Ball-Gerakos-Linnainmaa-Nikolaev 2016) lift it above `NONE`, but the real tape clears neither half of the bar. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Sharpe **+1.01**, 100% hit-rate, 0% drawdown — all small-sample artefacts of a 3-year window. Net of 0.67%/yr costs the spread is still indistinguishable from a within-year coin-flip (placebo *p* = 0.39). No out-of-sample, no delisted names. |
| **Does cash beat accrual-laden profit? (BGLN)** | ![Mixed](https://img.shields.io/badge/Mixed-8b949e?style=flat-square) | Directionally yes: cash-OP **+10.6%/yr** (*t* +1.75) vs gross-profitability GP/A **+4.1%/yr** (*t* +0.27) — the Ball et al. ordering holds. But on 3 annual points neither leg is significant: support, not confirmation. |

> **In one sentence:** stripping accruals out of operating profitability (Ball et al. 2016) does produce a bigger, cleaner spread than accrual-laden gross-profitability on a large-cap survivor basket — but yfinance gives only ~3 usable annual years, the *t* sits at 1.75, and the spread fails a label-shuffle placebo, so the encouraging direction stays Weak rather than Real.

## What we tested

Ball, Gerakos, Linnainmaa & Nikolaev (2016) "Accruals, cash flows, and operating
profitability in the cross section of stock returns": *cash-based* operating profitability
(accounting operating profit with the accrual block stripped out) predicts returns better
than the accrual-laden gross/operating profitability of Novy-Marx (2013). We compute
cash-OP = (Operating Income + D&A + ΔWorking-capital) / Total Assets from yfinance annual
statements, sort a fixed ~40-name large-cap survivor basket each fiscal year, go long the
top quintile and short the bottom quintile with a one-year reporting lag, charge one-way
costs × turnover plus borrow on the short leg, and run a 500-shuffle label-shuffle placebo.
We benchmark the cash measure head-to-head against the GP/A sibling of [Study 122](../122-gross-profitability).
The basket is survivorship-biased (names still trading in 2026), named explicitly and
treated as an upper bound. A deterministic synthetic panel with a tunable cash-OP premium
serves as the positive control. **Data limitation:** yfinance exposes only ~5 fiscal years
of statements, two of which lack a full forward window — leaving **3 usable hedge years**,
which alone caps any verdict short of REAL.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the BGLN claim in plain language, the synthetic positive control, the cash-OP vs GP/A bars, the honest small-sample verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | one-sample *t*, the label-shuffle placebo distribution, cash-OP vs GP/A head-to-head, the cost/turnover charge, and the 3-year-window caveat |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`cash_op/`](cash_op/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
