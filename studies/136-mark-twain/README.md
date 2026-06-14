# Study 136 -- Mark-Twain

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | October mean **+2.9 bps/day**, HAC t = **+0.95**; Welch t vs the rest = **-0.07**. October is the 5th-best month. Strip the three famous crashes and it rises to **+5.7 bps/day**. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No signal, nothing to trade. Avoiding October means sitting out a positive-return month. The cash-in-October rule underperforms buy-and-hold over a century. |
| **October dangerous?** | ![Myth](https://img.shields.io/badge/October_dangerous%3F-Myth-8b949e?style=flat-square) | Three anecdotes (1929, 1987, 2008) are the entire story. September (mean -5.5 bps/day, HAC t = -2.0) is the actual worst month. Permutation p = 0.47: October sits at the median. |

> **In one sentence:** the S&P 500's October reputation is a three-anecdote survivorship myth -- the month earns a positive mean return, sits fifth of twelve in the rankings, and loses its apparent drag entirely when the three famous crash months are removed; September is the genuine underperformer.

## What we tested

*Mark Twain's 1894 joke -- "October is one of the peculiarly dangerous months to speculate in stocks" -- is not the claim; the serious version is.* We steelman it: October equity returns are reliably negative or significantly below other months, October intraday volatility is structurally elevated, and both features are robust to removing the three canonical crash events (1929, 1987, 2008) from a 95-year sample. We test this on ^GSPC daily returns from 1928 to 2026 (n=24,727 trading days), using a Welch t comparing October vs the pooled other-eleven-months, a HAC t-stat within each month, a permutation null shuffling month labels 5,000 times, and a crash-strip robustness pass. A deterministic synthetic tape with a tunable October penalty serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | monthly bar chart, the crash-strip result in plain language, the permutation check, why the volatility story is one event |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-month HAC t table, Welch t with crash-strip, permutation null distribution, synthetic positive control sweep, avoid-October equity curve |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`mark_twain/`](mark_twain/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
