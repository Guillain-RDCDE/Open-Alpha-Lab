# Study 133 — Crypto-Seasonality

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | October HAC *t* = **+2.99**, September *t* = **−2.59** — real-looking, but **0/12 months** survive the Bonferroni-corrected bar of \|t\| ≥ 3.1 for 12 simultaneous tests. The signal exists but is not robust to multiple comparisons. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Oct-only long/flat Sharpe +0.68 vs buy-and-hold +0.62; differential *t* = **−1.29**. No seasonal rule reliably beats holding BTC all year. |
| **Multiple comparisons?** | ![Failed](https://img.shields.io/badge/Multiple_comparisons%3F-Failed-8b949e?style=flat-square) | Testing 12 months expects ~0.6 false discoveries at alpha=0.05; finding 3 naive-significant months is consistent with chance. The calendar narrative is built on post-hoc selection. |

> **In one sentence:** 'Uptober' and 'Rektember' are real-looking on ~12 data points each, but zero months survive the multiple-comparison correction for searching all 12, and no seasonal long/flat rule reliably beats buying and holding BTC all year.

## What we tested

Crypto folklore assigns strong personality to calendar months: October is 'Uptober'
(almost always bullish), September is 'Rektember' (reliably bearish), Q1 is the
altcoin season. We take the sharpest testable version: *do any calendar months
have reliably different Bitcoin returns vs the all-month average, after accounting for
the fact that we're testing 12 months simultaneously?* We use BTC-USD daily closes
from Yahoo Finance (2014-09-17 to 2026-06-14, ~12 observations per month), measure
each month's HAC t-stat vs the pooled baseline, apply a Bonferroni correction for
12 tests (threshold |t| ≥ 3.1), and separately test whether a seasonal long/flat
rule beats buy-and-hold on a risk-adjusted basis. October (+15.1% mean, *t* = +2.99)
and September (−3.4% mean, *t* = −2.59) both look real — but neither clears the
multiple-comparison bar, and the seasonal rule earns a differential t-stat of −1.29
vs B&H. A synthetic positive control confirms the engine detects planted effects; the
null result is about the market, not the method.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the monthly means chart, the Bonferroni correction explained plainly, the October scatter, why the seasonal rule fails B&H |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, power curve (what effect size we'd need), seasonal strategy vs B&H with differential t-stat, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`crypto_seasonality/`](crypto_seasonality/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
