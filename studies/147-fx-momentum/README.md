# Study 147 — FX-Momentum

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Gross **-1.64%/yr**, HAC *t* = **-1.17**; the 12-1 sort actually *underperforms* a random-ranking control by -1.37%/yr, and the bootstrap 95% Sharpe CI is [-0.67, +0.16] (89% negative). The original MSSMS (2012) effect on 48 currencies, 1976-2010 appears absent in G10-only post-publication data. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Gross is already negative; at a realistic 5 bps/leg (institutional FX) the net is **-1.90%/yr** with no positive break-even cost. |
| **Post-publication decay?** | ![Confirmed](https://img.shields.io/badge/Decay-Confirmed-8b949e?style=flat-square) | Negative in all three sub-periods (2006-12, 2013-19, 2020-26); no sub-period shows \|*t*\| > 1.5, consistent with the McLean-Pontiff (2016) post-publication erosion pattern. |

> **In one sentence:** the 12-1 G10 cross-sectional FX momentum factor documented by Menkhoff, Sarno, Schmeling & Schrimpf (2012) earns -1.64%/yr gross on Yahoo Finance spot data from 2006-2026 — statistically indistinguishable from zero, underperforming a random ranking control, and negative gross before any transaction costs.

## What we tested

The Menkhoff-Sarno-Schmeling-Schrimpf (2012) recipe: each month, rank the nine G10 currency pairs (EUR, GBP, JPY, AUD, CAD, CHF, NZD, NOK, SEK) vs USD by their trailing **12-1 month** log-return (skipping the most recent month to avoid short-term reversal bias). Go **long the top 3** and **short the bottom 3** currencies, equal weight within each leg, rebalance monthly. We measure the portfolio against a **random-ranking control** — the identical construction but with i.i.d. random rank assignment each month (averaged over 20 seeds) — so the test is whether the momentum signal adds anything over a coin flip. We sweep transaction costs from 0 to 20 bps/leg/month, examine sub-period performance to test post-publication decay, and verify the machinery with a synthetic panel that has tunable cross-sectional momentum planted deterministically. No look-ahead: signals are formed through month *t*, positions entered at *t+1*.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | plain-language explanation of cross-sectional FX momentum, the honest random control, why the original paper was probably right and the edge has since decayed, what transaction costs do |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t*-stats, block-bootstrap Sharpe CI, sub-period decay analysis, cost sweep with t-stats, synthetic positive control sweep |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`fx_momentum/`](fx_momentum/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
