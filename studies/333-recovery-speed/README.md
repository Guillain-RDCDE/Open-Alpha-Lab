# Study 333 — Recovery-Speed

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Pooled across **four** independent drawdowns (2008, 2020, 2022, 2025), the fastest-recoverers-minus-slowest spread is **+1.3% at 126 days, *t* = +0.23**, bootstrap CI **[−9.4%, +13.0%]** — squarely in the noise, market-neutral. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | There is no gross edge to charge costs against; the spread sits inside its own CI before a cent of cost or borrow. |
| **A "fast recoverers keep leading" rule?** | ![Cherry--picked](https://img.shields.io/badge/Cherry--picked-8b949e?style=flat-square) | It "works" only if you report the **one** drawdown that did (GFC: *t* = +2.78) and ignore the other three — and only on a survivor universe that already stacks the deck in its favour. |

> **In one sentence:** rank names by how fast they climbed out of a crash and the fastest recoverers do *not* keep leading — the headline only appears if you cherry-pick the single deepest drawdown on a universe of survivors, and it vanishes the moment you pool every episode honestly.

## What we tested

The folklore — repeated after every crash — is that the stocks which recover to their pre-drawdown highs **fastest** are the strong hands, and that strength persists: buy the fast recoverers and they keep beating the laggards. We take it literally as a cross-sectional sort. After each market-wide drawdown we detect the trough and recovery on SPY, score every name by how much of its own peak-to-trough loss it had clawed back per day by the recovery date, rank into quintiles, and hold a **market-neutral long-fast / short-slow** book over a forward window — one execution lag, costs one-way × NAV with the short leg paying borrow, and the long-short measured in excess of the market so we never credit recovery-rally beta as alpha. A deterministic synthetic panel with a tunable, plantable recovery-momentum edge is the positive control; a shuffle-the-scores permutation is the null.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story, the one-event mirage shown next to the pooled reality, why survivorship helps the claim and it still fails |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | drawdown detection, the cross-sectional sort, pooled HAC/bootstrap inference, the cherry-pick, the cost sweep, the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`recovery_speed/`](recovery_speed/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
