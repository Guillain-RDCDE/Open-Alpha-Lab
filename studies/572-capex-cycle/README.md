# Study 572 — Capex-Cycle

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> *Do companies in a capex-spending binge underperform the ones harvesting cash — the investment cousin of asset growth?*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Cross-sectional **IC = +0.004** (*t* = **+0.03**) — a dead zero; long-harvest / short-binge hedge **−10.2%/yr** (two-sample *t* = **−0.14**), placebo *p* = **0.88** (buried in the shuffle null). No stable sign across windows, and a yfinance **snapshot** cross-section caps it below REAL regardless. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No gross edge, faint wrong sign; after 5 bps × 4 crossings + a 100 bps short borrow the hedge is **−11.4%/yr net**. The bingeing leg you'd short is exactly the crowded AI-capex tail. Nothing to trade. |
| **Capex cousin of asset growth?** | ![Busted](https://img.shields.io/badge/Channel-Busted-8b949e?style=flat-square) | The capex-*growth* channel is no more present than the total-asset-growth channel of [Study 244](../244-asset-growth/) or the capex-*level* channel of [Study 523](../523-investment-to-assets/) — all three vanish on the survivor large-cap panel. |

> **In one sentence:** the investment cousin of the asset-growth anomaly — capex *bingers* (firms ramping capital spending) earning low future returns — is a flat zero on a 42-name survivor snapshot over 2025-26 (IC +0.004, *t* +0.03, hedge *t* −0.14, placebo *p* 0.88), because the biggest bingers on a survivor basket are the AI-hyperscaler winners (Amazon, Google, Microsoft, Meta) that ripped, not the over-investing failures the anomaly was built around.

## What we tested

Titman-Wei-Xie (2004) and the q-theory literature argue that firms *accelerating* their
capital expenditure earn low future returns — managers over-invest, the market over-extrapolates,
the stock disappoints. This is the **capex-growth / cycle** channel of the investment anomaly:
distinct from total-asset growth ([Study 244](../244-asset-growth/), Cooper-Gulen-Schill) and from
the capex *level* ([Study 523](../523-investment-to-assets/), Titman-Wei-Xie IA), and packaged at
the factor level as Fama-French **CMA**.

We compute **capex_cycle = |CapEx_t|/Assets_{t-1} − |CapEx_{t-1}|/Assets_{t-2}** (a *binge* if
positive) from a yfinance cash-flow snapshot, score a fixed large-cap survivor basket (42 names)
as-of fiscal year 2024, enter one execution lag later (2025-06-27), hold one year, and test the
cross-sectional **information coefficient** between capex_cycle and forward return, a
long-harvest / short-binge tercile hedge with a two-sample *t*, a label-shuffle placebo, explicit
costs + short borrow, a four-window robustness sweep, and a seed-robust synthetic positive control.

The basket is **survivorship-biased** and the yfinance fundamentals are a shallow **snapshot** (only
~4-5 annual statements, and a capex *cycle* costs two of them — so one cross-section, not a deep
panel). Critically, the high-capex-growth group on a survivor basket skews toward *successful*
capex-heavy expanders — AI hyperscalers, semiconductor fabs, energy majors — the opposite of the
over-investing failures the anomaly was built around. (Note 2024-25: the bingers *out*-earned the
harvesters.) Both facts are named on the SIGNAL axis and cap the study below REAL.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the capex-binge-vs-harvest recipe in plain English, why the survivor snapshot erases (even inverts) the sign, window-by-window results |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the cross-sectional IC with a *t*-stat, the tercile hedge two-sample *t*, the label-shuffle placebo, the four-window sweep, costs + borrow, and the seed-robust synthetic positive control |

The fingerprinted real-data run (42 survivors, scored FY2024, forward 2025-06 → 2026-06, panel fp
`5a4b0aebc102`) is in [docs/results.md](docs/results.md); the offline machinery proof runs on the
deterministic synthetic world in [`capex_cycle/data.py`](capex_cycle/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`capex_cycle/`](capex_cycle/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
