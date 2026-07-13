# Study 749 — Layoff-Drift 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a layoff pop, then drift? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | **`None` on the pop · `Weak` on the drift.** The "restructuring pop" is **absent** (**−0.34%** over [+1,+3], *t* = **−0.53**, placebo *p* = 0.58). The quarter-long drift is nominally significant (**+9.0%**, Welch *t* = **2.25**, HAC *t* = **3.04**) but **fragile** — drop three cycle-bottom recoveries and it's **+4.4%** at *t* = **1.28**, bootstrap 5th-pct *t* = **0.73** — on a **survivorship**-selected tape whose bias points straight at this result. Significant raw, fragile to selection. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Costs are trivial (net **+8.9%**), so cost isn't the constraint — **repeatability** is. You can't pick the survivors ex-ante, the significance rests on a few macro-timed recoveries (META +55%, XOM +45%, BA +42%), and the selection itself is unavailable at trade time. |
| **"Restructuring pop"?** | ![Busted](https://img.shields.io/badge/Restructuring_pop%3F-Busted-8b949e?style=flat-square) | The headline folklore — a stock *cheering* the announcement — **does not exist** (*t* = −0.53). What drifts up over the next quarter is a fragile, survivor-flattered recovery, the opposite of an instant cheer. |

> **In one sentence:** the legend that a mass-layoff announcement pops on the "cost discipline" news and then drifts up is half-busted, half-mirage — across a transparent table of ~28 large-cap layoff announcements (2015–2025) there is **no pop at all** (−0.3%, *t* = −0.53), and the eye-catching **+9% quarter-long drift** (Welch *t* = 2.25, HAC *t* = 3.04) collapses to *t* = 1.28 the moment you drop three cycle-bottom recoveries and remember the sample is only firms that **survived** their layoffs — so it's real-on-a-survivor-tape, weak-as-a-law, and untradable ex-ante.

## What we tested

We hardcode a **transparent table of ~28 documented large-cap mass-layoff announcements**
— the 2022–2024 tech "efficiency wave" (Meta, Amazon, Google, Microsoft, Salesforce, …)
balanced by industrial, energy and COVID-era cuts (GE, Ford, GM, ExxonMobil, Boeing, …).
Around each dated announcement we measure the **abnormal (market-model) return** vs SPY on
a short **pop** window `[+1,+3]` (the "restructuring pop") and a longer PEAD-style **drift**
window `[+4,+63]`, with a one-day execution lag, then judge each leg with a Welch *t*, a
**Newey-West HAC *t*** on the pooled daily drift, a placebo null sized to the event count,
and a cost-charged book. We name the governing honesty problem loudly: the table is only
firms that **survived** their layoffs — the distressed announcers that delisted leave no
price series — so any positive drift is an **upper bound**, not a tradable law. A
deterministic synthetic control with plantable pop/drift edges confirms the engine is
faithful **and** that ~two dozen events can't reach significance unless the drift is large.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the "cost-cutting cheer" never shows up, what a 3-month drift really is, and why "firms that survived their layoffs went up" isn't a trade — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | market-model pop/drift CARs, a Welch *t* + Newey-West HAC *t* + placebo null, the drop-3 / bootstrap fragility, the survivorship argument, the costed book, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`layoff_drift/`](layoff_drift/). Events are an explicit **hardcoded, labelled table**; the priced tape is **survivor-biased** (distressed layoff-announcers delisted), named on the Signal axis. Prices are yfinance **total-return** (auto-adjusted) closes. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
