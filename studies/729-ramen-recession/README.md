# Study 729 — The ramen index: does noodle demand call the downturn? 🍜

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the ramen index lead / hedge downturns? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The tell fails on its own tape: the best "demand-leads-a-downturn" lead-lag correlation is *t* = **−1.05** (+1yr), demand grows the **same** in recession years (+2.0%) as good ones (+1.9%, *t* = +0.02), and the noodle stocks' recession-window edge is insignificant (*t* = **+0.94** / **+1.34**). A CAPM α *does* clear \|*t*\| ≥ 2 vs the Nikkei — but that's survivorship + a weak benchmark, **not** the tell. |
| **Tradability** — could you act on it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A **double look-ahead**: WINA publishes a year's demand ~6 months late and the NBER dates a recession ~12 months late, so the reading always arrives after you needed it. The only market-beating leg means picking the two surviving noodle champions in hindsight. |
| **A leading tell?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Global noodle demand **fell** in the 2008 GFC (−4.2%, −3.8%) and again in 2014–2016 with *no* recession; the only spike was COVID-2020 (+9.6%, pantry-loading). Instant noodles are a secular Asian growth story, not a business-cycle instrument. |

> **In one sentence:** the "ramen index" doesn't lead anything — global noodle demand carries no cycle information (it actually *fell* in the last real recession), the two noodle stocks fell in 2001 and 2008 too and only clearly "won" in COVID, and the one number that beats the market is two hand-picked survivors enjoying the low-beta staple premium against a moribund Nikkei — a signal you could never have front-run.

## What we tested

The folklore — a cousin of the "lipstick index," popularised in 2008 coverage that *"in a
recession, people eat more instant ramen"* — says noodle demand is a **leading downturn tell**:
cheap inferior good, sales climb *before* the crash, and the noodle makers are a defensive
hide-out. The testable version is specific: (H₁) demand growth should **lead** a weak market
(a negative lead-lag correlation at a positive lead), and (H₂) the noodle **stocks** should
**beat the market in recessions**. We test the "ramen index" — the **WINA** world instant-noodle
demand series (2005–2024, a **hardcoded, cited, approximate** *labelled proxy*) — for the lead,
and the two tradable makers **Nissin (`2897.T`)** and **Toyo Suisan (`2875.T`)** against the
**Nikkei 225 (`^N225`)** on month-end yfinance data, over the three **NBER** recessions, with
bull/bear beta, a Newey-West CAPM alpha, the recession-window paired *t*, and the double
look-ahead the pitch never charges.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "ramen index" feels true, the demand line that *falls* into the 2008 recession, the lead-lag bars with no signal, and the survivorship twist — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the lead-lag cross-correlation, bull/bear beta, Newey-West CAPM alpha (and why the significant one isn't the claim), the recession-window *t* with per-recession provenance, and planted-lead + planted-defensive positive controls |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ramen_recession/`](ramen_recession/). Stock prices are real (yfinance, split+dividend adjusted — *total-return-ish*); the "ramen index" is the **WINA** world-demand series (**hardcoded, cited, approximate** — a labelled proxy, not a live feed) and the NBER windows are **cited official dates**. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
