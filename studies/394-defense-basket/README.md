# Study 394 — Defense-Basket 🛡️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do defense stocks beat the market after a shock? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Over **20** war/invasion/rearmament headlines, the LMT/RTX/NOC/GD basket's excess over SPY is **negative** at every horizon — **−1.32%** over the trading month (plain *t* = **−1.22**, HAC *t* = **−1.13**, placebo *p* = **0.94**), confirmed by the **ITA** ETF (*t* = −1.32). Not a positive edge — a mild post-headline *lag*. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | There is no positive edge to charge costs against; a market-neutral "buy defense on the headline" trade **loses money gross and net**, and **20 events in 31.5 years** could never be a NAV-scale rule regardless. |
| **Reliable rally?** | ![Busted](https://img.shields.io/badge/Reliable_rally%3F-Busted-8b949e?style=flat-square) | A **vivid-event + base-rate illusion**: Ukraine 2022 (+6%) and post-9/11 (+4%) are remembered; Desert Fox (**−13%**), the Afghanistan invasion (−9%) and the 2024 Iran barrage (−7%) are forgotten. Averaged honestly, defense **trails** the market after shocks. |

> **In one sentence:** the "defense stocks rally on war news" reflex is what survivorship of memory does to a coin flip — across 20 hardcoded geopolitical shocks the defense basket's return *in excess of SPY* is a statistically-insignificant **−1.3%** over the following month (and the famous winners are matched, draw for draw, by forgotten double-digit losers), so the reliable rally is real-as-anecdote and busted-as-edge.

## What we tested

The folklore says defense contractors **reliably rally** when war, invasion or a rearmament cycle hits the tape. We test it as a clean **event study**: take a fixed, hardcoded set of **20 geopolitical-shock dates** (US strikes, invasions, Russia–Ukraine, Israel–Iran…), and measure the equal-weight **LMT/RTX/NOC/GD** basket's cumulative return *in excess of SPY* over the **5/21/63 trading days after** each shock, entering **one day after** the headline (no look-ahead). We judge it with a plain *t*, an autocorrelation-robust **HAC** *t*, a **placebo** null sized to the event count, and a long/short cost charge — and we cross-check on the **ITA** sector ETF. A deterministic synthetic control with an *injected* window edge confirms the engine is faithful **and** that ~20 events can't reach significance unless the planted edge is implausibly large.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a couple of unforgettable rallies become a "law," what the basket actually does after each shock, and why 20 events can't be a strategy — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | event-window excess returns, conditional vs unconditional base rate, plain + HAC *t*, a placebo randomization null, costs, an ETF cross-check, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`defense_basket/`](defense_basket/). Defense = an explicit **4-name equal-weight basket** (+ ITA cross-check), excess measured vs SPY. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
