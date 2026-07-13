# Study 758 — TSA-Throughput 🛫

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does accelerating TSA throughput predict the travel trade? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The *direction* is real at long horizons — after TSA accelerates, the 12-month travel-basket return is **+15.4%** vs base **+11.9%** (up-rate **77%** vs **68%**) — but full-sample it **fails t ≥ 2** (best Welch *t* = **+0.82**, placebo *p* = **0.18**), **flips negative** for a 3-month window (*t* = −0.87) and for the biggest upticks (*t* = −1.42), and clears the bar in exactly one spec (12m ex-COVID, *t* = **2.12**). Real-as-lore, weak-as-edge. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | "Long travel when TSA accelerates" earns **+3.1%/yr** (Sharpe **0.12**) vs buy-and-hold **+15.9%/yr** (Sharpe **0.52**); the long/short version **loses 10%/yr**. YoY momentum lags the reopening rallies, so acting on the signal *destroys* return. |
| **Real-time nowcast?** | ![Not_supported](https://img.shields.io/badge/Nowcast%3F-Not_supported-8b949e?style=flat-square) | The lead/lag scan puts the strongest *positive* correlation at **L = −6 / −3** — TSA momentum *lags* the travel trade by a quarter-to-half-year; at every positive lead it's *negative*. Travel equities price the recovery **before** the checkpoints confirm it. The "early" part is what the data rejects. |

> **In one sentence:** accelerating TSA checkpoint throughput really does line up with a strong 12-month travel-basket tailwind, but the tilt is insignificant full-sample (Welch *t* = 0.82), clears the bar only at a single 12-month horizon once you delete the COVID reopening, and — fatally — the checkpoints *lag* the travel trade by a quarter-to-half-year, so travel equities discount the recovery months before TSA confirms it and a "buy travel when TSA accelerates" rule earns +3%/yr against +16% for simply holding.

## What we tested

The alt-data folklore says TSA checkpoint volumes are a free, daily, government-published, *real-time* read on travel demand — so when throughput **accelerates**, the travel sector (airlines + hotels) has a tailwind you can trade before official traffic reports, hotel RevPAR, or earnings confirm it (the "reopening trade" / nowcasting thesis; [TSA's public numbers](https://www.tsa.gov/travel/passenger-volumes)). We rebuild that signal on a monthly TSA-throughput tape and measure forward 1/3/6/12-month returns of an equal-weight travel basket (½ `JETS` · ½ `MAR`+`HLT`) conditional on rising vs falling year-over-year TSA momentum, against the base rate, with a one-month execution lag, a Welch *t*, a placebo null, an explicit **lead/lag** scan (does the uptick actually come *first*?), a market-beta control, and a tradable timing overlay. (TSA's site is firewalled in this build, so the throughput series is a hardcoded, clearly-**labelled proxy** snapshot of the public daily numbers — the COVID-2020 collapse to ~0.1M/day is included faithfully, caveated on the Signal axis.) A deterministic synthetic control with a *planted* TSA→returns link confirms the engine recovers a real edge and can't manufacture one from noise.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "TSA is a real-time travel nowcast" is mostly the *market* nowcasting TSA, what a throughput uptick really tells you, and why buying travel on it loses money — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | ACCEL-vs-base split returns, a Welch *t* + placebo null, the decisive lead/lag cross-correlation, a market-beta control, the timing overlay vs buy-and-hold, robustness (window / threshold / ex-COVID), and a synthetic planted-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`tsa_throughput/`](tsa_throughput/). TSA throughput here is a hardcoded, **labelled proxy** snapshot of TSA's public daily checkpoint numbers (monthly average, millions/day), named as such — never under a real-tape banner. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
