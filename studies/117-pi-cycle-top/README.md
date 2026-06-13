# Study 117 — Pi-Cycle-Top

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | n=**2** signals on Yahoo's tape (2017-12-16, 2021-04-11); the 2024 cycle never triggered the indicator. With n=2 and BTC's ~56% annual vol, no inference is possible regardless of how dramatic the forward returns look. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Fires once per ∼4-year cycle at most; missed the 2024 cycle entirely; rests on a multiplier (2.0) that was hand-fitted to two prior tops. |
| **Busted: a curve-fit to 2 tops?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | At mult=2.2 the indicator *never fires*; at 1.8 it fires 3 times including a false positive before a +300% rally. The "π" in the name is decorative arithmetic on a tuned parameter. |

> **In one sentence:** the Pi-Cycle-Top indicator fired twice in BTC history (both preceding crashes), never fired in 2024, and is a textbook curve-fit to n=2 tops dressed in numerology — the multiplier 2.0 was selected to hit known highs, not derived from π.

## What we tested

The Bitcoin Pi-Cycle-Top indicator: when the 111-day moving average crosses above 2× the 350-day moving average, it allegedly calls the Bitcoin cycle top within days. The "π connection" is that 111/35 ≈ π (or 111/(2×350) ≈ π/20), presented as a deep mathematical property of BTC cycles. We take this seriously: we compute every Pi-Cycle crossover on the full Yahoo BTC-USD daily tape (2014-09-17 to 2026-06-13), measure the forward log-returns at 30, 60, and 90 days, pin them against a **random-date control** (2,000 Monte Carlo draws), decompose the signal count's sensitivity to the multiplier parameter, and ask the overfitting question head-on: did the 2024 cycle confirm or break the pattern? A deterministic synthetic tape with tunable cycle-top structure serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the indicator looks like on a BTC chart, the two signals in plain language, the 2024 miss, why "π" is window-dressing |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | forward-return tables, HAC t-stats, random-day p-values, multiplier sensitivity sweep, power analysis (n required for detection), synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`pi_cycle_top/`](pi_cycle_top/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
