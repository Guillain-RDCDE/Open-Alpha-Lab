# Study 309 — OJ-Frost 🍊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Only **4** freezes fall inside the modern `OJ=F` tape (it starts in 2001), and on all four OJ *fell* over the next week. n = 4 certifies nothing; winter seasonality is insignificant (*t* = −1.36). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Nothing to trade: the famous freezes predate the tape, the in-tape ones go the wrong way, and `OJ=F` is a thin, wide-spread contract. |
| **The *Trading Places* trade?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Great cinema, money-losing finance: on every freeze we can actually measure, buying OJ afterwards lost money. |

> **In one sentence:** the *Trading Places* freeze trade lives entirely in freezes that happened before anyone could pull a daily OJ futures bar off Yahoo — and on the four that remain, buying the freeze lost money every time.

## What we tested

Folklore — popularised by the 1983 film *Trading Places* and grounded in real research
(Roll's 1984 *Orange Juice and Weather*) — says a hard freeze in the Florida citrus belt
destroys the crop, so frozen-concentrate OJ futures spike: corner the OJ market ahead of
the frost and you clean up. We take that literally on the Yahoo `OJ=F` continuous
front-month tape, with a **hardcoded table of severe Florida freezes**, running an
event study around each freeze (reactive entry, one execution lag — you can only act
*after* the cold night), a perfect-foresight ceiling, a random-date placebo control, and
a Dec–Feb winter-seasonality test. A deterministic synthetic tape with a planted
post-freeze spike is the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the film, the freeze table, why the famous freezes aren't even in the data, and what the four that remain actually did |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the event study, the n = 4 small-sample trap, placebo control, perfect-foresight ceiling, winter seasonality, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`oj_frost/`](oj_frost/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
