# Study 88 — Dogs-of-the-Dow 🐕

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Over **26 point-in-time years (2000–2025)** the Dogs beat the Dow by **+0.98 pts/yr** — but the annual excess has **HAC *t* = +0.59** and a bootstrap CI of **[−2.15%, +3.59%]** (*p* = 0.51): on this tape it's indistinguishable from noise. Alpha **+2.58%/yr** but *t* = **1.59** (sub-2), and **beta = 0.80** — much of the "edge" is a defensive value tilt. Named survivorship caveat (Kodak, Walgreens). |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Turnover is low (~23%/yr) so trading cost barely bites — but the basket is deliberately high-yield, so a taxable investor pays a real **dividend-tax drag** every year, and the trailing-yield screen is a **dividend-trap magnet** (Jan-2009: Citi at 22.8% yield, BofA at 20.8% — both about to cut). A ~1-pt gross gap that isn't statistically real doesn't survive that. |

> **In one sentence:** buying the ten highest-yielding Dow stocks each January did edge the Dow by ~1 point/year over 26 honestly-point-in-time years — but the gap **doesn't clear a significance bar**, most of it is a **lower-beta value tilt** you can get elsewhere, and what's left gets eaten by **taxes and dividend traps**.

## What we tested

The retail classic from O'Higgins & Downes' *Beating the Dow* (1991), stated at full
strength: *"each January, buy the 10 highest-dividend-yield stocks in the Dow 30,
equal-weight, hold a year, repeat — and you'll beat the Dow itself."* We take it literally —
total-return prices, the trailing-12-month yield measured at the prior December close, the
top ten equal-weight, held the calendar year, **20 bps** on turnover — and race it against
the **same-basis** Dow benchmark (DIA total return; no price-only-vs-total-return trick). The
data-design that decides honesty: the Dow's membership **changes**, so we encode a
**point-in-time** membership timeline (anchored 1999-11-01) and pick each year's Dogs **only
from the members as of that January** — never today's 30 back-dated. The sample is the
**26 years (2000–2025)** we can do correctly; two members with no recoverable tape (Kodak,
Walgreens) are **left out and disclosed, not silently capped**. A deterministic synthetic
panel (yield-forecasts-return vs yield-is-noise) is the positive/negative control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the rule, the point-in-time twist, the equity curves, the ~1-pt gap and why it's not a free lunch, the dividend-trap basket of 2009 |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t* and block-bootstrap on the annual excess, the alpha-vs-beta read, the tax/turnover capacity story, the survivorship handling |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`dogs_of_the_dow/`](dogs_of_the_dow/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
