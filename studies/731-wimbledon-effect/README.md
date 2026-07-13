# Study 731 — Wimbledon-Effect 🎾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No directional seasonal: raw fortnight return **+0.38%**, *t* = **+0.60** (placebo *p* = 0.67); UK-minus-Europe **−0.22%**, *t* = **−1.36** (placebo *p* = 0.42). Both cuts sit in the random-window luck cloud, robust to jackknife and the 2015 schedule shift. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Long-UK net **+0.28%** (*t* = +0.44) is just two weeks of market beta; the only \|*t*\| ≥ 2 number in the study — market-neutral net *t* = **−2.72** — is a *loss* generated entirely by a ~0.22% cost/borrow drag on a *t* = −1.36 gross spread, and the reverse trade nets −0.00%. No direction pays. |
| **A real lull?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Realized vol inside the fortnight equals the surrounding weeks (ratio **1.01**, *t* = +0.09); quieter in exactly **10/20** years — a coin flip. The literal "quiet window" fails on its own terms. |

> **In one sentence:** the Wimbledon "summer lull" is an ordinary two weeks that looks
> exactly like every other two weeks — no drift, no tradable edge, and not even measurably
> quieter — and its single "significant" statistic is a cost-drag artifact that loses money.

## What we tested

City folklore holds that the UK market goes quiet for the Wimbledon fortnight — thin
volume, tennis and strawberries, a sleepy summer lull with a signature you could position
around (the British cousin of *Sell-in-May*). We hardcode all 20 contested Championships
fortnights 2005→2025 (2020 COVID-cancelled) with their exact first-Monday-to-second-Sunday
dates, and measure the UK equity ETF `EWU`'s window return — raw and abnormal (minus the
`VGK` Europe benchmark, which strips out any pan-European summer drift) — across the years,
with a two-sided random-window placebo, a realized-volatility lull test, and a
calendar-known trade costed both long-only and market-neutral. The window dates are
published years ahead, so the whole test is zero-look-ahead by construction.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story, the fortnight-return chart, the vol-lull bust, and the "significant" trade that loses to costs |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the one-sample-*t* cuts, the random-window placebo, the vol-ratio test, jackknife + the 2015 split, the cost-drag anatomy, the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`wimbledon_effect/`](wimbledon_effect/). The Wimbledon calendar is hardcoded
from Wikipedia (every fortnight asserted Monday→Sunday at import). Series are
**total-return** (EWU, VGK); the window is **calendar-known** (no look-ahead). **Not
investment advice** — research & education. See [LICENSE](../../LICENSE).*
