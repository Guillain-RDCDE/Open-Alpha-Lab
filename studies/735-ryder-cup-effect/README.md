# Study 735 — Ryder-Cup-Effect ⛳

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the losing continent's market lag the winner's the Monday after? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The paired loser-minus-winner Monday spread is **+0.38%**, *t* = **+1.82** — the **wrong sign** (the loser slightly *out*performed), not significant, loser lagged in only **2 of 10** editions. Placebo left-tail *p* = **0.95**; the 1-week spread is *t* = −0.10. |
| **Tradability** — could you deploy it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The only tradable direction (long winner / short loser, entered after the result is public) nets **+0.32%** over a week at *t* = **0.63**, placebo *p* = 0.18 — noise, on an "effect" whose sign is backwards to begin with. |
| **Cross-Atlantic sentiment shock?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The Edmans "loser slumps" mechanism isn't there: the losing side's own market is a flat **−0.17%** (*t* = −0.82) and the *winner's* leg fell more. Wrong-sign spread, placebo, and leg decomposition all agree — no mood shock, either side. |

> **In one sentence:** across the 10 Ryder Cups with tradable coverage (2006→2025), the
> losing continent's stock market does **not** lag the winner's the Monday after — if
> anything it leans the other way (a non-significant +0.38%), the loser's own market
> never slumps, and the whole thing evaporates against a random-calendar placebo: the
> real football *loss* effect does not carry over to a biennial golf match.

## What we tested

Edmans, García & Norli (2007, *JF*) showed a country's stock market really does fall when
its football team is knocked out of the World Cup — a genuine *loss*-driven mood shock
(with, notably, **no** symmetric win effect). The Ryder-Cup folklore extrapolates that to
golf's biennial USA-vs-Europe team match: the *losing continent* should underperform the
Monday after. We hardcode all 23 modern editions 1979→2025 (1989 was a 14-14 tie, no
loser; 2001/2020 postponed), map Team USA → `SPY` and Team Europe → `VGK`, and measure the
**paired loser-minus-winner** abnormal return from the last close before the Sunday result
through the Monday and the week after — with a one-sample *t* across the 10 independent
events that clear VGK's 2005 coverage floor, a random-window placebo, a costed
zero-look-ahead capture test, and a constant-mean leg decomposition to ask whether the
*loser* actually slumps or the winner is just weak.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the real football effect it borrows, why the Monday spread points the *wrong* way, and why the loser's market never actually slumps |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the paired one-sample-*t* battery, the left-tail placebo, the leg decomposition, the USA-vs-Europe asymmetry check, the costed capture, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ryder_cup_effect/`](ryder_cup_effect/). The Ryder Cup calendar is hardcoded
from Wikipedia; **selection named on the Signal axis**: VGK's 2005 inception limits the
test to the 2006→2025 editions (10 events, 7 USA-losses to 3 Europe-losses), and both
legs are total-return `SPY`/`VGK`. **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
