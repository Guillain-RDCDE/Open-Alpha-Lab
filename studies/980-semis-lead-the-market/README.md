# Study 980 — The Silicon Canary 🧫

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does semiconductor relative strength predict the market? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Once the market factor is regressed out, the canary is much quieter than the folklore. At lag 1 the semiconductor residual leads the market with correlation **+0.044** while the market leads it by -0.088 — a difference of **+0.132**. The headline predictive regression (trailing 21-day residual → next 21-day market return) gives a slope of +2.35 with HAC *t* = **+1.59** and an R² of 1.044%; **0 of 2** semiconductor funds clear |*t*| = 2, and **7** of the 12 lookback × horizon cells do, against 0.6 expected by luck. The non-semiconductor control (XLK) scores +1.60 — if the effect were about chips rather than about tech, that number should be small. |
| **Tradability** — does anything survive costs and an execution lag? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | The rule — own the market while the canary's trailing relative strength is positive, hold bills otherwise — was invested 55% of the time, switched 25.2 times a year, and returned **+10.19%/yr** against **+10.70%/yr** for simply owning the market (-0.51%/yr, Sharpe +0.74 vs +0.61, HAC *t* on the daily difference -0.47). Its drawdown was -31.4% against -55.2% — the risk reduction is real; the return is not. |

> **In one sentence:** Semiconductors do move first — by about a day, with a residual lead-lag difference of +0.132 — but the signal is far smaller than the story, it is not distinguishable from a general tech lead, and the rule it implies gave up 0.51% a year against simply owning the market.

## What we tested

"Semis lead the market" is desk lore with an economic story attached: chips go into
everything, so chip demand turns before the cycle does. The lore is testable, and the obvious
test is wrong — a semiconductor ETF is roughly {the market plus a tilt}, so correlating it with
the market's next move mostly measures the factor they share. We remove that factor two ways (a
plain sector-minus-market difference, and a **rolling backward-looking beta** residual), then
measure the lead-lag in **both directions** — because a "leading indicator" whose reverse
statistic is the same size is a co-movement — and run predictive regressions with
horizon-lag HAC errors across a lookback × horizon grid, counting the cells that clear |*t*| = 2
against the number luck provides.

Two controls decide the study. **SOXX** is a second semiconductor fund tracking a different
index: an effect that appears in one and not the other is a statement about an index committee.
**XLK** is a technology fund that is not a semiconductor fund: if it predicts the market as well
as the chips do, the claim being supported is "tech leads", which is older and less interesting.
Finally the folklore is priced as the rule it implies — own the market while the canary is
strong, hold bills otherwise — against buy-and-hold, with an execution lag and costs.
**Dedup:** distinct from **870-industry-leader-lead-lag** (a single large *stock* leading its
own industry), **634-us-leads-the-world** and **981-asia-tech-canary** (cross-country
lead-lag), **225-sector-rotation** and **28-carousel** (choosing *between* sectors),
**393-ai-datacenter-basket** (a thematic basket's returns) and **131-utilities-canary** (a
defensive sector as a *risk-off* signal rather than a cyclical one).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a sector ETF cannot be tested against the market naively, what is left after the market factor is removed, and what the timing rule actually returned |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | rolling-beta residuals, two-directional lead-lag, horizon-lag HAC regressions over a full grid with the multiplicity count, peer and tech controls, the lookback sweep, era cut and a planted-lead simulation |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`semi_lead/`](semi_lead/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
