# Study 384 — ISM-PMI-Regime 🏭

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the PMI>50 regime predict higher stock returns? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The above-50 and below-50 monthly means are **the same number** (+0.958% vs +0.962%): the regime spread is **−0.004%/mo** at **Welch *t* = −0.01** (block-placebo *p* = 0.49), and never clears *t* = 0.6 at any threshold from 45 to 55. The 50 line carries **no** return-timing information — exactly what a coincident growth gauge should. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | "Own SPY only above 50" **ties** buy-and-hold on Sharpe (**0.77 vs 0.76**) while forfeiting **half the terminal wealth** (10.3× vs 24.9×) and 3.5 pts/yr. Its only effect is **less market exposure** — beta you de-risked, not alpha you found. Risk-matched, the edge vanishes. |
| **Free regime-timing lunch?** | ![Busted](https://img.shields.io/badge/Free_regime--timing_lunch%3F-Busted-8b949e?style=flat-square) | "Step out below 50, ride the expansion above it" sounds like avoiding the bad regime for free. But **below-50 months aren't bad** (they average +0.96%, same as above-50) — stepping out just removes good up-drift. The lunch is paid in forgone compounding. |

> **In one sentence:** the manufacturing PMI's 50 line is a real business-cycle marker but a non-event for equities — on a transparent PMI proxy the above-50 and below-50 monthly returns are identical to the basis point (*t* = −0.01), so "only own stocks above 50" is, to the basis point, "own stocks," and the timing rule that sits in cash a quarter of the time just hands back half your terminal wealth for a Sharpe it never beats.

## What we tested

The true ISM Manufacturing PMI is a **proprietary** survey, and the usual free fallback (FRED's `NAPM` / regional-Fed diffusion indices) is unreachable from this environment — so we **construct a transparent PMI proxy**: from a fixed **37-name** industrial/manufacturing basket we build a monthly **diffusion index** (the share of names with positive 3-month momentum, on a PMI-like 0–100 axis, 3-month EMA), labelled a proxy throughout. A reading **> 50** is the "expansion" regime the folklore says you should be invested in. Over **31.2 years** (1995–2026, **375** months) we split next-month SPY returns by the prior-month regime (1-month lag), compare the above-vs-below spread with a Welch *t* and a 20,000-draw **block-bootstrap** null sized to the few, long regimes, and race a deployable "above-50-or-cash" rule against buy-and-hold on Sharpe, drawdown and terminal wealth. A deterministic synthetic control with an *injected* above-50 edge confirms the engine is faithful **and** powered (it lights up only when a real regime edge is planted).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "above 50, own stocks" is just "own stocks," what a diffusion index is, and why sitting out the "bad" regime hands back your wealth — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the PMI proxy, regime-conditional returns, a Welch *t* + block-bootstrap regime-label null, the timing rule's Sharpe/drawdown/terminal-wealth race, costs, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ism_pmi_regime/`](ism_pmi_regime/). The PMI here is an explicit **proxy** (a 37-name industrial diffusion index), not the ISM survey. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
