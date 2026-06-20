# Study 302 — Lithium-Boom 🔋

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | LIT 200-day trend overlay net HAC *t* = **+2.09** (gross +2.24), and the trend timing beats a same-exposure **random-timing control** at the **100th percentile** on all three battery names — the *when* carried information (mostly: get out before the crash). |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Net Sharpe ~**0.5**, bootstrap CI **[0.02, 0.99]** barely clears zero; a single thin thematic sleeve that trails plain **SPY (0.77)**. The edge is risk reduction, not return. |
| **Did riding the boom actually pay?** | ![Misattributed](https://img.shields.io/badge/Misattributed-8b949e?style=flat-square) | The pitch is *boom-riding return*; what you got was *crash-dodging*. (Trend − Buy&Hold) return difference is HAC *t* ≈ **0** (−0.06): it **halved the −66% drawdown** to −41% and added **no** return. |

> **In one sentence:** trend-following the battery-metals theme did carry real timing information — but it paid you in a smaller drawdown, not in the boom-riding return the story promised, and the whole sleeve still earned worse risk-adjusted returns than just holding the S&P 500.

## What we tested

The 2021–22 narrative said the lithium / battery-metals **super-cycle** was a structural, multi-year bull market, and that a trend-follower riding the **Global X Lithium & Battery Tech ETF (LIT)** would have caught the boom *and* — by following the trend — stepped aside before the bust (Global X fund literature; Goldman / Benchmark Mineral Intelligence cycle notes; the canonical trend-momentum of Moskowitz-Ooi-Pedersen 2012 and Faber 2007). We take it literally: hold LIT (plus miners ALB, SQM) when price is above its 200-day moving average, else sit in cash — one execution lag, costs one-way × NAV — and race it against buy-and-hold (excess-of-cash vs excess-of-cash), a random-timing control, and SPY. A deterministic synthetic boom-bust tape with tunable persistence is the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the boom-and-bust chart, why "it dodged the crash" isn't the same as "it paid", in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | net HAC *t*, excess-vs-excess race, random-timing control, bootstrap CI, cost sweep, the SPY opportunity cost, synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`lithium_boom/`](lithium_boom/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
