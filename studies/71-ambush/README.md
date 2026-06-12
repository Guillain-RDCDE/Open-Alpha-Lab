# Study 71 — Ambush 🪤

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do four dead edges, firing together, still pay? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes — next-day SPY return climbs **monotonically** with the confluence count (+0.6 → **+42 bp** across 0…4 signals); the ≥3-signal stream earns **+19.6 bp/day** at HAC *t* = **+3.06**, Reality-Check **p = 0.009** over the whole announced family, and the premium is **undecayed** across 2015 (*t*-change +0.17) even though every ingredient died alone. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | The cost moat is wide (break-even ~**7 bp** one-way vs ~1 bp on a US500 CFD) — but at ~15 trades/yr under a 1%-of-NAV daily budget the prize is **+1.2%/yr excess** (net Sharpe +0.42 full, **+0.28 OOS** — under the pre-registered 0.30 bar, CI through zero). |
| **Rarity defeats costs?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | The design worked: spread + financing eat ~**25%** of gross here, vs **>100%** for the same signals traded daily (study 19). The first deliberate cost defence on this bench that held. |

> **In one sentence:** gate four individually cost-dead S&P 500 edges (low IBS, turn-of-month, red close, VIX stress) so the book only fires on their rare confluence, and a real, undecayed, family-corrected premium survives net costs — as a small, honest overlay (~1.2%/yr excess at 7.6% time-in-market), not a fund.

## What we tested

The bench's own inversion, pre-registered before the run ([docs/preregistration.md](docs/preregistration.md)): every short-horizon edge this desk certified gross died **net** for one reason — daily turnover (studies [01](../01-overnight-anomaly/), [03](../03-fear-gauge/), [13](../13-crimson-hour/), [19](../19-rubber-band/), [42](../42-last-call/)) — and *averaging* them fails too ([38](../38-chorus/)). The steelman (Nagel 2012's liquidity-provision premium, strongest under stress): when **three or four fire on the same close**, their stacked premium should survive precisely *because* such days are rare. We test the K∈{1..4} confluence family on SPY 1993–2026 (split-only OHLC + ^VIX), then run the K≥3 book as a CFD: study 16's vol targeting, a hard **1%-of-NAV daily risk budget** with an intraday stop, 1 bp one-way spread plus overnight financing, raced excess-of-cash vs excess-of-cash.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why dead edges leave a live confluence, the ambush ladder, and what 1.2%/yr honestly buys |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the lift table under HAC errors, the no-decay Welch test, White's Reality Check, the costed book and its bootstrap CIs |

The fingerprinted real run is in [docs/results.md](docs/results.md); the frozen protocol in [docs/preregistration.md](docs/preregistration.md). Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py) (machinery proof) and [examples/verify.py](examples/verify.py) (real tape, cache-only; `--fetch` once to populate).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
