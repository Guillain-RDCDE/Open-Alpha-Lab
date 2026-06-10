# Study 35 — Contango 🛢️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do backwardated commodities out-earn contangoed ones? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes — roll yield is a documented commodity premium (Gorton–Rouwenhorst 2006; Erb–Harvey 2006; Koijen et al. 2018). Our synthetic control recovers it (high-minus-low roll-yield spread **+27.6%/yr**, gross Sharpe **+1.86**) and the disconnected null is flat (Sharpe **−0.28**). |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | It's *cheap to run* (slow signal, turnover 0.19/wk, break-even ~160 bp — costs aren't the constraint) — but carry is a **volatile, crash-prone** stream that unwinds hard in commodity-wide risk-off, and the premium is biggest in the least-liquid contracts. |
| **Real-tape run?** | ![Pre-reg](https://img.shields.io/badge/Pre--reg-8b949e?style=flat-square) | Roll yield needs the **term structure** (front + deferred contracts), which this sandbox can't fetch — the cache holds only front-month continuous returns. The apparatus, mirage line and expected shape are pre-registered in [docs/results.md](docs/results.md); the run is **pending a curve fetch**. |

> **In one sentence:** the commodity carry premium — long the backwardated curves, short the contangoed — is a real, durable, cheap-to-run edge that is volatile and crash-prone (a `FRAGILE` cousin of the FX carry steamroller), and we've proven the machinery on a synthetic control and pre-registered the real-tape run, which is **pending the term-structure data the sandbox can't serve**.

> ⚠️ **Real run pending a term-structure fetch.** Computing roll yield needs the front *and* deferred contract for each commodity (the slope of the curve); the desk caches only front-month continuous returns, and no free source here serves the deferred leg. The verdict above is earned on the fully-validated synthetic control and the literature — exactly the honesty pattern of [Study 27 (Steamroller)](../../27-steamroller/) before its FRED download. Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); the pre-registered real run is [examples/verify.py](examples/verify.py) → [docs/results.md](docs/results.md).

## What we tested

The desk's idea from Kakushadze & Serur, *151 Trading Strategies* (**§9.1 roll yields**, **§9.4
value/carry in commodities**). The steelman: a long futures position earns a **roll yield** as it slides
along the term-structure curve — positive when the curve is **backwardated** (front > deferred, rolls up),
negative when **contangoed** (front < deferred, rolls down) — so a book long the most-backwardated and
short the most-contangoed commodities harvests a real carry premium (Gorton–Rouwenhorst 2006; Erb–Harvey
2006; Koijen et al. 2018). We prove the engine on a synthetic 12-commodity panel with a *baked* roll-yield
premium (and a disconnected null that earns nothing), run the dollar-neutral carry book, and — since the
real term structure isn't fetchable here — pre-register the real-tape run. It is the commodity sibling of
[Study 27 (Steamroller, FX carry)](../../27-steamroller/) and a cousin of
[Study 29 (Hedgers-Toll, commodity COT)](../../29-hedgers-toll/).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what roll yield is, why backwardation pays, why carry is cheap to run but crash-prone, and what the real run is waiting on |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the carry premium by roll-yield bucket, the control-vs-null, turnover & break-even, and the carry+momentum diversification blend |

The pre-registered real run — every number, once a curve fetch lands — is in [docs/results.md](docs/results.md);
the **beat-7 worked complement** (does a momentum sleeve diversify the carry book? — yes, blend Sharpe
beats either leg) is in [docs/extension.md](docs/extension.md).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
