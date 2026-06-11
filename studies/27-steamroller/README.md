# Study 27 — Steamroller 🚧

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do high-rate currencies out-earn? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The premium is there but thin on the real 2001–2024 G10 tape: lagged rate-bucket spread **+2.8%/yr**, carry book **+0.8%/yr** net at Newey–West *t* = **+0.9**, Sharpe **+0.22** (bootstrap 95% CI **[−0.25, +0.72]**). The right slope, decades of evidence behind it (Lustig–Verdelhan 2007; Menkhoff et al. 2012) — but this post-2000 sample alone can't reject zero, exactly the decay the literature documents. The synthetic control proves the machinery (*t* **+2.8** on a baked premium, flat on the full-UIRP null). |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | It *is* cheap to run — real turnover **0.55×/yr**, Sharpe still **+0.08** at a punitive 100 bp — but the edge is thin and it carries a fat negative tail: monthly skew **−0.90**, worst month **−5.8%** (Oct 2008), max drawdown **−15%** on a 3.6%-vol book. |
| **Crash risk?** | ![Severe](https://img.shields.io/badge/Severe-8b949e?style=flat-square) | The crash *resists* the desk's usual fix: on the real tape vol-targeting **cuts** the Sharpe (**+0.20 → +0.11**) and **deepens** the drawdown (**−15% → −26%**) — it levers you *into* the jump, because the crash is a sudden risk-off unwind, not a forecastable volatility build-up. |

> **In one sentence:** the carry trade is a durable, cheap-to-run premium that ran thin after 2000 — and it is rent paid for standing in front of a steamroller: a sharply negative-skewed crash that arrives all at once and shrugs off the vol-management that tamed the desk's other crashes.

## What we tested

The desk's tenth idea from Kakushadze & Serur, *151 Trading Strategies* (strategy **§8.2**, the FX carry trade). The steelman: borrow a low-interest currency, lend a high-interest one, and pocket the gap, because uncovered interest-rate parity fails (the high-rate currency doesn't depreciate enough to offset its yield). We prove the engine on a synthetic G10 with a *baked* carry premium punctuated by sticky risk-off crashes (and a full-UIRP null that earns nothing), then measure the verdict on the **real G10 tape** — OECD 3-month short rates + FX, 270 months 2001–2024, the same shared cache as [Study 36](../36-greenback/) — and show, uniquely on this desk, a crash that the vol-targeting overlay from [Study 16](../16-storm-shy/) can't dodge.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: steady nickels from the rate gap, the steamroller crash, and why risk management can't dodge it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: the carry premium by (lagged) rate bucket, the Newey–West *t*, the negative-skew/downside-concentration crash, and the vol-managed comparison |

The real run — every fingerprinted, as-of'd G10 number — is in [docs/results.md](docs/results.md); the **beat-7 worked complement** (why vol-targeting makes the steamroller *worse* on the real tape) is in [docs/extension.md](docs/extension.md). Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py) (synthetic control) and [examples/verify.py](examples/verify.py) / [examples/extension.py](examples/extension.py) (real tape, from the shared desk cache; `--fetch` only refills a missing cache via [tools/fetch_altdata.py](../../tools/fetch_altdata.py)).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
