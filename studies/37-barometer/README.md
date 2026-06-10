# Study 37 — Barometer 🌡️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the trend in macro data predict returns? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) on the level · ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) on the size | Real but **modest and slow**. Our synthetic cross-asset control recovers the macro-momentum premium (**Sharpe +1.09**, +5.1%/yr, low turnover) and the null is flat (**−0.17**); the literature (Brooks–Moskowitz 2017) agrees it's real but a ~0.4–0.8 Sharpe with long flat stretches. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | It's *cheap to run* — macro signals are slow, so turnover is low and break-even cost is ~91 bp (cost is **not** the threat). The threat is the modest stand-alone Sharpe, the long droughts, and the **episodic** inflation hedge: it pays in rising-inflation regimes (+0.59) more than falling (+0.46), so it's dead weight most of the time. |
| **Real-tape run?** | ![Pre-reg](https://img.shields.io/badge/Pre--reg-8b949e?style=flat-square) | The macro state needs FRED series (INDPRO/PAYEMS/CPIAUCSL/T10YIE) that **time out / are intermittent** in this sandbox. So — like [Study 27](../../27-steamroller/) — the verdict is earned on the synthetic control + literature, and the real run is **pending a reliable FRED fetch** ([docs/results.md](docs/results.md)). |

> **In one sentence:** the trend in fundamental macro data (growth, inflation) is a *real* cross-asset predictor — but a slow, modest, diversifying one, and the inflation-hedge tilt only earns its keep in the rare regimes it targets, so the verdict is `REAL`-but-`WEAK` and `FRAGILE`, with the real-tape run **pre-registered** and pending a reliable FRED macro fetch.

> ⚠️ **Real run pending one fetch.** This study's real tape is **FRED macro series + asset proxies** and has no pre-populated cache — the daily FRED series time out here and even monthly CPI is intermittent. Run `python examples/verify.py --fetch` once a reliable fetch is available to download it and write the fingerprinted [docs/results.md](docs/results.md). The verdict above is earned on the fully-validated synthetic control and the literature; the offline core reproduces via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py).

## What we tested

The desk's idea from Kakushadze & Serur, *151 Trading Strategies* (**§19.2 fundamental macro-momentum** and
**§19.3 inflation hedging**). The steelman (Brooks–Moskowitz 2017 "Macro Momentum"; Neville et al. 2021
"The Best Strategies for Inflationary Times"): the *trend* in fundamental macro data predicts asset returns
— go long the assets favoured by improving growth/inflation momentum, and tilt toward **real** assets
(commodities, TIPS, gold) when inflation is rising, a real, slow, diversifying cross-asset premium. We prove
the engine on a synthetic cross-asset world (equities, bonds, commodities, a TIPS proxy, gold) driven by two
latent, persistent, regime-switching macro states whose *momentum* predicts next-month returns (with a
`macro_strength=0` null that earns nothing), run the macro-momentum and inflation-hedge books, and split the
inflation book by regime to ask whether the hedge actually pays when inflation is rising.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: the barometer, the slow steady macro premium, and the inflation hedge that only earns its keep in a storm |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: macro-momentum Sharpe vs a null, the cost/break-even sweep, and the regime split that tests the conditional inflation hedge |

The real run — every fingerprinted, as-of'd cross-asset number — is **pending a reliable FRED macro fetch**
in [docs/results.md](docs/results.md) (a pre-registration); the **beat-7 worked complement** (does the
inflation hedge pay when inflation is rising?) is in [docs/extension.md](docs/extension.md). Reproduce
offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); the real-tape hook is
[examples/verify.py](examples/verify.py) (`--fetch`).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
