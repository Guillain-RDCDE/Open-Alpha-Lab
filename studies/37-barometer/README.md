# Study 37 — Barometer 🌡️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the trend in macro data predict returns? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Every cut lands on the predicted side — an always-long real-asset basket out-earned nominal bonds **when inflation rose** (+1.8%/yr) and not when it fell (−0.3%/yr) — but none clears the desk's bar: that spread is Sharpe **+0.10**, *t* **≈ 0.3** (104 rising months), and the standalone macro-momentum book is flat (net Sharpe **−0.05**, HAC *t* −0.22) on the short post-2007 sample (217 months). Right-sided direction at *t* < 2 earns `WEAK`, not `REAL`. (The synthetic control's Sharpe +1.09 vs null −0.17 proves the *machinery*, not the market — and the "growth" driver is the yield-curve slope, a market price, not a fundamental release.) |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | It's *cheap to run* — turnover is low (5-7×/yr) and the cost sweep is linear — so **cost is not** the threat. The threat is that, gross of cost, the timed monthly books make ~nothing on one short cycle: neither beats a passive equal-weight (+0.57) or 60/40 (+0.84) hold of the same assets, and the inflation hedge only earns its keep in the rare rising-inflation regime. |
| **Real-tape run?** | ![Done](https://img.shields.io/badge/Done-8b949e?style=flat-square) | Run on the desk's cached tape — 18 cross-asset ETFs + CPI-YoY / yield-curve-slope macro, **2007-02 → 2025-02**, all macro lagged one month — offline and fingerprinted (`baa416a9db25`) in [docs/results.md](docs/results.md). |

> **In one sentence:** the trend in macro data points the right way in every cut of the real tape — real assets beat nominal bonds when inflation rises (+1.8%/yr, though only *t* ≈ 0.3) — but it is slow and small, nothing clears the desk's significance bar, the timed monthly books don't beat a passive hold, and the inflation hedge only pays in the regime it targets, so the verdict is `WEAK` and `FRAGILE`.

> **Real run done.** The headline numbers are from `python examples/verify.py` on the cached macro + ETF tape (offline, fingerprinted [docs/results.md](docs/results.md), 2007-2025). The offline machinery proof (control vs null) reproduces via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py).

## What we tested

The desk's idea from Kakushadze & Serur, *151 Trading Strategies* (**§19.2 fundamental macro-momentum** and
**§19.3 inflation hedging**). The steelman (Brooks–Moskowitz 2017 "Macro Momentum"; Neville et al. 2021
"The Best Strategies for Inflationary Times"): the *trend* in fundamental macro data predicts asset returns
— go long the assets favoured by improving growth/inflation momentum, and tilt toward **real** assets
(commodities, TIPS, gold) when inflation is rising, a real, slow, diversifying cross-asset premium. We run
it on the real tape — 18 liquid cross-asset ETFs plus a macro state of CPI-YoY (inflation) and the
yield-curve slope (growth), all lagged one month for publication delay, 2007-2025 — and prove the engine on
a synthetic cross-asset world (equities, bonds, commodities, a TIPS proxy, gold) driven by two latent,
persistent, regime-switching macro states whose *momentum* predicts next-month returns (with a
`macro_strength=0` null that earns nothing). We run the macro-momentum and inflation-hedge books, benchmark
them against a passive hold, and split the inflation book by regime to ask whether the hedge actually pays
when inflation is rising.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: the barometer, the slow steady macro premium, and the inflation hedge that only earns its keep in a storm |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: macro-momentum Sharpe vs a null, the cost/break-even sweep, and the regime split that tests the conditional inflation hedge |

The real run — every fingerprinted, as-of'd cross-asset number — is in [docs/results.md](docs/results.md);
the **beat-7 worked complement** (does the inflation hedge pay when inflation is rising?) is in
[docs/extension.md](docs/extension.md). Reproduce the real book via
[examples/verify.py](examples/verify.py) and the offline machinery proof via
[examples/run_synthetic_demo.py](examples/run_synthetic_demo.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
