# Study 18 — Dull-Roar 🐢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do calm stocks really out-earn wild ones, risk-adjusted? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The *mechanism* is real — our synthetic control (a baked flat security-market line) recovers a beta-neutral alpha of **+7.4%/yr** at HAC *t* ≈ **5.6**, and a fair-CAPM null is flat. But on the **current S&P 500** (2010→) the SML slope is **−0.07**, and the beta-neutral Frazzini–Pedersen long-short earns **−2.2%/yr** (*t* = **−0.4**) — a statistical zero. Real in the long-run literature, decayed-to-absent in the sample you'd actually trade. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The clean spread needs **shorting the high-vol decile** — small, lottery-like, expensive-to-borrow names — and **81%** of what little spread exists sits in exactly that leg. The dollar-neutral book's borrow break-even is **0 bps/yr**: under water before the short even costs anything. |
| **Free alpha?** | ![Beta-tilt](https://img.shields.io/badge/Beta--tilt-8b949e?style=flat-square) | What survives is a long-only low-vol book that is mostly *lower beta* (**0.59**): its CAPM alpha is **+1.6%/yr** (*t* = +0.8), and levered back to the market's risk the excess is a thin **+2.1%/yr**. The one durable benefit is a gentler ride (drawdown **−38% → −34%**) — what real min-vol funds quietly sell. |

> **In one sentence:** the most-cited free lunch in finance is real in the textbooks and on our control, but on the modern, investable cross-section it collapses to a low-beta defensive tilt whose tradable long-short form is fragile to exactly the short-leg borrow it can't avoid.

## What we tested

The desk's first idea pulled from a *book* — Kakushadze & Serur, *151 Trading Strategies* (strategy **§3.4**, the low-volatility anomaly). The steelman, at full strength (Ang–Hodrick–Xing–Zhang 2006; Baker–Bradley–Wurgler 2011; **Frazzini–Pedersen, "Betting Against Beta", *Journal of Financial Economics* 2014**): sort stocks by past volatility and the calm decile out-earns the wild one on a risk-adjusted basis — the security-market line is *too flat*, because leverage-constrained investors overpay for high-beta excitement. We prove the apparatus on a synthetic universe with a *baked-in* flat SML (and a fair-CAPM null that must — and does — kill every leg), then point the same tools at the current S&P 500 cross-section (daily total-return closes, monthly-rebalanced decile books, equal-weight universe benchmark).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: why calm-beats-wild is *almost* true, the trap in the naive long-short, why "buying calm" is mostly buying less beta, and the edge that quietly went missing |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: the SML slope, the beta-neutral BAB alpha with HAC errors, the beta-tilt counter, leg attribution, the borrow break-even, and the fair-CAPM null |

The real run — every fingerprinted, as-of'd S&P 500 number — is in [docs/results.md](docs/results.md); the **beat-7 worked complement** (the *no-shorting test* — forbid the short and sweep the knobs; the salvageable long-only slice is a thin defensive tilt across every setting) is in [docs/extension.md](docs/extension.md). Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); on the real tape via [examples/verify.py](examples/verify.py) and [examples/extension.py](examples/extension.py) (`--fetch` once to populate the shared S&P 500 panel cache).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
