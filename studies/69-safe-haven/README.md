# Study 69 — Safe-Haven 🪙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does gold hedge inflation? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No: YoY gold vs YoY inflation correlate just **+0.09**, and gold earned the same in high-inflation months (+10.6%) as low (+10.2%) — a +0.4% gap. |
| **Tradability** — does it protect in a crash? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Partly: flat in equity crashes (**−0.2% vs SPY −10.1%**), near-zero stock correlation — but it *rose* in only **50%** of them. Ballast, not a reliable haven. |
| **"Gold hedges inflation"?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | Its +10%/yr was a price bull, not an inflation response (Erb-Harvey's "golden dilemma"). |

> **In one sentence:** gold is a real, uncorrelated diversifier that tends not to fall when stocks do — but the legend oversells it: as a month-to-month inflation hedge it's absent (corr +0.09, no high-inflation premium), and as a crisis haven it's a coin-flip to actually rally (up in 50% of crashes), dependable only to *not crash with* equities.

## What we tested

Gold's two folk jobs. **(1) Inflation hedge:** does gold track inflation, and earn more when inflation is high? We correlate YoY gold returns with YoY US CPI and split returns by high vs low inflation. **(2) Crisis hedge:** does gold protect when equities crash? We measure its stock correlation and its return in equity-crash months (SPY monthly < −8%). Real data is monthly **GLD / SPY / CPI**, 2005–2025. The offline control is a synthetic world where gold tracks a smooth inflation cycle by a tunable amount (and a null where it ignores inflation).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why gold's big returns aren't inflation protection — and what it *is* good for |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the inflation correlation, the high/low split, the crash-month behaviour |

The fingerprinted real-data run (GLD/SPY/CPI, 2005–2025, fp `e41afebbd671`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (reads the shared cross-asset + macro pulls); the offline machinery proof runs on the synthetic world in [safe_haven/data.py](safe_haven/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
